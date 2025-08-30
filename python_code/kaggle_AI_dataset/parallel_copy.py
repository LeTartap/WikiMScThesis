"""
Parallel processing utilities for the Wikipedia article analysis pipeline.
This module provides functions for parallelizing various parts of the data processing pipeline,
including article fetching, text parsing, and feature computation.
"""

import concurrent.futures
import multiprocessing
import pandas as pd
import numpy as np
import time
import os
import traceback
import psutil
from tqdm.auto import tqdm
from functools import partial
import torch

def get_spacy_model():
    # Use a global variable, but only inside each worker
    # This is safe inside a process, not between processes!
    if not hasattr(get_spacy_model, "nlp"):
        get_spacy_model.nlp = spacy.load("en_core_web_sm")
    return get_spacy_model.nlp

# Function to check available system memory
def check_memory_usage():
    """
    Check the current memory usage of the system.

    Returns:
        tuple: (memory_percent, available_gb)
            - memory_percent: Percentage of memory used (0-100)
            - available_gb: Available memory in GB
    """
    try:
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        available_gb = memory.available / (1024 ** 3)  # Convert to GB
        return memory_percent, available_gb
    except Exception as e:
        print(f"Error checking memory: {str(e)}")
        return 0, 0

# Determine the optimal number of workers based on CPU cores
def get_optimal_workers(cpu_intensive=True):
    """
    Determine the optimal number of worker processes/threads.

    Args:
        cpu_intensive (bool): If True, use fewer workers for CPU-intensive tasks.
                             If False, use more workers for I/O-bound tasks.

    Returns:
        int: Recommended number of workers
    """
    cpu_count = multiprocessing.cpu_count()

    if cpu_intensive:
        # For CPU-intensive tasks, use N-1 workers (leave one core free)
        return max(1, cpu_count - 1)
    else:
        # For I/O-bound tasks, can use more workers
        return cpu_count * 2

# Function to process a batch of items with progress tracking
def process_batch_with_progress(process_func, items, desc="Processing", 
                               max_workers=None, use_threads=False, 
                               cpu_intensive=True, timeout=600, max_retries=5, 
                               batch_size=None, **kwargs):
    """
    Process a batch of items in parallel with progress tracking.

    Args:
        process_func: Function to apply to each item
        items: List of items to process
        desc: Description for progress bar
        max_workers: Maximum number of workers (processes/threads)
        use_threads: If True, use threads instead of processes
        cpu_intensive: If True, optimize for CPU-intensive tasks
        timeout: Timeout in seconds for each item processing
        max_retries: Maximum number of retry attempts
        batch_size: If provided, process items in batches of this size to manage memory better.
                   Each batch will be processed with its own executor, and memory will be
                   cleared between batches.
        **kwargs: Additional arguments to pass to process_func
    """
    if max_workers is None:
        # Reduce default max_workers for processes to prevent resource exhaustion
        if cpu_intensive and not use_threads:
            max_workers = min(get_optimal_workers(cpu_intensive), 2)  # More conservative limit for processes
        else:
            max_workers = min(get_optimal_workers(cpu_intensive), 4)  # Limit max workers

    if kwargs:
        func = partial(process_func, **kwargs)
    else:
        func = process_func

    # Set a more conservative timeout for processes to prevent abrupt termination
    if not use_threads and timeout > 300:
        timeout = 300  # 5 minutes max for processes

    executor_class = concurrent.futures.ThreadPoolExecutor if use_threads else concurrent.futures.ProcessPoolExecutor

    results = []
    retries = max_retries  # Configurable retry mechanism

    # Process in batches if batch_size is specified
    if batch_size and batch_size > 0:
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        total_items = len(items)
        processed_count = 0

        with tqdm(total=total_items, desc=desc) as pbar:
            for batch_idx, batch in enumerate(batches):
                # Check memory usage before processing batch
                if not use_threads:  # Only check for processes, not threads
                    memory_percent, available_gb = check_memory_usage()
                    print(f"Memory usage before batch {batch_idx+1}: {memory_percent:.1f}% used, {available_gb:.2f} GB available")

                    # Adjust max_workers based on available memory
                    if memory_percent > 85:  # High memory usage
                        adjusted_workers = 1  # Use only 1 worker when memory is very low
                        print(f"WARNING: High memory usage detected! Reducing workers to {adjusted_workers}")
                    elif memory_percent > 70:  # Moderate memory usage
                        adjusted_workers = max(1, max_workers // 2)
                        print(f"Memory usage is high. Reducing workers to {adjusted_workers}")
                    else:
                        adjusted_workers = max_workers

                    # Update max_workers for this batch
                    current_max_workers = adjusted_workers
                else:
                    current_max_workers = max_workers

                print(f"Processing batch {batch_idx+1}/{len(batches)} ({len(batch)} items) with {current_max_workers} workers")
                batch_results = []

                # Track which items in the batch have been processed
                processed_items = set()

                # Try to process the batch, with handling for process pool errors
                try:
                    with executor_class(max_workers=current_max_workers) as executor:
                        # Submit all tasks
                        future_to_idx = {executor.submit(func, item): i for i, item in enumerate(batch)}

                        # Process completed futures
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            retry_count = 0

                            # Mark this item as processed (even if it fails)
                            processed_items.add(idx)

                            while retry_count < retries:
                                try:
                                    result = future.result(timeout=timeout)  # Configurable timeout
                                    batch_results.append((idx, result))
                                    break
                                except concurrent.futures.TimeoutError:
                                    retry_count += 1
                                    if retry_count == retries:
                                        print(f"Timeout processing item {idx} in batch {batch_idx+1} after {retries} retries (timeout={timeout}s)")
                                        batch_results.append((idx, None))
                                    else:
                                        print(f"Timeout on attempt {retry_count}/{retries} for item {idx} in batch {batch_idx+1}. Retrying...")
                                        # Exponential backoff for retries
                                        time.sleep(2 ** retry_count)  # 2, 4, 8, 16... seconds
                                except MemoryError:
                                    print(f"Memory error processing item {idx} in batch {batch_idx+1}. Skipping.")
                                    batch_results.append((idx, None))
                                    break  # Don't retry on memory errors
                                except Exception as e:
                                    retry_count += 1
                                    if retry_count == retries:
                                        print(f"Error processing item {idx} in batch {batch_idx+1} after {retries} retries: {str(e)}")
                                        batch_results.append((idx, None))
                                    else:
                                        print(f"Retry {retry_count}/{retries} for item {idx} in batch {batch_idx+1}: {str(e)}")
                                        # Exponential backoff for retries
                                        time.sleep(2 ** retry_count)  # 2, 4, 8, 16... seconds

                            pbar.update(1)
                            processed_count += 1

                except concurrent.futures.process.BrokenProcessPool as e:
                    print(f"Process pool broken in batch {batch_idx+1}: {str(e)}")
                    print("This usually happens when a worker process is terminated unexpectedly.")
                    print("Continuing with remaining items in a new process pool...")

                    # Process remaining items in the batch that weren't processed yet
                    remaining_items = [item for i, item in enumerate(batch) if i not in processed_items]
                    if remaining_items:
                        print(f"Processing {len(remaining_items)} remaining items in batch {batch_idx+1}")

                        # Process remaining items with a new executor, but with more conservative settings
                        try:
                            # Check memory again before creating new executor
                            if not use_threads:
                                memory_percent, available_gb = check_memory_usage()
                                print(f"Memory usage before recovery: {memory_percent:.1f}% used, {available_gb:.2f} GB available")

                                # Use even more conservative settings for recovery
                                if memory_percent > 80:
                                    recovery_workers = 1
                                else:
                                    recovery_workers = max(1, current_max_workers // 2)

                                print(f"Using {recovery_workers} workers for recovery")
                            else:
                                recovery_workers = max(1, current_max_workers // 2)

                            with executor_class(max_workers=recovery_workers) as new_executor:
                                for i, item in enumerate(remaining_items):
                                    local_idx = len(processed_items) + i
                                    try:
                                        # Process one at a time with a shorter timeout
                                        result = new_executor.submit(func, item).result(timeout=min(timeout, 120))
                                        batch_results.append((local_idx, result))
                                    except Exception as e:
                                        print(f"Error processing remaining item {local_idx} in batch {batch_idx+1}: {str(e)}")
                                        batch_results.append((local_idx, None))

                                    pbar.update(1)
                                    processed_count += 1
                        except Exception as e:
                            print(f"Failed to process remaining items in batch {batch_idx+1}: {str(e)}")
                            # Add None results for any remaining items
                            for i in range(len(processed_items), len(batch)):
                                batch_results.append((i, None))
                                pbar.update(1)
                                processed_count += 1

                # Map batch indices to global indices
                batch_start = batch_idx * batch_size
                for local_idx, result in batch_results:
                    global_idx = batch_start + local_idx
                    results.append((global_idx, result))

                # Clear memory between batches
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    else:
        # Implementation for processing without batches, with improved error handling
        processed_indices = set()

        # Check memory usage before processing
        if not use_threads:  # Only check for processes, not threads
            memory_percent, available_gb = check_memory_usage()
            print(f"Memory usage before processing: {memory_percent:.1f}% used, {available_gb:.2f} GB available")

            # Adjust max_workers based on available memory
            if memory_percent > 85:  # High memory usage
                adjusted_workers = 1  # Use only 1 worker when memory is very low
                print(f"WARNING: High memory usage detected! Reducing workers to {adjusted_workers}")
            elif memory_percent > 70:  # Moderate memory usage
                adjusted_workers = max(1, max_workers // 2)
                print(f"Memory usage is high. Reducing workers to {adjusted_workers}")
            else:
                adjusted_workers = max_workers

            # Update max_workers
            current_max_workers = adjusted_workers
        else:
            current_max_workers = max_workers

        print(f"Processing {len(items)} items with {current_max_workers} workers")

        with tqdm(total=len(items), desc=desc) as pbar:
            try:
                with executor_class(max_workers=current_max_workers) as executor:
                    # Submit all tasks
                    future_to_idx = {executor.submit(func, item): i for i, item in enumerate(items)}

                    # Process completed futures
                    for future in concurrent.futures.as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        retry_count = 0

                        # Mark this item as processed
                        processed_indices.add(idx)

                        while retry_count < retries:
                            try:
                                result = future.result(timeout=timeout)  # Configurable timeout
                                results.append((idx, result))
                                break
                            except concurrent.futures.TimeoutError:
                                retry_count += 1
                                if retry_count == retries:
                                    print(f"Timeout processing item {idx} after {retries} retries (timeout={timeout}s)")
                                    results.append((idx, None))
                                else:
                                    print(f"Timeout on attempt {retry_count}/{retries} for item {idx}. Retrying...")
                                    # Exponential backoff for retries
                                    time.sleep(2 ** retry_count)  # 2, 4, 8, 16... seconds
                            except MemoryError:
                                print(f"Memory error processing item {idx}. Skipping.")
                                results.append((idx, None))
                                break  # Don't retry on memory errors
                            except Exception as e:
                                retry_count += 1
                                if retry_count == retries:
                                    print(f"Error processing item {idx} after {retries} retries: {str(e)}")
                                    results.append((idx, None))
                                else:
                                    print(f"Retry {retry_count}/{retries} for item {idx}: {str(e)}")
                                    # Exponential backoff for retries
                                    time.sleep(2 ** retry_count)  # 2, 4, 8, 16... seconds

                        pbar.update(1)

            except concurrent.futures.process.BrokenProcessPool as e:
                print(f"Process pool broken: {str(e)}")
                print("This usually happens when a worker process is terminated unexpectedly.")
                print("Continuing with remaining items in a new process pool...")

                # Process remaining items that weren't processed yet
                remaining_indices = [i for i in range(len(items)) if i not in processed_indices]
                if remaining_indices:
                    print(f"Processing {len(remaining_indices)} remaining items")

                    # Process remaining items with a new executor, but with more conservative settings
                    try:
                        # Use half the workers and process one at a time
                        with executor_class(max_workers=max(1, max_workers // 2)) as new_executor:
                            for idx in remaining_indices:
                                try:
                                    # Process one at a time with a shorter timeout
                                    result = new_executor.submit(func, items[idx]).result(timeout=min(timeout, 120))
                                    results.append((idx, result))
                                except Exception as e:
                                    print(f"Error processing remaining item {idx}: {str(e)}")
                                    results.append((idx, None))

                                pbar.update(1)
                    except Exception as e:
                        print(f"Failed to process remaining items: {str(e)}")
                        # Add None results for any remaining items
                        for idx in remaining_indices:
                            results.append((idx, None))
                            pbar.update(1)

    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]

# Function to process DataFrame rows in parallel
def process_dataframe_parallel(df, process_func, column=None, new_column=None, 
                              max_workers=None, use_threads=False, 
                              cpu_intensive=True, batch_size=None, **kwargs):
    """
    Apply a function to each row or column value of a DataFrame in parallel.

    Args:
        df (pandas.DataFrame): DataFrame to process
        process_func (callable): Function to apply to each row or column value
        column (str, optional): If provided, process only this column's values
        new_column (str or list, optional): Name(s) for the new column(s) to store results
        max_workers (int, optional): Maximum number of workers
        use_threads (bool): If True, use threads instead of processes
        cpu_intensive (bool): If True, optimize for CPU-intensive tasks
        batch_size (int, optional): Size of batches to process. If None, determined automatically.
        **kwargs: Additional arguments to pass to process_func

    Returns:
        pandas.DataFrame: DataFrame with processed results
    """
    # Make a copy to avoid modifying the original
    result_df = df.copy()

    if column is not None:
        # Process a specific column's values
        items = df[column].tolist()
        desc = f"Processing {column}"
    else:
        # Process entire rows
        items = [row for _, row in df.iterrows()]
        desc = "Processing rows"

    # For CPU-intensive tasks using processes, enable automatic batching to prevent memory issues
    if cpu_intensive and not use_threads and batch_size is None:
        # Determine a reasonable batch size based on the number of items
        if len(items) > 1000:
            batch_size = 100
        elif len(items) > 100:
            batch_size = 20
        elif len(items) > 10:
            batch_size = 5
        else:
            batch_size = 2

        print(f"Using automatic batch size of {batch_size} for CPU-intensive processing")

    # Process items in parallel
    results = process_batch_with_progress(
        process_func, items, desc=desc, 
        max_workers=max_workers, use_threads=use_threads,
        cpu_intensive=cpu_intensive, batch_size=batch_size, **kwargs
    )

    # Handle results
    if new_column is not None:
        if isinstance(new_column, list):
            # Multiple return values expected
            result_arrays = list(zip(*[r for r in results if r is not None]))
            for i, col_name in enumerate(new_column):
                if i < len(result_arrays):
                    result_df[col_name] = pd.Series(result_arrays[i], index=result_df.index)
        else:
            # Single return value expected
            result_df[new_column] = pd.Series(results, index=result_df.index)

    return result_df

# Function to chunk a DataFrame for batch processing
def chunk_dataframe(df, chunk_size):
    """
    Split a DataFrame into chunks for batch processing.

    Args:
        df (pandas.DataFrame): DataFrame to chunk
        chunk_size (int): Size of each chunk

    Returns:
        list: List of DataFrame chunks
    """
    return [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]

# Function to handle GPU memory management for batch processing
def gpu_batch_process(process_func, items, batch_size=1, desc="GPU Processing", **kwargs):
    """
    Process items in batches to manage GPU memory efficiently.

    Args:
        process_func (callable): Function to apply to each item (should use GPU)
        items (list): List of items to process
        batch_size (int): Number of items to process in each batch
        desc (str): Description for the progress bar
        **kwargs: Additional arguments to pass to process_func

    Returns:
        list: Results from processing each item
    """
    results = []

    # Process in batches with progress tracking
    with tqdm(total=len(items), desc=desc) as pbar:
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]

            try:
                # Process the batch
                batch_results = [process_func(item, **kwargs) for item in batch]
                results.extend(batch_results)

                # Clear GPU cache if using PyTorch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing batch {i//batch_size}: {str(e)}")
                traceback.print_exc()
                # Add None for failed items
                results.extend([None] * len(batch))

                # Try to recover GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.update(len(batch))

    return results

# Function for resilient API calls with retries and backoff
def resilient_api_call(api_func, max_retries=5, initial_backoff=1.0, 
                      backoff_factor=2.0, jitter=0.1, **kwargs):
    """
    Make an API call with exponential backoff retry logic for resilience.

    Args:
        api_func (callable): Function that makes the API call
        max_retries (int): Maximum number of retry attempts
        initial_backoff (float): Initial backoff time in seconds
        backoff_factor (float): Multiplier for backoff time after each retry
        jitter (float): Random factor to add to backoff time (0-1)
        **kwargs: Arguments to pass to the API function

    Returns:
        The result of the API call if successful

    Raises:
        Exception: The last exception encountered if all retries fail
    """
    import requests
    from urllib3.exceptions import HTTPError as URLLibHTTPError

    retry_count = 0
    backoff_time = initial_backoff
    last_exception = None

    # Define which exceptions should trigger a retry
    retry_exceptions = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.HTTPError,
        requests.exceptions.RequestException,
        URLLibHTTPError,
        ConnectionResetError,
        ConnectionError,
        TimeoutError,
    )

    while retry_count < max_retries:
        try:
            return api_func(**kwargs)
        except retry_exceptions as e:
            # Network-related errors - always retry these
            last_exception = e
            retry_count += 1

            if retry_count >= max_retries:
                print(f"Network error after {max_retries} retries: {str(e)}")
                break

            # Calculate backoff time with jitter
            jitter_amount = backoff_time * jitter * np.random.random()
            sleep_time = backoff_time + jitter_amount

            print(f"Network error (attempt {retry_count}/{max_retries}). "
                  f"Retrying in {sleep_time:.2f}s. Error: {str(e)}")

            time.sleep(sleep_time)
            backoff_time *= backoff_factor
        except Exception as e:
            # Other exceptions - log more details
            last_exception = e
            retry_count += 1

            if retry_count >= max_retries:
                print(f"API call failed after {max_retries} retries. Error: {str(e)}")
                break

            # Calculate backoff time with jitter
            jitter_amount = backoff_time * jitter * np.random.random()
            sleep_time = backoff_time + jitter_amount

            print(f"API call failed (attempt {retry_count}/{max_retries}). "
                  f"Retrying in {sleep_time:.2f}s. Error type: {type(e).__name__}, Message: {str(e)}")

            time.sleep(sleep_time)
            backoff_time *= backoff_factor

    # If we get here, all retries failed
    raise last_exception

# Checkpoint manager for long-running processes
class CheckpointManager:
    """
    Manages checkpoints for long-running processes to enable resuming after failures.
    """

    def __init__(self, checkpoint_path, save_interval=100):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_path (str): Path to save checkpoint files
            save_interval (int): How often to save checkpoints (in number of items processed)
        """
        self.checkpoint_path = checkpoint_path
        self.save_interval = save_interval
        self.processed_count = 0
        self.results = []
        self.completed_indices = set()

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        # Try to load existing checkpoint
        self.load_checkpoint()

    def load_checkpoint(self):
        """Load the checkpoint if it exists."""
        if os.path.exists(self.checkpoint_path):
            try:
                checkpoint = pd.read_pickle(self.checkpoint_path)
                self.results = checkpoint.get('results', [])
                self.completed_indices = set(checkpoint.get('completed_indices', []))
                self.processed_count = len(self.completed_indices)
                print(f"Loaded checkpoint with {self.processed_count} processed items")
                return True
            except Exception as e:
                print(f"Error loading checkpoint: {str(e)}")
        return False

    def save_checkpoint(self, force=False):
        """
        Save the current state to the checkpoint file.

        Args:
            force (bool): If True, save regardless of the save_interval
        """
        if force or (self.processed_count % self.save_interval == 0 and self.processed_count > 0):
            checkpoint = {
                'results': self.results,
                'completed_indices': list(self.completed_indices)
            }
            try:
                pd.to_pickle(checkpoint, self.checkpoint_path)
                print(f"Saved checkpoint with {self.processed_count} processed items")
            except Exception as e:
                print(f"Error saving checkpoint: {str(e)}")

    def add_result(self, index, result):
        """
        Add a processed result to the checkpoint.

        Args:
            index (int): Index of the processed item
            result: Result of processing the item
        """
        if index not in self.completed_indices:
            self.results.append((index, result))
            self.completed_indices.add(index)
            self.processed_count += 1
            self.save_checkpoint()

    def get_pending_indices(self, total_count):
        """
        Get indices of items that still need to be processed.

        Args:
            total_count (int): Total number of items

        Returns:
            list: Indices of items that need processing
        """
        all_indices = set(range(total_count))
        return list(all_indices - self.completed_indices)

    def get_results(self):
        """
        Get all results in the original order.

        Returns:
            list: Results in order of original indices
        """
        sorted_results = sorted(self.results, key=lambda x: x[0])
        return [r[1] for r in sorted_results]
