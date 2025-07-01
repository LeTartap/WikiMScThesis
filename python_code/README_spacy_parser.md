# SpaCy Parser and Fetch Code with Improved Error Handling

This module provides a solution to the issues where:
1. The spaCy parsing process was failing after 3 retries in the original pipeline
2. The fetch code was encountering errors during execution

## Problems

### 1. SpaCy Parser Issues

The original code in `optimized_pipeline.ipynb` was using `process_dataframe_parallel` to parse text with spaCy in parallel:

```python
# Parse text with spaCy in parallel
# This is CPU-intensive, so we use processes instead of threads
tiny_revs = process_dataframe_parallel(
    tiny_revs,
    parse_with_spacy,
    column="plain_text",
    new_column="parsed",
    use_threads=False,  # Use processes for CPU-intensive task
    cpu_intensive=True
)

# Save intermediate result
tiny_revs.to_pickle("history_politics_100_articles_every1m_after_spacy_bedfore_delta.pkl")
```

This function was consistently failing after 3 retries, likely due to:
1. Insufficient timeout (300 seconds) for processing large texts
2. Limited number of retries (3)
3. Potential memory issues when processing large documents

### 2. Fetch Code Issues

The fetch code was also encountering errors, particularly in the `process_batch_with_progress` function:

```python
# Process pending articles with improved error handling
if pending_articles:
    results = process_batch_with_progress(
        fetch_article_revisions,
        pending_articles,
        desc="Fetching revisions",
        max_workers=3,  # Reduce concurrent workers
        use_threads=True,
        cpu_intensive=False,
        batch_size=5  # Process in smaller batches
    )
```

The issues with the fetch code included:
1. The `batch_size` parameter was being passed but wasn't defined in the function signature
2. The `resilient_api_call` function didn't have specific handling for network-related exceptions
3. Memory management wasn't optimal when processing large batches of articles

## Solution

Several changes were implemented to fix these issues:

### 1. SpaCy Parser Fixes

1. **Enhanced the `process_batch_with_progress` function in `parallel.py`**:
   - Made timeout and max_retries configurable parameters
   - Added specific handling for TimeoutError and MemoryError
   - Implemented exponential backoff for retries (2^retry_count seconds)
   - Improved error messages with more details

2. **Created a standalone `spacy_parser.py` script** with:
   - The same `parse_with_spacy` function from the notebook
   - A new `parse_dataframe_with_spacy` function with improved parameters:
     - Increased timeout to 900 seconds (15 minutes)
     - Increased max retries to 7
     - Reduced max_workers to 2 to avoid memory issues
   - Command-line interface for running the script directly

### 2. Fetch Code Fixes

1. **Added batch processing to `process_batch_with_progress`**:
   - Added a `batch_size` parameter to the function signature
   - Implemented a new mode that processes items in smaller batches
   - Added memory cleanup between batches using garbage collection and CUDA cache clearing
   - Improved progress tracking and error reporting for batched processing

2. **Enhanced the `resilient_api_call` function**:
   - Added specific handling for network-related exceptions
   - Improved error messages with exception type information
   - Separated handling for network errors vs. other types of errors
   - Added more detailed logging to help diagnose issues

## Usage

### 1. SpaCy Parser Usage

#### From Command Line

```bash
python spacy_parser.py input_file.pkl output_file.pkl [text_column] [output_column]
```

Example:
```bash
python spacy_parser.py mini_history_politics_100_articles_every1m_before_cleaning.pkl history_politics_100_articles_every1m_after_spacy_bedfore_delta.pkl
```

#### From Python/Jupyter

```python
from spacy_parser import parse_dataframe_with_spacy

# Load your DataFrame
df = pd.read_pickle("mini_history_politics_100_articles_every1m_before_cleaning.pkl")

# Parse with improved error handling
result_df = parse_dataframe_with_spacy(df, "plain_text", "parsed")

# Save the result
result_df.to_pickle("history_politics_100_articles_every1m_after_spacy_bedfore_delta.pkl")
```

### 2. Using the Enhanced Parallel Processing

#### For CPU-Intensive Tasks with Batch Processing

```python
from parallel import process_batch_with_progress

# Process items in batches to manage memory better
results = process_batch_with_progress(
    your_function,
    items_to_process,
    desc="Processing items",
    max_workers=3,
    use_threads=False,  # Use processes for CPU-intensive tasks
    cpu_intensive=True,
    timeout=900,  # 15 minutes
    max_retries=7,
    batch_size=5  # Process in batches of 5 items
)
```

#### For DataFrame Processing

```python
from parallel import process_dataframe_parallel

# Use the enhanced parameters
result_df = process_dataframe_parallel(
    df,
    your_function,
    column="your_column",
    new_column="result_column",
    timeout=900,  # 15 minutes
    max_retries=7,
    max_workers=2,
    batch_size=5  # Process in batches of 5 rows
)
```

### 3. Using the Enhanced API Call Function

```python
from parallel import resilient_api_call

def api_function(**params):
    # Your API call implementation
    response = requests.get("https://api.example.com", params=params)
    response.raise_for_status()
    return response.json()

# Make API call with improved error handling
try:
    result = resilient_api_call(
        api_function,
        max_retries=7,
        initial_backoff=1.0,
        backoff_factor=2.0,
        jitter=0.1,
        # Your API parameters:
        param1="value1",
        param2="value2"
    )
    print("API call successful!")
except Exception as e:
    print(f"API call failed after all retries: {str(e)}")
```

## Benefits

1. **More Robust**: The solution is more resilient to timeouts, network errors, and other exceptions
2. **Better Memory Management**: 
   - Reduced workers help avoid memory issues
   - Batch processing allows handling larger datasets with limited memory
   - Memory cleanup between batches prevents memory leaks
3. **Configurable**: 
   - Parameters can be adjusted based on specific needs
   - Timeout, retry count, and batch size can be customized for different workloads
4. **Standalone**: Can be used independently of the notebook
5. **Improved Logging**: 
   - Better error messages help diagnose issues
   - Specific exception handling provides more context about failures
   - Batch progress reporting gives better visibility into long-running processes
6. **Network Resilience**: 
   - Enhanced handling of network-related exceptions
   - Exponential backoff with jitter for API calls
   - Specific handling for different types of network errors
