# Optimized Wikipedia Article Analysis Pipeline

This document explains the optimizations made to the Wikipedia article analysis pipeline to improve performance, stability, and resilience when processing large datasets.

## Overview of Improvements

The original pipeline in `newPipieline.ipynb` has been optimized in the following ways:

1. **Parallelization**: Key processing steps now run in parallel, utilizing multiple CPU cores
2. **Checkpointing**: Long-running processes can be resumed after interruptions
3. **Error Handling**: Robust error handling prevents crashes on malformed data or API failures
4. **Memory Management**: Improved memory usage, especially for GPU operations
5. **Input Validation**: Comprehensive validation prevents crashes on unexpected inputs

## Files

- `parallel.py`: Contains utilities for parallel processing and resilient operations
- `optimized_pipeline.ipynb`: The optimized version of the original pipeline

## Key Features

### Parallel Processing

The pipeline now uses parallel processing for:
- Fetching articles from categories
- Fetching revision snapshots for articles
- Text cleaning and parsing with spaCy
- Feature computation (delta, readability metrics, vocabulary diversity)

This significantly reduces processing time, especially for large datasets.

### Checkpointing

The pipeline now saves intermediate results at regular intervals, allowing you to resume processing after interruptions. Checkpoints are saved in the `checkpoints/` directory.

### Resilient API Calls

API calls to Wikipedia now include:
- Automatic retries with exponential backoff
- Proper error handling
- Timeout management

This makes the pipeline more robust against network issues and API rate limits.

### GPU Memory Management

GPU operations (like perplexity calculation) now use batch processing with memory cleanup between batches to prevent GPU memory exhaustion.

## Usage

1. Run the cells in `optimized_pipeline.ipynb` sequentially
2. If processing is interrupted, simply re-run the notebook - it will resume from the last checkpoint
3. Adjust batch sizes and worker counts as needed for your hardware

## Configuration Options

You can adjust these parameters in the notebook:

- `sample_size`: Number of articles to process (adjust based on available time/resources)
- `max_workers`: Maximum number of parallel workers (default: auto-detected based on CPU cores)
- `batch_size`: For GPU operations, controls memory usage (smaller = less memory but slower)
- `save_interval`: How often to save checkpoints (in number of processed items)

## Performance Tips

1. For CPU-intensive tasks (spaCy parsing, readability metrics):
   - Use processes instead of threads
   - Leave at least one CPU core free for system operations

2. For I/O-bound tasks (API calls, file operations):
   - Use threads instead of processes
   - You can use more workers than CPU cores

3. For GPU operations:
   - Adjust batch size based on available GPU memory
   - Clear GPU cache between batches

## Error Handling

The pipeline now handles these error conditions gracefully:

- Network failures during API calls
- Malformed or unexpected data
- Memory limitations
- Invalid inputs

Errors are logged but don't crash the pipeline - failed items are skipped and processing continues.

## Comparison with Original Pipeline

The optimized pipeline maintains the same functionality as the original but adds:

1. **Speed**: Parallel processing makes it significantly faster
2. **Stability**: Better error handling prevents crashes
3. **Resumability**: Checkpointing allows resuming after interruptions
4. **Scalability**: Can handle larger datasets without running out of memory

These improvements make the pipeline suitable for processing larger datasets over longer periods without supervision.