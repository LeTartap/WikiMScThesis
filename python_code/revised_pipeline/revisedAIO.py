# Import required libraries
import requests
import pandas as pd
import spacy
from textstat import textstat
import re
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
import pickle
import os
import time
from tqdm.auto import tqdm
from dateutil import parser
import numpy as np

# Import our parallel processing utilities
from parallel import (
    process_batch_with_progress,
    process_dataframe_parallel,
    gpu_batch_process,
    resilient_api_call,
    CheckpointManager
)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load GPT-2 model
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
if torch.cuda.is_available():
    print("CUDA available! Moving model to GPU...")
    model.to("cuda")
else:
    print('CUDA not available! Using CPU (this will be slow).')


def get_category_members(category, cmtype="page", namespace=0, limit=500):
    """
    Fetches members of a given Wikipedia category with improved error handling.

    Args:
        category: Category name without the 'Category:' prefix
        cmtype: 'page', 'subcat', or 'file'
        namespace: Namespace index (0 for articles)
        limit: Number of results per request (max 500 for users)

    Returns:
        List of dicts with 'pageid' and 'title'
    """

    def api_call(**params):
        S = requests.Session()
        res = S.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
        res.raise_for_status()
        return res.json()

    members = []
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmtype": cmtype,
        "cmlimit": limit,
        "cmnamespace": namespace,
    }

    while True:
        try:
            # Use our resilient API call function with retries
            data = resilient_api_call(api_call, max_retries=5, **params)
            batch = data.get("query", {}).get("categorymembers", [])
            members.extend(batch)

            if "continue" in data:
                params.update(data["continue"])
            else:
                break
        except Exception as e:
            print(f"Error fetching category members for {category}: {str(e)}")
            break

    return members

def fetch_articles_from_categories(categories, include_subcats=False, max_subcat_depth=1, checkpoint_path=None):
    """
    Given a list of category names, fetches all articles in them with checkpointing.
    Optionally includes pages from subcategories up to specified depth.

    Args:
        categories: List of category names (strings without prefix)
        include_subcats: Whether to traverse into subcategories
        max_subcat_depth: Maximum depth for subcategory traversal
        checkpoint_path: Path to save checkpoint data

    Returns:
        Set of page titles
    """
    # Initialize checkpoint manager if path provided
    checkpoint_mgr = None
    if checkpoint_path:
        checkpoint_mgr = CheckpointManager(checkpoint_path, save_interval=10)
        # If we have completed results, return them
        if checkpoint_mgr.processed_count == len(categories):
            all_articles = set()
            for result in checkpoint_mgr.get_results():
                all_articles.update(result)
            return all_articles

    all_articles = set()
    seen_cats = set()

    def _recurse(cat, depth):
        if cat in seen_cats or depth < 0:
            return set()
        seen_cats.add(cat)

        cat_articles = set()

        # Fetch pages
        pages = get_category_members(cat, cmtype="page")
        for p in pages:
            cat_articles.add(p["title"])

        if include_subcats and depth > 0:
            subcats = get_category_members(cat, cmtype="subcat", namespace=14)
            for sc in subcats:
                sc_name = sc["title"].replace("Category:", "")
                subcat_articles = _recurse(sc_name, depth - 1)
                cat_articles.update(subcat_articles)

        return cat_articles

    # Process categories in parallel
    def process_category(cat):
        try:
            return _recurse(cat, max_subcat_depth)
        except Exception as e:
            print(f"Error processing category {cat}: {str(e)}")
            return set()

    # Get pending categories if using checkpoints
    if checkpoint_mgr:
        pending_indices = checkpoint_mgr.get_pending_indices(len(categories))
        pending_categories = [categories[i] for i in pending_indices]
        print(f"Processing {len(pending_categories)} pending categories out of {len(categories)} total")
    else:
        pending_categories = categories

    # Process categories in parallel
    results = process_batch_with_progress(
        process_category,
        pending_categories,
        desc="Traversing categories",
        use_threads=True,  # Use threads for I/O-bound tasks
        cpu_intensive=False
    )

    # Save results to checkpoint if using checkpoints
    if checkpoint_mgr:
        for i, result in enumerate(results):
            idx = pending_indices[i] if pending_indices else i
            checkpoint_mgr.add_result(idx, result)

        # Get all results including previously checkpointed ones
        all_results = checkpoint_mgr.get_results()
    else:
        all_results = results

    # Combine all article sets
    for article_set in all_results:
        if article_set:  # Skip None results from errors
            all_articles.update(article_set)

    return all_articles

def fetch_revision_snapshots(
        title: str,
        start_ts: str,
        end_ts: str,
        freq: str = "7D",
        bot_test_fn: callable = None,
        carry_forward: bool = True,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        checkpoint_path: str = None,
) -> pd.DataFrame:
    """
    Fetches all revisions (including full content) for `title` between start_ts and end_ts,
    then returns snapshots at given frequency with content included.

    This version includes improved error handling and checkpointing.

    Args:
        title: Wikipedia page title
        start_ts: ISO8601 timestamp string, e.g. "2020-01-01T00:00:00Z"
        end_ts: ISO8601 timestamp string, e.g. "2020-12-31T23:59:59Z"
        freq: pandas offset alias (e.g. "7D")
        bot_test_fn: callable to flag bots (user->bool)
        carry_forward: reuse last snapshot if no newer revision
        max_retries: number of retries on network failure
        backoff_factor: multiplier for retry backoff in seconds
        checkpoint_path: path to pickle intermediate results

    Returns:
        DataFrame with columns ['page_title','snapshot_ts','rev_id','timestamp',
        'user','is_bot','content']
    """
    # Check if we have a checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            return pd.read_pickle(checkpoint_path)
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")

    def api_call(**params):
        session = requests.Session()
        resp = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    # Phase 1: Fetch metadata + content in one pass
    params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "ids|timestamp|user|content",
        "rvstart": end_ts,
        "rvend": start_ts,
        "rvlimit": "max",
        "titles": title,
        "redirects": 1,
        "rvslots": "main",
    }

    all_revs = []

    try:
        while True:
            # Use our resilient API call function
            data = resilient_api_call(
                api_call,
                max_retries=max_retries,
                initial_backoff=backoff_factor,
                **params
            )

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                for rev in page.get("revisions", []) or []:
                    ts_str = rev.get("timestamp")
                    if not ts_str:
                        continue
                    ts = parser.isoparse(ts_str)
                    user = rev.get("user", "")
                    # content slot
                    content = rev.get("slots", {}).get("main", {}).get("*", "")
                    all_revs.append({
                        "rev_id": rev.get("revid"),
                        "timestamp": ts,
                        "user": user,
                        "is_bot": bot_test_fn(user) if bot_test_fn else False,
                        "content": content,
                    })

            if "continue" in data:
                params.update(data["continue"])
            else:
                break
    except Exception as e:
        print(f"Error fetching revisions for {title}: {str(e)}")
        # If we have some revisions, continue with what we have
        if not all_revs:
            return pd.DataFrame(columns=["page_title", "snapshot_ts", "rev_id",
                                         "timestamp", "user", "is_bot", "content"])

    # If no revisions, return empty DataFrame
    if not all_revs:
        return pd.DataFrame(columns=["page_title", "snapshot_ts", "rev_id",
                                     "timestamp", "user", "is_bot", "content"])

    # Build DataFrame and sort
    df = pd.DataFrame(all_revs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Phase 2: Snapshot selection
    timestamps = df["timestamp"]
    sample_times = pd.date_range(
        start=pd.to_datetime(start_ts),
        end=pd.to_datetime(end_ts),
        freq=freq,
        tz=timestamps.dt.tz
    )

    snaps = []
    last_snap_time = None

    for snap_t in sample_times:
        pos = timestamps.searchsorted(snap_t, side='right') - 1
        if pos < 0:
            continue
        rev_time = timestamps.iloc[pos]
        # skip if no carry_forward and no new revision
        if not carry_forward and last_snap_time is not None and rev_time <= last_snap_time:
            last_snap_time = snap_t
            continue
        row = df.iloc[pos].to_dict()
        row["snapshot_ts"] = snap_t
        row["page_title"] = title
        snaps.append(row)
        last_snap_time = snap_t

    df_snap = pd.DataFrame(snaps)
    cols = ["page_title", "snapshot_ts", "rev_id", "timestamp", "user", "is_bot", "content"]
    result_df = df_snap[cols] if not df_snap.empty else pd.DataFrame(columns=cols)

    # Save checkpoint if path provided
    if checkpoint_path and not result_df.empty:
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            result_df.to_pickle(checkpoint_path)
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")

    return result_df

# Modify parallel processing configuration
from parallel import process_batch_with_progress

# Define categories to fetch
# categories = [
#     "Politics",
#     "History"
# ]

# Create checkpoint directory
os.makedirs("checkpoints", exist_ok=True)

# Configure more conservative parallel processing settings
PARALLEL_CONFIG = {
    'max_workers': 4,  # Reduce number of concurrent workers
    'use_threads': True,  # Use threads instead of processes for API calls
    'cpu_intensive': False,
    'timeout': 300,  # 5 minutes timeout
    'chunk_size': 5  # Process in smaller chunks
}


# Fetch articles with improved error handling
def fetch_with_retries(category, **kwargs):
    try:
        results = fetch_articles_from_categories(
            [category],
            include_subcats=True,
            max_subcat_depth=2
        )
        return results
    except Exception as e:
        print(f"Error processing category {category}: {str(e)}")
        return set()


# Process categories with more robust error handling
articles = set()
results = process_batch_with_progress(
    fetch_with_retries,
    categories,
    desc="Processing categories",
    **PARALLEL_CONFIG
)

# Combine results
for result in results:
    if result:
        articles.update(result)

print(f"Fetched {len(articles)} articles from categories and subcategories.")

# Save to file with error handling
try:
    with open("FINAL_NO_RECURSE_article_names_list.pkl", "wb") as f:
        pickle.dump(articles, f)
    print("Successfully saved articles list")
except Exception as e:
    print(f"Error saving articles list: {str(e)}")

# Load the articles from the saved file if needed
try:
    with open("article_names_list.pkl", "rb") as f:
        articles = pickle.load(f)
    print(f"Loaded {len(articles)} articles from file.")
except FileNotFoundError:
    print("Article list file not found.")


# load from all_articles_by_category.csv if available
try:
    articles_df = pd.read_csv("all_articles_by_category.csv")
    articles = set(articles_df["title"].tolist())
    print(f"Loaded {len(articles)} articles from CSV.")
except FileNotFoundError:
    print("CSV file not found. Using previously fetched articles.")

articles_df.head()

#sample 100 articles per category
mini_articles_df = articles_df.groupby("root").apply(lambda x: x.sample(100)).reset_index(drop=True)
# remove duplicates
mini_articles_df = mini_articles_df.drop_duplicates(subset=["title"])

mini_articles_df.to_csv("mini_articles_by_category.csv", index=False)

# # print all categories and their article counts
# for root, group in articles_df.groupby("root"):
#     print(f"{root}: {len(group)} articles")
# same for mini_articles_df
for root, group in mini_articles_df.groupby("root"):
    print(f"{root}: {len(group)} articles")


# Define time range
START_TIMESTAMP = "2022-01-01T00:00:00Z"
END_TIMESTAMP = "2024-01-31T23:59:59Z"

# Define columns for the DataFrame
columns = [
    "page_title", "rev_id", "timestamp", "user", "is_bot", "content"
]


# Function to check if a username is a bot
def is_bot_username(username: str) -> bool:
    return username.lower().endswith("bot")


# Function to fetch revisions for a single article with checkpointing
def fetch_article_revisions(article, **kwargs):
    checkpoint_path = f"checkpoints/revisions/{article.replace('/', '_')}.pkl"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    try:
        revs = fetch_revision_snapshots(
            article,
            START_TIMESTAMP,
            END_TIMESTAMP,
            freq="1ME",
            bot_test_fn=is_bot_username,
            carry_forward=False,
            checkpoint_path=checkpoint_path
        )
        return revs
    except Exception as e:
        print(f"Error fetching revisions for {article}: {str(e)}")
        return pd.DataFrame(columns=columns)



sample_pages = mini_articles_df["title"].tolist()

# Modified checkpoint handling
checkpoint_mgr = CheckpointManager(
    "checkpoints/article_processing.pkl",
    save_interval=2  # Save more frequently
)

# Get pending articles with error handling
try:
    pending_indices = checkpoint_mgr.get_pending_indices(len(sample_pages))
    pending_articles = [sample_pages[i] for i in pending_indices]
    print(f"Processing {len(pending_articles)} pending articles out of {len(sample_pages)} total")
except Exception as e:
    print(f"Error getting pending articles: {str(e)}")
    pending_articles = sample_pages
    pending_indices = list(range(len(sample_pages)))

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

    # Save results with error handling
    for i, result in enumerate(results):
        try:
            if result is not None:
                idx = pending_indices[i]
                checkpoint_mgr.add_result(idx, result)
        except Exception as e:
            print(f"Error saving result {i}: {str(e)}")
            continue

# Combine results with error handling
try:
    all_results = checkpoint_mgr.get_results()
    all_dfs = [df for df in all_results if df is not None and not df.empty]

    if all_dfs:
        tiny_revs = pd.concat(all_dfs, ignore_index=True)
        print(f"Combined DataFrame has {len(tiny_revs)} rows")
    else:
        columns = ["page_title", "snapshot_ts", "rev_id", "timestamp", "user", "is_bot", "content"]
        tiny_revs = pd.DataFrame(columns=columns)
        print("No revisions found")
except Exception as e:
    print(f"Error combining results: {str(e)}")
