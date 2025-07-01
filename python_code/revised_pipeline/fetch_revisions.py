"""
Streamlined script for fetching Wikipedia article lists and revision snapshots.
Reads mini_articles_by_category.csv (100 samples per category, with title and pageid)
and fetches revision snapshots for each article between START and END timestamps at a given frequency.
"""
import os
import pandas as pd
import requests
from dateutil import parser
from parallel import process_batch_with_progress, resilient_api_call

# Configuration
ARTICLES_CSV = "category_titles_by_group.csv"  # CSV now contains 'title' and 'pageid'
OUTPUT_REVS_PICKLE = "revision_snapshots.pkl"
CHECKPOINT_DIR = "checkpoints/revisions"
START_TS = "2022-01-01T00:00:00Z"
END_TS = "2024-01-31T23:59:59Z"
FREQ = "1ME"  # monthly snapshots

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Load title->pageid mapping
mini_df = pd.read_csv(ARTICLES_CSV)
# Normalize column names to lowercase
mini_df.columns = [c.strip() for c in mini_df.columns]
cols_lower = [c.lower() for c in mini_df.columns]
# Ensure 'pageid' exists, or attempt to find alternative
if 'pageid' not in cols_lower:
    for alt in ('page_id', 'page id', 'id'):
        if alt in cols_lower:
            orig = mini_df.columns[cols_lower.index(alt)]
            mini_df = mini_df.rename(columns={orig: 'pageid'})
            break
    else:
        raise KeyError(f"Expected column 'pageid' in {ARTICLES_CSV}, found {mini_df.columns.tolist()}")
TITLE_TO_PAGEID = dict(zip(mini_df['title'], mini_df['pageid']))

# Bot detection helper
def is_bot_username(username: str) -> bool:
    return username.lower().endswith("bot")

# API call wrapper
def wiki_api_call(**params):
    session = requests.Session()
    resp = session.get("https://en.wikipedia.org/w/api.php", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

# Fetch snapshots of revisions for one article
def fetch_revision_snapshots(
    title: str,
    carry_forward: bool = True
) -> pd.DataFrame:
    """
    Fetches all revisions for `title` between START_TS and END_TS,
    then returns snapshots at frequency FREQ with full wikitext content.
    Checkpoints each article to avoid re-fetching.

    Args:
        title: page title
        carry_forward: if False, skip snapshot when no new revision since last interval
    Returns:
        DataFrame with columns [page_title, pageid, snapshot_ts, rev_id, timestamp, user, is_bot, content]
    """
    # Prepare checkpoint file per article
    safe_title = title.replace("/", "_")
    chk_path = os.path.join(CHECKPOINT_DIR, f"{safe_title}.pkl")
    if os.path.exists(chk_path):
        try:
            return pd.read_pickle(chk_path)
        except Exception:
            pass

    # Phase 1: fetch all revisions
    base_params = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvprop": "ids|timestamp|user|content",
        "rvstart": END_TS,
        "rvend": START_TS,
        "rvlimit": "max",
        "titles": title,
        "redirects": 1,
        "rvslots": "main",
    }
    all_revs = []
    continue_params = {}
    while True:
        params = {**base_params, **continue_params}
        data = resilient_api_call(wiki_api_call, max_retries=3, **params)
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            for rev in page.get("revisions", []) or []:
                ts = parser.isoparse(rev["timestamp"])
                all_revs.append({
                    "rev_id": rev["revid"],
                    "timestamp": ts,
                    "user": rev.get("user", ""),
                    "is_bot": is_bot_username(rev.get("user", "")),
                    "content": rev.get("slots", {}).get("main", {}).get("*", ""),
                })
        if "continue" in data:
            continue_params = data["continue"]
        else:
            break

    if not all_revs:
        cols = ["page_title", "pageid", "snapshot_ts", "rev_id", "timestamp", "user", "is_bot", "content"]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(all_revs).sort_values("timestamp").reset_index(drop=True)

    # Phase 2: sampling snapshots
    times = df["timestamp"]
    sample_times = pd.date_range(
        start=pd.to_datetime(START_TS),
        end=pd.to_datetime(END_TS),
        freq=FREQ,
        tz=times.dt.tz
    )
    rows = []
    last_time = None
    for snap in sample_times:
        pos = times.searchsorted(snap, side='right') - 1
        if pos < 0:
            continue
        rev_time = times.iloc[pos]
        # if no new revision and carry_forward is False, skip
        if not carry_forward and last_time is not None and rev_time <= last_time:
            continue
        row = df.iloc[pos].to_dict()
        row['snapshot_ts'] = snap
        row['page_title'] = title
        # include pageid
        row['pageid'] = TITLE_TO_PAGEID.get(title)
        rows.append(row)
        last_time = rev_time

    df_snap = pd.DataFrame(rows)
    cols = ["page_title", "pageid", "snapshot_ts", "rev_id", "timestamp", "user", "is_bot", "content"]
    result = df_snap[cols]

    # Checkpoint save
    result.to_pickle(chk_path)
    return result


# def main():
#     titles = list(TITLE_TO_PAGEID.keys())
#
#     # Process in parallel with progress
#     rev_dfs = process_batch_with_progress(
#         fetch_revision_snapshots,
#         titles,
#         desc="Fetching revision snapshots",
#         use_threads=True,
#         cpu_intensive=False,
#         max_workers=4,
#         batch_size=5
#     )
#
#     # Concatenate all results
#     all_revs = pd.concat(rev_dfs, ignore_index=True)
#     print(f"Fetched {len(all_revs)} revision snapshots for {len(titles)} articles.")
#
#     # Save final DataFrame
#     all_revs.to_pickle(OUTPUT_REVS_PICKLE)
#     print(f"Saved combined snapshots to {OUTPUT_REVS_PICKLE}")
#
# if __name__ == "__main__":
#     main()
