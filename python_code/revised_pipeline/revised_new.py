"""
Streamlined script for fetching Wikipedia article lists and revision snapshots.
Reads mini_articles_by_category.csv (100 samples per category) and fetches revision
snapshots for each article between START and END timestamps at a given frequency.
"""
import os
import pandas as pd
import requests
from dateutil import parser
from parallel import process_batch_with_progress, resilient_api_call, CheckpointManager

# Configuration
MINI_ARTICLES_CSV = "mini_articles_by_category.csv"
OUTPUT_REVS_PICKLE = "revision_snapshots.pkl"
CHECKPOINT_DIR = "checkpoints/revisions"
START_TS = "2022-01-01T00:00:00Z"
END_TS = "2024-01-31T23:59:59Z"
FREQ = "1ME"  # monthly snapshots

# Ensure checkpoint directory exists
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

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

def fetch_revision_snapshots(title: str) -> pd.DataFrame:
    """
    Fetches all revisions for `title` between START_TS and END_TS, then returns
    snapshots at frequency FREQ with full wikitext content.
    Checkpoints each article to avoid re-fetching.
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
    params = {
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
    cont = {}
    while True:
        data = resilient_api_call(wiki_api_call, max_retries=3, **params, **cont)
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
            cont = data["continue"]
        else:
            break

    if not all_revs:
        return pd.DataFrame(columns=["page_title","snapshot_ts","rev_id","timestamp","user","is_bot","content"])

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
        if rev_time == last_time:
            continue
        row = df.iloc[pos].to_dict()
        row['snapshot_ts'] = snap
        row['page_title'] = title
        rows.append(row)
        last_time = rev_time

    df_snap = pd.DataFrame(rows)
    cols = ["page_title","snapshot_ts","rev_id","timestamp","user","is_bot","content"]
    result = df_snap[cols]

    # Checkpoint save
    result.to_pickle(chk_path)
    return result


def main():
    # Load mini-articles sample
    mini_df = pd.read_csv(MINI_ARTICLES_CSV)
    titles = mini_df['title'].unique().tolist()

    # Process in parallel with progress
    rev_dfs = process_batch_with_progress(
        fetch_revision_snapshots,
        titles,
        desc="Fetching revision snapshots",
        use_threads=True,
        cpu_intensive=False,
        max_workers=4,
        batch_size=5
    )

    # Concatenate all results
    all_revs = pd.concat(rev_dfs, ignore_index=True)
    print(f"Fetched {len(all_revs)} revision snapshots for {len(titles)} articles.")

    # Save final DataFrame
    all_revs.to_pickle(OUTPUT_REVS_PICKLE)
    print(f"Saved combined snapshots to {OUTPUT_REVS_PICKLE}")

if __name__ == "__main__":
    main()
