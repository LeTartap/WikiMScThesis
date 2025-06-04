#!/usr/bin/env python3
"""
vocabulary_spike_pipeline.py  —  tiny‑data LLM‑style spike detector
======================================================================
Fully self‑contained prototype that streams small batches of Wikipedia
revisions (JSONL *or* live MediaWiki API), fits a pre‑ChatGPT baseline, and
flags vocabulary spikes.

This patch fixes two column‑naming bugs uncovered during notebook testing:
* `build_counts()` now always returns a DataFrame with **explicit `month` and
  `word` columns**, regardless of index layout.
* `fit_baseline()` robustly resets the index if `month` is not yet a column,
  so `KeyError("month")` can no longer occur.

-------------------------------------------------------------
Usage (notebook):
-------------------------------------------------------------
from vocabulary_spike_pipeline import Config, run_pipeline
cfg = Config(
    fetch_pages=["ChatGPT", "OpenAI"],
    fetch_start="2022-12-01", fetch_end="2023-01-07",
    fetch_revs_per_page=10,
    baseline_start="2022-12", baseline_end="2022-12",
)
base_df, spike_df = run_pipeline(cfg)
spike_df.head()

The run completes in < 10 s with 20 revisions.
"""
from __future__ import annotations

import dataclasses as _dc
import datetime as _dt
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Generator, Iterable, Iterator, List, Optional

import difflib
import requests
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------------------------
@_dc.dataclass
class Config:
    # ---- Input selection ----
    file_paths: Optional[List[str]] = None  # local JSONL(s) (one revision per line)
    fetch_pages: Optional[List[str]] = None  # live titles
    fetch_start: str | None = None  # ISO date
    fetch_end: str | None = None
    fetch_revs_per_page: int | None = 30
    fetch_delay: float = 0.1  # polite API delay

    # JSON field names (for file_paths route)
    text_field: str = "added_text"
    timestamp_field: str = "timestamp"

    # Dev‑mode sampling
    sample_size_revisions: int | None = 50_000
    sample_seed: int = 42

    # Baseline window (inclusive, yyyy‑mm)
    baseline_start: str = "2020-01"
    baseline_end: str = "2022-11"

    # Spike thresholds
    gap_min: float = 0.5
    ratio_min: float = 10.0

    # Misc.
    verbose: bool = True

# ---------------------------------------------------------------------------
# 2. I/O — fetch or load revisions
# ---------------------------------------------------------------------------

Revision = dict  # alias for readability

def load_revisions(cfg: Config) -> Iterator[Revision]:
    """Yield revision dicts according to cfg.*"""
    if cfg.file_paths:
        for fp in cfg.file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    rev = json.loads(line)
                    yield rev
    elif cfg.fetch_pages:
        yield from fetch_revisions_api(cfg)
    else:
        raise ValueError("Either file_paths or fetch_pages must be set")


def fetch_revisions_api(cfg: Config) -> Iterator[Revision]:
    session = requests.Session()
    ENDPOINT = "https://en.wikipedia.org/w/api.php"
    params_template = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "ids|timestamp|comment|content",
        "rvslots": "main",
        "format": "json",
        "formatversion": "2",
    }
    for title in cfg.fetch_pages:
        cont = {}  # continuation token
        fetched = 0
        while True:
            params = {
                **params_template,
                "titles": title,
                "rvlimit": min(cfg.fetch_revs_per_page or 50, 50),
                **cont,
            }
            if cfg.fetch_start:
                params["rvstart"] = cfg.fetch_start + "T00:00:00Z"
            if cfg.fetch_end:
                params["rvend"] = cfg.fetch_end + "T00:00:00Z"
            resp = session.get(ENDPOINT, params=params, timeout=30).json()
            page = resp["query"]["pages"][0]
            revs = page.get("revisions", [])
            # Ensure chronological order (oldest→newest) for diffs
            revs = list(reversed(revs))
            prev_text = ""
            for r in revs:
                if cfg.fetch_revs_per_page and fetched >= cfg.fetch_revs_per_page:
                    break
                new_text = r["slots"]["main"].get("content", "")
                added = extract_added_text(prev_text, new_text)
                yield {
                    "page": title,
                    "rev_id": r["revid"],
                    "timestamp": r["timestamp"],
                    "added_text": added,
                }
                prev_text = new_text
                fetched += 1
            if cfg.fetch_revs_per_page and fetched >= cfg.fetch_revs_per_page:
                break
            cont = resp.get("continue", {})
            if not cont:
                break
            time.sleep(cfg.fetch_delay)


def extract_added_text(old: str, new: str) -> str:
    """Return the '+' side of a diff (very naïve but good enough for dev)."""
    added_tokens = []
    sm = difflib.SequenceMatcher(None, old.split(), new.split())
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("insert", "replace"):
            added_tokens.extend(new.split()[j1:j2])
    return " ".join(added_tokens)

# ---------------------------------------------------------------------------
# 3. Counting and baseline fit
# ---------------------------------------------------------------------------

def build_counts(revisions: Iterable[Revision], cfg: Config) -> pd.DataFrame:
    """Return DataFrame[word, month, freq_obs, total_tokens]."""
    counter = Counter()
    totals = defaultdict(int)
    for rev in revisions:
        month = rev["timestamp"][:7]  # yyyy-mm
        toks = re.findall(r"[a-z]{4,}", rev[cfg.text_field].lower())
        totals[month] += len(toks)
        counter.update((tok, month) for tok in toks)
    rows = [
        {"word": w, "month": m, "count": c, "total": totals[m]}
        for (w, m), c in counter.items()
    ]
    df = pd.DataFrame(rows)
    df["freq_obs"] = df["count"] * 1e6 / df["total"]
    return df[["word", "month", "freq_obs"]]


def fit_baseline(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add expected freq per word & month using robust linear fit."""
    if "month" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "month"})
    df = df.copy()
    df["month_int"] = df["month"].str.replace("-", "").astype(int)

    # Sub‑frame for baseline period
    base = df[(df["month"] >= cfg.baseline_start) & (df["month"] <= cfg.baseline_end)].copy()
    if base.empty:
        raise ValueError("Baseline period contains no data; widen fetch_start/baseline_*.")

    # Fit slope & intercept for each word
    def theil_sen(sub: pd.DataFrame):
        x = sub["month_int"].values
        y = sub["freq_obs"].values
        if len(x) < 2:
            return 0.0, y.mean()
        slopes = [(y[j] - y[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x))]
        slope = pd.Series(slopes).median()
        intercept = pd.Series(y - slope * x).median()
        return slope, intercept

    params = base.groupby("word").apply(theil_sen).apply(pd.Series)
    params.columns = ["slope", "intercept"]
    df = df.merge(params, on="word", how="left")
    df["freq_exp"] = df["slope"] * df["month_int"] + df["intercept"]
    return df.drop(columns=["slope", "intercept", "month_int"])

# ---------------------------------------------------------------------------
# 4. Spike detection
# ---------------------------------------------------------------------------

def detect_spikes(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    post = df[df["month"] > cfg.baseline_end].copy()
    post["gap"] = post["freq_obs"] - post["freq_exp"]
    post["ratio"] = post["freq_obs"] / post["freq_exp"].replace(0, pd.NA)
    spikes = post[(post["gap"] >= cfg.gap_min) & (post["ratio"] >= cfg.ratio_min)]
    first_spike = spikes.groupby("word")["month"].min().rename("first_month")
    out = spikes.groupby("word").agg({"gap": "max", "ratio": "max"}).join(first_spike)
    return out.reset_index().sort_values("gap", ascending=False)

# ---------------------------------------------------------------------------
# 5. Orchestration
# ---------------------------------------------------------------------------

def run_pipeline(cfg: Config):
    rev_iter = load_revisions(cfg)
    if cfg.sample_size_revisions:
        rev_iter = sample_revisions(rev_iter, cfg)
    counts_df = build_counts(rev_iter, cfg)
    base_df = fit_baseline(counts_df, cfg)
    spikes_df = detect_spikes(base_df, cfg)
    return base_df, spikes_df


def sample_revisions(revs: Iterable[Revision], cfg: Config) -> Iterator[Revision]:
    random.seed(cfg.sample_seed)
    reservoir = []
    for i, item in enumerate(revs, 1):
        if i <= cfg.sample_size_revisions:
            reservoir.append(item)
        else:
            j = random.randint(1, i)
            if j <= cfg.sample_size_revisions:
                reservoir[j - 1] = item
    return iter(reservoir)

# ---------------------------------------------------------------------------
if __name__ == "__main__":  # CLI quick test
    cfg = Config(
        fetch_pages=["ChatGPT"],
        fetch_start="2022-11-01",
        fetch_end="2023-01-07",
        fetch_revs_per_page=15,
        baseline_start="2022-11",
        baseline_end="2022-12",
        sample_size_revisions=None,
        gap_min=0.2,
        ratio_min=5,
    )
    base, spikes = run_pipeline(cfg)
    print(spikes.head())
