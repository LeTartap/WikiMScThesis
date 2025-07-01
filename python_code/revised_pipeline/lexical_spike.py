# lexical_spike.py

import re
import pandas as pd
from typing import Set, Union


def load_trigger_set(filepath: str) -> Set[str]:
    """
    Load one‐column CSV of trigger words into a Python set.
    """
    # header=None so we read it as a single column, squeeze to Series
    words = pd.read_csv(filepath, header=None).squeeze().astype(str).str.lower()
    return set(words.tolist())


def compute_pt(text: str, trigger_set: Set[str]) -> float:
    """
    Compute p_t = fraction of tokens in `text` that are in `trigger_set`.
    """
    tokens = re.findall(r'\w+', str(text).lower())
    if not tokens:
        return 0.0
    hits = sum(1 for t in tokens if t in trigger_set)
    return hits / len(tokens)


def compute_baseline_q(
    df: pd.DataFrame,
    trigger_set: Set[str],
    timestamp_col: str = "timestamp",
    text_col: str = "plain_text",
    cutoff_date: Union[str, pd.Timestamp] = "2022-11-01",
) -> float:
    """
    Compute baseline q as the mean p_t over all rows whose timestamp is before cutoff_date.
    """
    # ensure timestamps are datetime
    ts = pd.to_datetime(df[timestamp_col])
    mask = ts < pd.to_datetime(cutoff_date)

    p_series = df.loc[mask, text_col].apply(lambda txt: compute_pt(txt, trigger_set))
    return p_series.mean()


def add_lexical_spike_delta(
    df: pd.DataFrame,
    q: float,
    trigger_set: Set[str],
    text_col: str = "plain_text",
) -> pd.DataFrame:
    """
    Returns a copy of df with two new columns:
      - 'p_t'              : trigger‐word fraction per row
      - 'lexical_spike_delta': p_t minus baseline q
    """
    out = df.copy()
    out["p_t"] = out[text_col].apply(lambda txt: compute_pt(txt, trigger_set))
    out["lexical_spike_delta"] = out["p_t"] - q
    return out
