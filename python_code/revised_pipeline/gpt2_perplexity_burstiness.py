# gpt2_metrics.py

import torch
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import pandas as pd
import re
from typing import Tuple

# ——————————————————————————————————————————————————————————————
#  GPU / model initialization
# ——————————————————————————————————————————————————————————————

def get_gpu_info() -> dict:
    """Return a small dict of PyTorch/CUDA status & GPU name (if any)."""
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }
    return info

# load GPT-2 once on import
_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
_model     = GPT2LMHeadModel.from_pretrained("gpt2")
_model.eval()

if torch.cuda.is_available():
    _model.to("cuda")


# ——————————————————————————————————————————————————————————————
#  Perplexity & burstiness
# ——————————————————————————————————————————————————————————————

def compute_perplexity_and_burstiness(
    text: str,
    max_length: int = 512,
    chunk_size: int = 8
) -> Tuple[float, float]:
    """
    Returns (ppl, burstiness) for a single string using GPT-2.
    If CUDA is unavailable or input too short/invalid, returns (0.0, 0.0).
    """
    # quick sanity / device checks
    if not isinstance(text, str) or len(text.strip()) < 5:
        return 0.0, 0.0
    if not torch.cuda.is_available():
        # user can still see this warning and decide
        print("WARNING: CUDA not available, skipping.")
        return 0.0, 0.0
    if _model.device.type != "cuda":
        _model.to("cuda")

    # ensure pad token is set
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    try:
        enc = _tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length
        )
        input_ids = enc.input_ids.clamp(0, _model.config.vocab_size - 1).to("cuda")
        seq_len = input_ids.shape[1]

        # too short for meaningful ppl
        if seq_len < 5:
            return 0.0, 0.0

        # compute token‐level cross‐entropy in chunks
        total_loss = 0.0
        total_tokens = 0
        for i in range(0, seq_len, chunk_size):
            chunk = input_ids[:, i : i + chunk_size]
            with torch.no_grad():
                out = _model(chunk, labels=chunk)
            total_loss   += out.loss.item() * chunk.shape[1]
            total_tokens += chunk.shape[1]

        if total_tokens == 0:
            return 0.0, 0.0

        avg_loss = total_loss / total_tokens
        ppl = float(torch.exp(torch.tensor(avg_loss)))

        # burstiness: stddev of the log‐prob at two sample positions
        log_probs = []
        for pos in (min(10, seq_len - 1), min(20, seq_len - 1)):
            if pos < 5:
                continue
            seg = input_ids[:, :pos]
            with torch.no_grad():
                out = _model(seg, labels=seg)
            log_probs.append(-out.loss.item())

        burst = float(pd.Series(log_probs).std()) if len(log_probs) > 1 else 0.0
        return ppl, burst

    except Exception as e:
        print(f"Error in compute_perplexity_and_burstiness: {e}")
        return 0.0, 0.0

# At the bottom of gpt2_perplexity_burstiness.py

import pandas as pd
from tqdm.auto import tqdm

def add_perplexity_and_burstiness_to_df(
    df: pd.DataFrame,
    text_col: str = "plain_text",
    perplexity_col: str = "perplexity",
    burstiness_col: str = "burstiness",
    batch_size: int = 8,
    desc: str = "Computing perplexity and burstiness"
) -> pd.DataFrame:
    """
    Computes (ppl, burstiness) for each row in df[text_col] in batches,
    using compute_perplexity_and_burstiness(), and writes the results into
    two new columns. Returns the modified DataFrame.
    """
    texts = df[text_col].fillna("").astype(str).tolist()
    results = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i : i + batch_size]
        batch_results = [compute_perplexity_and_burstiness(txt) for txt in batch]
        results.extend(batch_results)

    # unzip into two lists
    ppl_vals, burst_vals = zip(*results) if results else ([], [])
    df[perplexity_col] = ppl_vals
    df[burstiness_col] = burst_vals
    return df
