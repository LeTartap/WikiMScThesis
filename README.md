# Detecting AI-Assisted Edits in Wikipedia

This repository contains a prototype pipeline for extracting, processing, and analyzing Wikipedia revision data to detect AI-assisted writing using a lightweight, rule-/threshold-based approach (“Method B”). The code is organized as Jupyter notebooks and supporting Python modules.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Dependencies & Installation](#dependencies--installation)
4. [Data Format](#data-format)
5. [Usage](#usage)

   1. [1. Fetch Tiny Revision Sample](#1-fetch-tiny-revision-sample)
   2. [2. Clean & Tokenize Text](#2-clean--tokenize-text)
   3. [3. Extract Features](#3-extract-features)
   4. [4. Rule-/Threshold-Based Detection](#4-rule--threshold-based-detection)
   5. [5. Inspect & Iterate](#5-inspect--iterate)
6. [Configuration](#configuration)
7. [Extending to Full-Scale Data](#extending-to-full-scale-data)
8. [Future Work](#future-work)
9. [License](#license)

---

## Project Overview

The goal of this project is to build a scalable, transparent pipeline that flags AI-assisted revisions on English Wikipedia without relying on heavy neural-network training.

* **Data source:** Wikipedia revision histories via MediaWiki API
* **Scope:** A small “toy” sample of pages for prototype, eventually scaling to tens of thousands of articles
* **Features (lightweight signals):**

  1. **Lexical spike (δ):** Relative frequency of a fixed set of LLM-favored words
  2. **Perplexity & Burstiness:** GPT-2 small scores as proxies for AI-likeness
  3. **Syntactic Profile:** UPOS tag proportions, mean dependency depth, clause ratio
  4. **Readability & Verbosity:** Flesch Reading Ease, Gunning Fog, chars/sentence, sentences/paragraph
  5. **Vocabulary Diversity:** Normalized type–token ratio (nTTR), word-density index
  6. **Voice & Layout:** Active/passive ratio, average raw-text line length
  7. **Citation Delta:** Net change in `<ref>` tags per tokens changed
* **Detection (Method B):** Standardize each feature, apply simple thresholds (vote system) to flag likely AI-assisted edits

This prototype is written in Python notebooks to allow rapid iteration and clear documentation of each step.

---

## Repository Structure

```
.
├── README.md
├── pipeline_prototype.ipynb         # Main Jupyter notebook for toy-data pipeline
├── requirements.txt                 # Python dependencies
├── utils/
│   ├── api_helpers.py               # Functions for fetching revisions via MediaWiki API
│   ├── text_cleaning.py             # clean_text(), parse_with_spacy() functions
│   ├── feature_extraction.py        # compute_delta(), compute_perplexity_and_burstiness(), etc.
│   ├── detection.py                 # rule_based_flag() and threshold definitions
│   └── config.py                    # Constants for sample pages, date ranges, thresholds
├── data/
│   ├── tiny_revisions.pkl           # (Optional) Cached revision metadata & content
│   ├── tiny_features.csv            # (Optional) Extracted features for toy data
│   └── …                            # Placeholder for future large-scale data files
├── docs/
│   ├── project_plan.md              # Detailed task list & pipeline plan
│   ├── feature_spec_sheet.md        # Spec sheet describing each feature block
│   └── related_work_table.md        # Table of AI-usage indicators from the literature
└── LICENSE
```

* **pipeline\_prototype.ipynb:**
  A step-by-step Jupyter notebook implementing the tiny-data pipeline.
* **utils/\*.py:**
  Modular Python functions called by the notebook for API access, text cleaning, feature extraction, and detection logic.
* **data/:**
  Storage for any cached toy data (pickles, CSVs) during prototype development.
* **docs/:**
  Documentation and supporting materials (project plan, feature spec, related work table).

---

## Dependencies & Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/wikipedia-ai-detection.git
   cd wikipedia-ai-detection
   ```

2. **Create a virtual environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

   **`requirements.txt`** should include at least:

   ```
   requests
   pandas
   spacy
   textstat
   transformers
   torch
   wikipedia-api
   sklearn
   seaborn
   matplotlib
   ```

---

## Data Format

For the prototype, each revision’s data will be stored in a Pandas DataFrame (or pickled to disk) with these columns:

| Column           | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `page_title`     | Wikipedia page title                                        |
| `rev_id`         | Revision ID (integer)                                       |
| `timestamp`      | UTC timestamp of the revision (e.g. `2023-02-15T12:34:56Z`) |
| `user`           | Username of editor                                          |
| `is_bot`         | Boolean: true if username ends with “bot”                   |
| `content`        | Raw wikitext of the revision                                |
| `plain_text`     | Cleaned, lowercase, stripped plain text                     |
| `delta`          | Lexical-spike value (float)                                 |
| `perplexity`     | GPT-2 small perplexity (float)                              |
| `burstiness`     | Standard deviation of GPT-2 log-probs (float)               |
| `upos_*`         | One column per UPOS tag proportion (e.g. `upos_NOUN`)       |
| `mean_dep_depth` | Mean dependency‐parse depth (float)                         |
| `clause_ratio`   | Clause-per-sentence ratio (float)                           |
| `voice_ratio`    | Active-minus-passive ratio (float)                          |
| `fre`            | Flesch Reading Ease (float)                                 |
| `fog`            | Gunning Fog index (float)                                   |
| `chars_per_sent` | Characters per sentence (float)                             |
| `sents_per_para` | Sentences per paragraph (int or float)                      |
| `nTTR`           | Normalized TTR on first 250 tokens (float)                  |
| `word_density`   | Word-density index (float)                                  |
| `avg_line_len`   | Average characters per line (float)                         |
| `citation_delta` | (`<ref>` added – removed) / tokens\_changed (float)         |
| `ai_flag`        | Boolean: 1 if rule-based detector labels as AI-assisted     |

This schema can be extended for full‐scale runs, but it’s sufficient for the prototype.

---

## Usage

Open `pipeline_prototype.ipynb` in Jupyter or JupyterLab and follow the cells in order. Below is a summary of the main steps:

### 1. Fetch Tiny Revision Sample

* Use `api_helpers.py`’s `fetch_revisions_for_page(title, start_ts, end_ts)` to grab all revisions for each page in the sample.
* Store results in a Pandas DataFrame (`tiny_revs`).
* (Optional) Save to `data/tiny_revisions.pkl` for caching.

### 2. Clean & Tokenize Text

* Run `clean_text(wikitext)` from `text_cleaning.py` to strip wiki markup, remove non-letters, lowercase, and collapse whitespace.
* Process cleaned text with spaCy (`parse_with_spacy`) to get sentences, tokens, UPOS tags, dependency depth, clause ratio, and voice ratio.

### 3. Extract Features

* **Lexical spike (δ):** `compute_delta(text, trigger_set, baseline_freq)` in `feature_extraction.py`.
* **Perplexity & Burstiness:** `compute_perplexity_and_burstiness(text)` using GPT-2 small.
* **Syntactic profile:** Already available from spaCy parse outputs.
* **Readability & Verbosity:** `compute_readability(text)` using `textstat`.
* **Vocabulary Diversity:** `compute_vocab_diversity(text, window_size=250)`.
* **Voice & Layout:** `voice_ratio`, `avg_line_len`.
* **Citation Delta:** `compute_citation_delta(wikitext)` via regex.
* Aggregate all feature columns into a single DataFrame `features_df` and save as `data/tiny_features.csv`.

### 4. Rule-/Threshold-Based Detection

* Standardize numeric features with `StandardScaler`.
* Define simple thresholds in `config.py` (e.g. `delta > 0.05`, `perplexity < -1`, `citation_delta < -0.1`).
* Use `rule_based_flag(row, thresholds)` from `detection.py` to assign `ai_flag`.
* Inspect `features_df[features_df["ai_flag"] == True]` to see flagged revisions.

### 5. Inspect & Iterate

* Plot distributions (e.g., KDE of `delta` by `ai_flag`) to visualize separation.
* Adjust thresholds or add/remove feature rules to improve toy-data performance.
* Document your observations in the notebook’s commentary cells.

---

## Configuration

All tunable constants are defined in `utils/config.py`:

* `sample_pages`: List of page titles to fetch.
* `START_TIMESTAMP`, `END_TIMESTAMP`: Time window for revision fetching.
* `TRIGGER_SET`: Set of LLM-favored words for δ.
* `THRESHOLDS`: Dictionary of feature thresholds for rule-based voting.
* `VOTE_CUTOFF`: Minimum number of triggered features to mark `ai_flag=True`.

Modify `config.py` as needed before running the notebook.

---

## Extending to Full-Scale Data

Once the prototype is validated on tiny data, you can scale up by:

1. **Increasing `sample_pages`** to include all pages in chosen strata (via `categorymembers`).
2. **Adjusting date ranges** to cover 2020–2024 (261 weekly snapshots per page).
3. **Switch GPT-2 small → GPT-2 XL** (or use an n-gram model) for better perplexity fidelity.
4. **Parallelize spaCy and transformers passes** (e.g., using multiprocessing or batching on GPU).
5. **Store features in a distributed format** (e.g., Parquet on S3 or a local XCom) for memory‐efficient processing.
6. **Refine detection rules** or replace with a lightweight logistic regression (still under Method B).

Refer to `docs/project_plan.md` for a detailed task breakdown, and `docs/feature_spec_sheet.md` for a complete feature definition.

---

## Future Work

* **Temporal Analysis:** Use the per-revision `P_AI` scores to run Interrupted Time-Series or mixed-effects models (R or Python).
* **Fine-tuning thresholds:** Explore a small labeled set (if available) to calibrate rules or weights.
* **Error Analysis:** Inspect false positives/negatives on a mid-sized random sample and refine feature rules.
* **Related Physiology:** Incorporate editor metadata (new-user flags, bot-flag, total edits) if it proves useful.
* **Open Source Release:** Package code into a CLI tool or Airflow DAG so others can replicate on their own strata.

---

## License

This project is released under the MIT License. See `LICENSE` for details.
