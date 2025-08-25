"""
SpaCy Parser Script

This script provides a function to parse text with spaCy in parallel with improved error handling.
It's designed to be used as a standalone script or imported into a notebook.
"""

import pandas as pd
import spacy
from parallel import process_dataframe_parallel

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def parse_with_spacy(text: str):
    """Parse text with spaCy and extract linguistic features"""
    if not isinstance(text, str) or not text.strip():
        return {
            "upos_props": {},
            "mean_dep_depth": 0,
            "clause_ratio": 0,
            "voice_ratio": 0,
            "sentences": [],
            "tokens": []
        }
    
    try:
        # Limit text length to prevent memory issues
        max_length = 100000  # Adjust based on available memory
        if len(text) > max_length:
            text = text[:max_length]
            
        doc = nlp(text)
        total_tokens = len(doc)
        
        if total_tokens == 0:
            return {
                "upos_props": {},
                "mean_dep_depth": 0,
                "clause_ratio": 0,
                "voice_ratio": 0,
                "sentences": [],
                "tokens": []
            }

        # POS proportions
        pos_counts = doc.count_by(spacy.attrs.POS)
        upos_props = {nlp.vocab[pos].text: cnt / total_tokens for pos, cnt in pos_counts.items()}

        # Dependency depth approximation
        def token_depth(token):
            depth = 0
            while token != token.head:
                token = token.head
                depth += 1
            return depth

        depths = [token_depth(token) for token in doc]
        mean_depth = sum(depths) / total_tokens if total_tokens else 0

        # Clause ratio
        clause_tags = sum(1 for token in doc if token.dep_ in ("advcl", "ccomp", "xcomp"))
        clause_ratio = clause_tags / (len(list(doc.sents)) or 1)

        # Passive voice ratio
        passive_count = sum(1 for token in doc if token.dep_ == "auxpass")
        voice_ratio = (total_tokens - passive_count) / (total_tokens or 1)

        return {
            "upos_props": upos_props,
            "mean_dep_depth": mean_depth,
            "clause_ratio": clause_ratio,
            "voice_ratio": voice_ratio,
            "sentences": [sent.text for sent in doc.sents],
            "tokens": [token.text for token in doc]
        }
    except Exception as e:
        print(f"Error parsing text: {str(e)}")
        return {
            "upos_props": {},
            "mean_dep_depth": 0,
            "clause_ratio": 0,
            "voice_ratio": 0,
            "sentences": [],
            "tokens": []
        }

def parse_dataframe_with_spacy(df, text_column="plain_text", output_column="parsed"):
    """
    Parse text in a DataFrame with spaCy in parallel with improved error handling.
    
    Args:
        df (pandas.DataFrame): DataFrame containing text to parse
        text_column (str): Name of column containing text to parse
        output_column (str): Name of column to store parsed results
        
    Returns:
        pandas.DataFrame: DataFrame with parsed results
    """
    print(f"Parsing {len(df)} rows with spaCy...")
    
    # Parse text with spaCy in parallel
    # This is CPU-intensive, so we use processes instead of threads
    result_df = process_dataframe_parallel(
        df,
        parse_with_spacy,
        column=text_column,
        new_column=output_column,
        use_threads=False,  # Use processes for CPU-intensive task
        cpu_intensive=True,
        timeout=900,  # Increase timeout to 15 minutes
        max_retries=7,  # Increase max retries
        max_workers=2  # Reduce number of workers to avoid memory issues
    )
    
    print("Parsing complete!")
    return result_df

if __name__ == "__main__":
    # This code runs when the script is executed directly
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python spacy_parser.py input_file.pkl output_file.pkl [text_column] [output_column]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    text_column = sys.argv[3] if len(sys.argv) > 3 else "plain_text"
    output_column = sys.argv[4] if len(sys.argv) > 4 else "parsed"
    
    print(f"Loading data from {input_file}...")
    df = pd.read_pickle(input_file)
    
    result_df = parse_dataframe_with_spacy(df, text_column, output_column)
    
    print(f"Saving results to {output_file}...")
    result_df.to_pickle(output_file)
    print("Done!")