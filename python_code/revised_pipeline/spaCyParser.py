import spacy



def get_spacy_model():
    # This attribute is specific to each process
    if not hasattr(get_spacy_model, "nlp"):
        get_spacy_model.nlp = spacy.load("en_core_web_sm")
    return get_spacy_model.nlp


def parse_with_spacy(text: str):
    """Parse text with spaCy and extract linguistic features (multiprocessing safe)."""
    nlp = get_spacy_model()  # <- Load per worker!
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
        upos_props = {doc.vocab[pos].text: cnt / total_tokens for pos, cnt in pos_counts.items()}

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
