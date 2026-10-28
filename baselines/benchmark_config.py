from pathlib import Path

# --- Input files ---
CHAPTER_FILES = [
    Path("data/raw_data/Handbook_of_Unmanned_Aerial_Vehicles_chapter5.mmd"),
    Path("data/raw_data/Handbook_of_Unmanned_Aerial_Vehicles_chapter6.mmd"),
]

# One combined gold JSON for both chapters together
GOLD_FILE = Path("data/gold/terms_aliases 1.json")

# --- Output locations ---
INTERIM_DIR = Path("data/interim")
EXTRACTIONS_DIR = Path("data/processed/extractions")
EVALUATION_DIR = Path("data/processed/evaluation")
REPORTS_DIR = Path("data/processed/reports")

# --- Extraction settings ---
TOP_K = 100
NGRAM_MIN = 1
NGRAM_MAX = 4

# --- KeyBERT local models ---
KEYBERT_MINILM_MODEL = "models/all-MiniLM-L6-v2"
KEYBERT_MPNET_MODEL = "models/all-mpnet-base-v2"

KEYBERT_USE_MMR = True
KEYBERT_DIVERSITY = 0.5

# --- YAKE settings ---
YAKE_LANGUAGE = "en"
YAKE_DEDUP_LIM = 0.9
YAKE_WINDOW_SIZE = 1

# --- Matching / normalization ---
LOWERCASE = True
STRIP_PUNCT = True
NORMALIZE_DASHES = True
COLLAPSE_WHITESPACE = True
MATCH_ALIASES = True

# --- Benchmark methods ---
DEFAULT_METHODS = [
    #"yake",
    #"rake",
    #"keybert_minilm",
    #"keybert_mpnet",
    #"spacy_noun_chunks",
   # "spacy_pos_patterns",
    #"scispacy_pos_patterns_lg",
   # "ensemble_yake_spacy_union",
   # "ensemble_yake_spacy_regex_acronym_union",
    "ensemble_yake_spacy_regex_acronym_pattern_union"
]