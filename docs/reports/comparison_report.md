# Term Extraction Benchmark Report

- Combined corpus: `data/interim/combined_corpus.txt`
- Gold JSON: `data/gold/terms_aliases 1.json`
- Flattened gold vocabulary size: `147`
- Top-K: `100`

| Method | Precision | Recall | F1 | TP | FP | FN | Predicted | Gold Terms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yake | 0.12 | 0.0816 | 0.0972 | 12 | 88 | 135 | 100 | 147 |
| spacy_pos_patterns | 0.07 | 0.0476 | 0.0567 | 7 | 93 | 140 | 100 | 147 |
| ensemble_yake_spacy_union | 0.0856 | 0.1088 | 0.0958 | 16 | 171 | 131 | 187 | 147 |
| ensemble_yake_spacy_regex_acronym_union | 0.1156 | 0.2313 | 0.1542 | 34 | 260 | 113 | 294 | 147 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 0.1167 | 0.2381 | 0.1566 | 35 | 265 | 112 | 300 | 147 |