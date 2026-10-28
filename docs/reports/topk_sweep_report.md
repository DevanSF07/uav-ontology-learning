# Top-K Sweep Benchmark Report

- Combined corpus: `data/interim/combined_corpus.txt`
- Gold JSON: `data/gold/terms_aliases 1.json`
- Gold vocabulary size: `147`
- Methods: `yake, ensemble_yake_spacy_union, ensemble_yake_spacy_regex_acronym_union, ensemble_yake_spacy_regex_acronym_pattern_union`
- Top-K values: `25, 50, 75, 100, 150, 200`

## Best result per method

| Method | Best Top-K | Precision | Recall | F1 | TP | FP | FN | Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yake | 200 | 0.09 | 0.1224 | 0.1037 | 18 | 182 | 129 | 200 |
| ensemble_yake_spacy_union | 200 | 0.0814 | 0.2109 | 0.1174 | 31 | 350 | 116 | 381 |
| ensemble_yake_spacy_regex_acronym_union | 150 | 0.0998 | 0.2857 | 0.1479 | 42 | 379 | 105 | 421 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 150 | 0.1026 | 0.2993 | 0.1528 | 44 | 385 | 103 | 429 |

## Full sweep results

| Method | Top-K | Precision | Recall | F1 | TP | FP | FN | Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ensemble_yake_spacy_regex_acronym_pattern_union | 25 | 0.1304 | 0.1837 | 0.1525 | 27 | 180 | 120 | 207 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 50 | 0.1151 | 0.1973 | 0.1454 | 29 | 223 | 118 | 252 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 75 | 0.1077 | 0.2177 | 0.1441 | 32 | 265 | 115 | 297 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 100 | 0.105 | 0.2449 | 0.1469 | 36 | 307 | 111 | 343 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 150 | 0.1026 | 0.2993 | 0.1528 | 44 | 385 | 103 | 429 |
| ensemble_yake_spacy_regex_acronym_pattern_union | 200 | 0.0928 | 0.3265 | 0.1446 | 48 | 469 | 99 | 517 |
| ensemble_yake_spacy_regex_acronym_union | 25 | 0.1276 | 0.1701 | 0.1458 | 25 | 171 | 122 | 196 |
| ensemble_yake_spacy_regex_acronym_union | 50 | 0.1116 | 0.1837 | 0.1388 | 27 | 215 | 120 | 242 |
| ensemble_yake_spacy_regex_acronym_union | 75 | 0.1045 | 0.2041 | 0.1382 | 30 | 257 | 117 | 287 |
| ensemble_yake_spacy_regex_acronym_union | 100 | 0.1021 | 0.2313 | 0.1417 | 34 | 299 | 113 | 333 |
| ensemble_yake_spacy_regex_acronym_union | 150 | 0.0998 | 0.2857 | 0.1479 | 42 | 379 | 105 | 421 |
| ensemble_yake_spacy_regex_acronym_union | 200 | 0.0922 | 0.3197 | 0.1431 | 47 | 463 | 100 | 510 |
| ensemble_yake_spacy_union | 25 | 0.08 | 0.0272 | 0.0406 | 4 | 46 | 143 | 50 |
| ensemble_yake_spacy_union | 50 | 0.0625 | 0.0408 | 0.0494 | 6 | 90 | 141 | 96 |
| ensemble_yake_spacy_union | 75 | 0.0699 | 0.068 | 0.069 | 10 | 133 | 137 | 143 |
| ensemble_yake_spacy_union | 100 | 0.0741 | 0.0952 | 0.0833 | 14 | 175 | 133 | 189 |
| ensemble_yake_spacy_union | 150 | 0.0842 | 0.1633 | 0.1111 | 24 | 261 | 123 | 285 |
| ensemble_yake_spacy_union | 200 | 0.0814 | 0.2109 | 0.1174 | 31 | 350 | 116 | 381 |
| yake | 25 | 0.04 | 0.0068 | 0.0116 | 1 | 24 | 146 | 25 |
| yake | 50 | 0.06 | 0.0204 | 0.0305 | 3 | 47 | 144 | 50 |
| yake | 75 | 0.0933 | 0.0476 | 0.0631 | 7 | 68 | 140 | 75 |
| yake | 100 | 0.09 | 0.0612 | 0.0729 | 9 | 91 | 138 | 100 |
| yake | 150 | 0.1 | 0.102 | 0.101 | 15 | 135 | 132 | 150 |
| yake | 200 | 0.09 | 0.1224 | 0.1037 | 18 | 182 | 129 | 200 |