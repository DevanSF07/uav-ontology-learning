# Term Typing Benchmark Report

- Model: `llama3.1:8b`
- Gold labels CSV: `/home/faujdar/onto_gen/data/gold/terms_labels_GT.csv`
- Extracted terms CSV: `data/processed/extractions/llm_llama3.1_8b_terms.csv`
- Combined corpus: `data/interim/combined_corpus.txt`

## Pure typing (overlap-only) summary

- n_items: `75`
- accuracy: `0.44`
- macro_precision: `0.5249`
- macro_recall: `0.4161`
- macro_f1: `0.3753`
- weighted_precision: `0.779`
- weighted_recall: `0.44`
- weighted_f1: `0.5389`

## End-to-end (term,label pair) summary

- n_gold_pairs: `175`
- n_pred_pairs: `453`
- tp: `33`
- fp: `420`
- fn: `142`
- precision: `0.0728`
- recall: `0.1886`
- f1: `0.1051`

## Output files
- Predictions: `data/processed/evaluation/predicted_term_labels.csv`
- Confusion matrix: `data/processed/evaluation/term_typing_confusion_matrix.csv`
- Per-class report: `data/processed/evaluation/term_typing_classification_report.csv`
- Runtime: `data/processed/evaluation/term_typing_runtime.csv`