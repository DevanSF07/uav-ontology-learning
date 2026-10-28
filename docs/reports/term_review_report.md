# Term Review Benchmark Report

- Model: `llama3.1:8b`
- Initial term-label file: `data/processed/evaluation/predicted_term_labels.csv`
- Combined corpus: `data/interim/combined_corpus.txt`
- Max review context sentences: `10`
- Max review context chars: `5000`

## Review summary

- Total reviewed rows: `567`
- Confirmed labels: `132`
- Changed labels: `325`
- Average confidence: `67.9475`
- Average context sentence count: `2.2801`
- Average context char count: `821.2582`

## Output files

- Reviewed CSV: `data/processed/evaluation/reviewed_term_labels_with_confidence.csv`
- Runtime CSV: `data/processed/evaluation/reviewed_term_labels_runtime.csv`
- Raw outputs CSV: `data/processed/evaluation/reviewed_term_labels_raw_outputs.csv`