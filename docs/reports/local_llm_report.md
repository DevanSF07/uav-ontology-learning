# Local LLM Term Extraction Benchmark

- Combined corpus: `data/interim/combined_corpus.txt`
- Gold JSON: `data/gold/terms_aliases 1.json`
- Models: `qwen2.5:7b, llama3.1:8b, mistral:7b`

## Accuracy Results

| Method | Precision | Recall | F1 | TP | FP | FN | Predicted |
|---|---:|---:|---:|---:|---:|---:|---:|
| llm_qwen2.5_7b | 0.1681 | 0.5306 | 0.2553 | 78 | 386 | 69 | 464 |
| llm_llama3.1_8b | 0.1637 | 0.6259 | 0.2595 | 92 | 470 | 55 | 562 |
| llm_mistral_7b | 0.0938 | 0.4082 | 0.1525 | 60 | 580 | 87 | 640 |

## Runtime Results

| Method | Wall Clock (s) | Prompt Tokens | Output Tokens | Tokens/s | Total Duration (ns) | Load Duration (ns) |
|---|---:|---:|---:|---:|---:|---:|
| llm_qwen2.5_7b | 1026.7704 | 23585 | 2895 | 3.1829 | 1026638241071 | 19239402265 |
| llm_llama3.1_8b | 1796.1003 | 22717 | 4002 | 2.4819 | 1795950023508 | 56128702802 |
| llm_mistral_7b | 2346.5578 | 26875 | 5313 | 2.5341 | 2346394555595 | 47990710128 |