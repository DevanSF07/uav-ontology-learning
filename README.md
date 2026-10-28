# Automated UAV Domain Ontology Learning Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest%20passing-brightgreen.svg)](tests/)
[![Architecture: Precision--First](https://img.shields.io/badge/architecture-precision--first%20multi--agent-orange.svg)](#system-architecture)
[![Models: Local LLM + MPNet](https://img.shields.io/badge/models-Qwen%20%7C%20Llama%20%7C%20all--mpnet--base--v2-blueviolet.svg)](#key-experimental-findings--benchmarks)

A reproducible, production-grade framework for **automated domain ontology extraction, validation, structural entity typing, and hierarchical taxonomy synthesis** from unstructured technical aerospace literature (*Handbook of Unmanned Aerial Vehicles*).

Developed by **Devan Singh Faujdar** (IIT Kharagpur) during research at **Synopsys Inc.**

---

## 📌 1. Executive Summary & Problem Formulation

Domain ontology engineering is historically a labor-intensive manual task. Unsupervised statistical extractors (YAKE, TF-IDF, POS regex patterns) suffer from high false-positive rates (**<12% precision**), flooding candidate sets with generic aerospace verbs, boilerplate phrases, and low-information n-grams. Conversely, naive one-shot LLM prompting suffers from hallucinations, non-standardized typing, and lack of corpus grounding.

This framework introduces a **4-Layer, 13-Stage Precision-First Architecture**:
1. **Evidence Reservoir Paradigm:** Statistical extractors (YAKE, KeyBERT, SciSpacy POS, TF-IDF) are treated as *term evidence pools* rather than final ontology terms.
2. **Context-Aware Noise & Boundary Filtering:** Pre-filtering using GLiNER zero-shot NER and cosine boundary thresholds prune non-domain candidates.
3. **Seed-Guided Semantic Neighborhoods:** Dense sentence embeddings (`all-mpnet-base-v2`) and Non-Negative Matrix Factorization (NMF) cluster candidate terms into topic subspaces without negative distortion.
4. **Local Multi-Agent LLM Validation & Metamodel Typing:** Local LLMs (Qwen 2.5 7B, LLaMA 3.1 8B) validate candidates against exact excerpted sentence evidence and classify terms into standardized structural metamodel categories (`System`, `Subsystem`, `Component`, `Part`, `Interface`, `Flow`).
5. **Corpus-Grounded Recursive Taxonomy Induction:** A resilient bottom-up and top-down taxonomy builder constructs parent-child relations with strict corpus presence enforcement.

---

## 🏗️ 2. System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Raw Aerospace Literature (.mmd)          │
                    │      (Handbook of Unmanned Aerial Vehicles, Ch 5 & 6)  │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: Corpus Preprocessing & Evidential Extraction                                      │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Clean MMD tables, broken hyphenation, figure captions & low-information lines            │
│  • Sentence & Paragraph Semantic Chunking                                                  │
│  • Multi-Extractor Reservoir: YAKE | KeyBERT (MiniLM & MPNet) | SciSpacy POS | TF-IDF      │
│  • Ensemble & Frequency-Weighted Voting                                                     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │ Raw Candidates
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: Noise Filtering & Semantic Subspace Modeling                                       │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • GLiNER Zero-Shot Noise Boundary Pruning                                                  │
│  • Dense Sentence Embeddings (`sentence-transformers/all-mpnet-base-v2`)                    │
│  • Cosine Similarity Seed Graph Construction & Threshold Sweep                             │
│  • Non-Negative Matrix Factorization (NMF) on TF-IDF Term-Chunk Matrix                      │
│  • Contextual Clustering & Cluster Equivalence Matching                                     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │ Filtered Concept Clusters
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: Precision-First Local LLM Evidence Validation & Typing                              │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Evidence Extraction: Excerpt supporting sentences per term from corpus                   │
│  • Local LLM Validation Gate (Qwen-2.5-7B-Instruct / LLaMA-3.1-8B via Ollama)              │
│  • 5x Precision Boost: Discards out-of-domain terms while retaining high-confidence concepts│
│  • Multi-Agent Metamodel Entity Typing:                                                     │
│      ├── Structural: System, Subsystem, Component, Part, Interface, Flow                    │
│      └── Contextual: Mission, Capability, Environment, Parameter, Classification Class     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │ Validated & Typed Terms
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: Corpus Grounding & Hierarchical Taxonomy Induction                                 │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│  • Strict Exact & Lemma Corpus Presence Filter (Disallows hallucinated parents)             │
│  • Alias & Acronym Resolution (Regex + Cosine > 0.7 candidate pairing)                      │
│  • Hybrid Recursive Tree Builder: Links Subsystems to Systems and Components to Subsystems  │
│  • Ground Truth Evaluation against Expert Benchmarks (GT1-GT4)                              │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 3. The 13-Stage Experimental Pipeline

The repository houses the complete, self-contained multi-stage pipeline under [`experiments/`](experiments/):

| Stage | Script | Purpose & Methodology |
|:---:|:---|:---|
| **01** | [`test_01_cleaning.py`](experiments/test_01_cleaning.py) | Strip tables, markdown artifacts, references, and format corpus into clean normalized text. |
| **02** | [`test_02_chunking.py`](experiments/test_02_chunking.py) | Dynamic chunking with token overlap preserves contextual coherence across sentence boundaries. |
| **02a**| [`test_02a_sae_based_chunking.py`](experiments/test_02a_sae_based_chunking.py) | Sparse Autoencoder (SAE) concept discovery trial for latent domain concept activation. |
| **03** | [`test_03_extraction.py`](experiments/test_03_extraction.py) | Multi-extractor execution across YAKE, KeyBERT, TF-IDF, and POS noun phrases. |
| **03b**| [`test_03b_ensemble.py`](experiments/test_03b_ensemble.py) | Set-union and intersection ensemble strategies to assess precision vs recall trade-offs. |
| **03c**| [`test_03c_weighted_voting.py`](experiments/test_03c_weighted_voting.py) | Rank-weighted confidence scoring across extractor concurrence. |
| **04** | [`test_04_gliner_filtering.py`](experiments/test_04_gliner_filtering.py) | Zero-shot NER filtering using GLiNER to discard non-entity text noise. |
| **05** | [`test_05_clustering_filtering.py`](experiments/test_05_clustering_filtering.py) | Agglomerative and K-Means clustering on MPNet representations to group lexical variants. |
| **05b**| [`test_05b_match_equivalent_clusters_v2.py`](experiments/test_05b_match_equivalent_clusters_v2.py) | Inter-cluster similarity matching to merge duplicate semantic concepts. |
| **05c**| [`test_05c_remove_cluster_terms_evaluate.py`](experiments/test_05c_remove_cluster_terms_evaluate.py) | Prune peripheral noise clusters and evaluate impact on ground truth recall. |
| **06** | [`test_06_mpnet_seed_graph_threshold_sweep.py`](experiments/test_06_mpnet_seed_graph_threshold_sweep.py) | Cosine similarity threshold sweep ($0.50 \to 0.85$) optimizing seed graph density. |
| **07** | [`test_07_nmf_filtering.py`](experiments/test_07_nmf_filtering.py) | Non-Negative Matrix Factorization isolates orthogonal UAV sub-domains without negative embedding artifacts. |
| **07b**| [`test_07b_qwen_yake_mpnet_validation.py`](experiments/test_07b_qwen_yake_mpnet_validation.py) | Qwen-2.5-7B validation pass on YAKE + MPNet candidates using excerpted sentence context. |
| **08** | [`test_08_s3_concept_coverage_demo.py`](experiments/test_08_s3_concept_coverage_demo.py) | Concept coverage verification ensuring all core UAV subsystems are represented. |
| **09a**| [`test_09a_trace_yake_mpnet_qwen_recall_losses.py`](experiments/test_09a_trace_yake_mpnet_qwen_recall_losses.py) | Detailed audit tracing exactly where ground truth terms are filtered out across pipeline hops. |
| **10** | [`test_10_multiagent_qwen_entity_typing_final.py`](experiments/test_10_multiagent_qwen_entity_typing_final.py) | Multi-agent structural typing classifying terms into SysML/MBSE ontological types. |
| **11** | [`test_11_taxonomy_from_all_typed_terms_resilient.py`](experiments/test_11_taxonomy_from_all_typed_terms_resilient.py) | Resilient graph builder constructing parent-child hierarchies from typed entities. |
| **12** | [`test_12_all_qwen_typing_then_taxonomy.py`](experiments/test_12_all_qwen_typing_then_taxonomy.py) | End-to-end typing-through-taxonomy validation pipeline. |
| **13a**| [`test_13a_noisefilter.py`](experiments/test_13a_noisefilter.py) | Strict corpus presence and noise filtering preventing phantom taxonomy root nodes. |
| **13** | [`test_13_hybrid_recursive_taxonomy_builder.py`](experiments/test_13_hybrid_recursive_taxonomy_builder.py) | Final recursive tree synthesis outputting validated, navigable UAV domain taxonomy. |

---

## 📊 4. Key Experimental Findings & Benchmarks

All experiments are evaluated against verified Ground Truth datasets in [`gold/`](gold/) (`GT1` domain terms, `GT2` acronyms/aliases, `GT3` concept clusters, `GT4` taxonomy hierarchy).

### A. Raw Statistical Extractor Comparison (Top-100 Candidates)
*Evaluated against 147 Ground Truth Aerospace Domain Terms:*

| Method | Precision | Recall | F1-Score | True Positives | False Positives |
|:---|:---:|:---:|:---:|:---:|:---:|
| **YAKE** | 12.0% | 8.2% | 0.097 | 12 | 88 |
| **SciSpacy POS Patterns** | 7.0% | 4.8% | 0.057 | 7 | 93 |
| **Ensemble (YAKE ∪ SpaCy)** | 8.6% | 10.9% | 0.096 | 16 | 171 |
| **Ensemble (YAKE ∪ SpaCy ∪ Acronym)** | 11.6% | 23.1% | 0.154 | 34 | 260 |
| **Ensemble (Full Reservoir Pattern Union)** | **11.7%** | **23.8%** | **0.157** | **35** | **265** |

> **Key Insight:** Raw extractors max out at ~11.7% precision. An evidence ensemble increases recall from 8.2% to 23.8%, but creates a reservoir with 265 false positives requiring downstream semantic gating.

### B. Precision-First Local LLM Evidence Gating
By running Qwen-2.5-7B and LLaMA-3.1-8B over extracted candidate evidence sentences:
- **Precision Uplift:** Precision improved from **11.7% $\to$ 58.4%** (**~5x increase**).
- **False Positive Reduction:** Over **82% of non-domain noise** (e.g. *"preliminary design phase"*, *"various factors"*, *"further investigation"*) was systematically eliminated before structural typing.

### C. Structural Metamodel Typing Performance
*Tested across 13 structural and contextual classes (`System`, `Subsystem`, `Component`, `Part`, `Interface`, `Flow`, etc.):*

| Metric | Overlap Evaluation (Direct Match) | End-to-End Evaluation |
|:---|:---:|:---:|
| **Weighted Precision** | **77.9%** | 52.5% |
| **Weighted Recall** | 44.0% | 41.6% |
| **Weighted F1-Score** | **53.9%** | 37.5% |
| **Accuracy** | 44.0% | — |

---

## 📂 5. Repository Structure

```text
uav-ontology-learning/
├── README.md                      # Comprehensive project documentation
├── pyproject.toml                 # Package definition & CLI configuration
├── requirements.txt               # Locked environment dependencies
├── LICENSE                        # MIT License
├── .gitignore                     # Production ignore rules (excludes >1GB outputs & secrets)
│
├── olbench/                       # Core reusable framework package
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                     # CLI commands (run, collect, eval)
│   ├── pipeline.py                # Pipeline execution engine
│   ├── preprocess.py              # Text cleaning and chunking
│   ├── extractors.py              # Statistical & embedding extractors
│   ├── noise.py                   # GLiNER & heuristic noise filtering
│   ├── aliasing.py                # Cosine & regex acronym resolution
│   ├── concepts.py                # NMF & embedding clustering
│   ├── seed_similarity.py         # Seed contextual similarity
│   ├── typing.py                  # Metamodel entity typing logic
│   ├── evaluate.py                # Ground truth metric calculations
│   └── utils.py                   # Content-addressed caching & IO
│
├── experiments/                   # 13 Stage Experiment & Evaluation Pipelines
│   ├── test_01_cleaning.py
│   ├── test_02_chunking.py
│   ├── test_02a_sae_based_chunking.py
│   ├── test_03_extraction.py
│   ├── test_03b_ensemble.py
│   ├── test_03c_weighted_voting.py
│   ├── test_04_gliner_filtering.py
│   ├── test_05_clustering_filtering.py
│   ├── test_05_seed_similarity_filtering.py
│   ├── test_05b_match_equivalent_clusters_v2.py
│   ├── test_05c_remove_cluster_terms_evaluate.py
│   ├── test_06_mpnet_seed_graph_threshold_sweep.py
│   ├── test_06_seed_guided_contextual_clustering.py
│   ├── test_07_nmf_filtering.py
│   ├── test_07_qwen_llm_term_validation.py
│   ├── test_07b_qwen_yake_mpnet_validation.py
│   ├── test_08_s3_concept_coverage_demo.py
│   ├── test_09a_trace_yake_mpnet_qwen_recall_losses.py
│   ├── test_10_multiagent_qwen_entity_typing_final.py
│   ├── test_11_taxonomy_from_all_typed_terms_resilient.py
│   ├── test_12_all_qwen_typing_then_taxonomy.py
│   ├── test_13a_noisefilter.py
│   ├── test_13_hybrid_recursive_taxonomy_builder.py
│   └── uav_term_context_mpnet_nmf_pipeline.py
│
├── baselines/                     # Baseline extractors, OpenIE & legacy models
│   ├── alexbek_term_typing.py
│   ├── benchmark_config.py
│   ├── clean_corpus_advanced.py
│   ├── extract_triplets_openie.py
│   ├── extract_triplets_spacy.py
│   ├── local_llm_extractors.py
│   ├── local_llm_term_review.py
│   ├── run_benchmark.py
│   └── run_benchmark_topk_sweep.py
│
├── configs/                       # Experiment parameter specifications (JSON & YAML)
│   ├── core_experiments.json
│   ├── config_embedding_precision_v2.yaml
│   ├── sae_trial.json
│   ├── seed_guided_contextual_clustering.json
│   └── smoke.json
│
├── data/                          # Unstructured aerospace literature & raw corpora
│   ├── chapter5_chapter6.mmd      # Combined UAV technical chapters
│   ├── demo_chapter5_chapter6.mmd # Lightweight smoke test dataset
│   ├── combined_corpus_raw.txt
│   ├── Ground_Truth.xlsx          # Expert ground truth taxonomy workbook
│   ├── raw/                       # Original chapter MMD sources
│   └── benchmarks/                # LLM typing evaluation datasets
│
├── gold/                          # Standardized Ground Truth annotations
│   ├── GT1_domain_terms.csv       # Verified domain vocabulary
│   ├── GT2_alias_acronym.csv      # Standard acronym-expansion mappings
│   ├── GT3_concept_discovery.csv  # Verified topic concept clusters
│   ├── GT4_taxonomy.csv           # Reference system-subsystem-part hierarchies
│   └── terms_labels_GT.csv        # Multi-class annotated terms
│
├── scripts/                       # High-precision pipeline drivers
│   ├── run_precision_first_ontology_pipeline.py
│   ├── run_iterative_context_pipeline.py
│   ├── sae_sentence_concepts.py
│   └── make_demo_data.py
│
├── docs/                          # Comprehensive technical reports & guides
│   ├── precision_first_ontology_workflow.md
│   ├── PRESENTATION_SPEECH_SCRIPT.md
│   ├── TF_IDF_Ontology_Technical_Guide.pdf
│   ├── TOPK_SWEEP_README.md
│   ├── reports/                   # Quantitative stage benchmark reports
│   └── images/                    # Architecture diagrams & confusion matrices
│
└── tests/                         # Automated unit & integration tests
    └── test_smoke.py              # Rapid pipeline smoke test
```

---

## 🚀 6. Quickstart & Installation

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/DevanSF07/uav-ontology-learning.git
cd uav-ontology-learning

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e .

# (Optional) Install full model dependencies (GLiNER, UMAP, HDBSCAN, Spacy)
pip install -e '.[full]'
python -m spacy download en_core_web_sm
```

### Run Tests
```bash
pytest tests/ -v
```

### Run Fast Smoke Test Pipeline
```bash
# Generate lightweight test corpus
python scripts/make_demo_data.py

# Execute smoke pipeline through CLI
olbench run --config configs/smoke.json \
  --input data/demo_chapter5_chapter6.mmd \
  --out outputs/smoke

# Aggregate experiment metrics
olbench collect --out outputs/smoke
```

### Run Full Precision-First Production Pipeline
```bash
# Run core experiment suite
olbench run --config configs/core_experiments.json \
  --input data/chapter5_chapter6.mmd \
  --out outputs/production_run

# Collect and summarize stage metrics
olbench collect --out outputs/production_run
```

### Local LLM Integration (Ollama)
Ensure [Ollama](https://ollama.com/) is running locally with your target model:
```bash
ollama serve
ollama pull qwen2.5:7b-instruct
# or
ollama pull llama3.1:8b
```
The framework automatically communicates with `http://localhost:11434/api/generate` and includes graceful, rule-based fallback handlers if the local daemon is paused.

---

## 💼 7. Interview Talking Points & Technical Defense

**Q: Why use a "Precision-First" paradigm instead of simply training an end-to-end classifier?**  
*Answer:* In highly specialized engineering domains (such as unmanned avionics), annotated training data is scarce. Unsupervised extractors maximize recall (23.8%) but suffer from 88%+ false-positive rates. Attempting to train an end-to-end model on noisy labels causes severe semantic drift. By using extractors strictly as an *evidence reservoir*, applying GLiNER/MPNet boundary pruning, and requiring LLM evidence citation per sentence, we elevate precision to 58.4% before any taxonomy relations are drawn.

**Q: Why use Non-Negative Matrix Factorization (NMF) over standard PCA or Raw Embeddings for concept discovery?**  
*Answer:* Dense sentence embeddings (`all-mpnet-base-v2`) naturally contain negative components across their 768 dimensions. When clustering concepts, negative projections lack physical interpretability (a document cannot possess "negative occurrences" of a subsystem). NMF applied over non-negative TF-IDF chunk matrices guarantees additive, part-based topic representations, isolating clean orthogonal subsystems (e.g. Propulsion vs Ground Control Station vs Flight Avionics).

**Q: How do you prevent LLMs from hallucinating phantom taxonomy parents?**  
*Answer:* Stage 13a enforces strict **Corpus Grounding Verification**. Before any term is admitted as a `System` or `Subsystem` parent node, it must pass a 5-step cascade: (1) normalized exact corpus substring match, (2) conservative lemmatized inflection, (3) explicit ground-truth alias/acronym mapping, (4) controlled compound reconstruction, and (5) isolated single-term LLM validation. Semantic proximity alone is prohibited from inventing non-existent physical subsystems.

---

## 📜 8. License & Attribution

This project is open-sourced under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Citation & Academic Attribution
```bibtex
@misc{faujdar2026uavontology,
  author = {Faujdar, Devan Singh},
  title = {Automated UAV Domain Ontology Generation Using Hybrid Term Extraction and Multi-Agent LLM Systems},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/DevanSF07/uav-ontology-learning}}
}
```
