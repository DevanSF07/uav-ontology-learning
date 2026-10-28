# Precision-First Ontology Term Discovery with Recall Expansion

## 1. Goal
Use LLM chunk-level ontology evidence detection for precision first, then recover recall later using extractor terms, semantic similarity, clustering, and LLM revalidation.

Older approach:

```text
Document -> term extraction -> noise filtering -> ontology candidates
```

New approach:

```text
Chunk understanding -> evidence-backed terms -> semantic expansion -> revalidation -> promotion
```

YAKE, KeyBERT, TF-IDF and noun phrase extractor outputs are not treated as ontology terms directly. They are treated as term evidence reservoirs.

---

## 2. Target Ontology Scope
A chunk or term is relevant only if it supports one or more of:

```text
classification
mission_usecase
operating_condition
environment
scenario_operation
system
subsystem
component
part
flow
port
interface
```

Out of scope by default:

```text
regulation
certification
legal policy
country names
organisation names
history
market discussion
generic background without reusable system ontology value
```

---

## 3. Main Workflow

```text
1. Input cleaned_chunks.csv
2. Process each chunk independently
3. LLM Gate: Does chunk contain ontology evidence?
4. If NO: skip / reject / backlog chunk with reason
5. If YES: LLM extracts evidence sentences, concepts, terms and proposed types
6. Run extractors only on approved chunk
7. Compute concurrence: LLM + extractor + evidence support
8. Assign initial term state: Accepted / Candidate / Backlog / Rejected
9. Run embeddings over accepted, candidate and extractor-only terms
10. Cluster terms for aliases, variants and related concepts
11. Recover recall candidates from semantic neighbours and clusters
12. Revalidate recovered candidates with LLM using evidence sentence
13. Promote / candidate / backlog / reject
14. Consolidate final ontology candidates
15. Optional loop: newly promoted terms can seed another embedding and clustering pass
```

Important rule:

```text
Do NOT pass previous chunk terms into the next chunk prompt.
Each chunk is analysed independently.
Consolidation happens only after all chunks are processed.
```

---

## 4. LLM Gate Prompt

Use this for every chunk.

```text
You are an ontology engineer analysing one document chunk.

Goal:
Decide whether this chunk contains evidence useful for building a system ontology.

Target ontology categories:
- classification
- mission / use case / purpose
- operating condition
- environment
- scenario / operation
- system
- subsystem
- component
- part
- flow
- port
- interface

Out of scope unless directly connected to system ontology evidence:
- regulation
- certification
- legal compliance
- organisation names
- country names
- history
- market discussion
- generic background

Question 1:
Does this chunk contain direct or indirect evidence for at least one target ontology category?

Return JSON only:
{
  "gate_decision": "YES" or "NO",
  "reason": "one short reason",
  "possible_categories": ["..."],
  "out_of_scope_theme": "... or null"
}

Rules:
- Use only the given chunk.
- Do not use previous chunks.
- Do not invent concepts outside the text.
- If evidence is weak but potentially useful, return YES with low confidence later.
```

If gate_decision is NO:

```text
save to skipped_chunks.csv
status = skipped / rejected / backlog_chunk
reason = gate reason
```

Do not run extractors unless DEBUG_EXTRACT_SKIPPED_CHUNKS = True.

---

## 5. LLM Evidence Extraction Prompt

Run only for YES chunks.

```text
The chunk has passed the ontology evidence gate.

Extract ontology-relevant evidence from this chunk.

Return JSON only with this structure:
{
  "chunk_id": "...",
  "concepts_present": [
    {
      "category": "classification | mission_usecase | operating_condition | environment | scenario_operation | system | subsystem | component | part | flow | port | interface | other",
      "evidence_sentence": "exact sentence from chunk",
      "confidence": "high | medium | low"
    }
  ],
  "candidate_terms": [
    {
      "term": "...",
      "proposed_type": "classification | mission_usecase | operating_condition | environment | scenario_operation | system | subsystem | component | part | flow | port | interface | other",
      "evidence_sentence": "exact sentence from chunk",
      "confidence": "high | medium | low",
      "reason": "short reason"
    }
  ]
}

Rules:
- Extract only terms grounded in the chunk.
- Prefer explicit noun phrases.
- Allow multiple ontology categories in the same chunk.
- Type each term using its local evidence sentence, not the whole document.
- Do not include regulation, certification, organisation, country, or legal terms unless the sentence directly supports the target ontology.
- If a term is relevant but type is uncertain, mark confidence as low.
```

---

## 6. Extractors
Run only on chunks approved by the LLM gate.

Recommended methods:

```text
YAKE
KeyBERT / MPNet
TF-IDF
noun phrase extraction
```

Extractor terms are evidence only. They are not accepted automatically.

Recommended extractor output:

```text
chunk_id
term
normalised_term
extractor_name
extractor_score
source_sentence_if_available
```

---

## 7. Concurrence Scoring
Concurrence means the same or similar term is supported by multiple evidence sources.

Evidence sources:

```text
LLM extraction
YAKE
KeyBERT / MPNet
TF-IDF
noun phrase extraction
chunk evidence sentence
semantic similarity
cluster membership
co-occurrence
```

Suggested score:

```text
score = 0
+ 0.35 if LLM extracted term with evidence sentence
+ 0.15 per extractor that found the term
+ 0.15 if term appears in more than one approved chunk
+ 0.10 if semantically close to accepted term
+ 0.10 if belongs to strong cluster
+ 0.10 if co-occurs with accepted ontology terms
```

---

## 8. Term States

### ACCEPTED
Use when:

```text
LLM extracted term with high/medium confidence
evidence sentence exists
at least one extractor agrees OR strong concurrence exists
```

### CANDIDATE
Use when:

```text
LLM-only term with evidence but no extractor agreement
OR extractor term found by two or more extractors in approved chunks
```

### BACKLOG
Use when:

```text
weak evidence
uncertain type
semantically close but no direct evidence yet
needs more supporting chunks
```

### REJECTED
Use when:

```text
out of scope
regulation/certification/organisation/country/legal/history
no evidence sentence
generic phrase with no ontology value
```

### PROMOTE
Promotion is the action of moving a term from Candidate / Backlog / Recovered into Accepted after revalidation confirms evidence.

---

## 9. Embeddings Stage
Embedding occurs after initial high-confidence terms exist.

Do not start expansion from only "UAV".

Embed:

```text
accepted terms
candidate terms
extractor-only terms from approved chunks
chunk evidence sentences
```

Recommended model:

```text
MPNet or the same embedding model already used in the pipeline
```

Similarity stages:

### 9A. Local similarity
Search within the same approved chunk or neighbouring approved chunks first.

```text
accepted term -> nearby extractor terms in approved chunks
```

### 9B. Cluster-level similarity
After local expansion, search inside the same semantic cluster.

```text
accepted cluster -> candidate terms in same cluster
```

### 9C. Global similarity, optional
Use only for analysis, not direct promotion.

```text
accepted term -> whole document term pool
```

Global expansion must be revalidated before promotion.

---

## 10. Clustering Stage
Cluster terms using embeddings and optionally co-occurrence.

Input:

```text
accepted terms
candidate terms
extractor-only terms from approved chunks
```

Methods:

```text
Agglomerative clustering
HDBSCAN
KMeans for controlled comparison
```

Cluster output:

```text
cluster_id
canonical_term
aliases
member_terms
supporting_chunks
dominant_proposed_type
cluster_confidence
```

Clustering is used for:

```text
alias grouping
variant grouping
finding semantically related missed terms
duplicate detection
recall expansion
```

Clustering should not directly accept terms.

---

## 11. Recall Expansion
Use extractor terms as recall reservoir.

A recovered term can be proposed if:

```text
it was extracted by YAKE / KeyBERT / TF-IDF / noun phrase extractor
AND appears in an approved or related chunk
AND is semantically close to an accepted term or accepted cluster
AND has an evidence sentence
```

Recovered terms go into:

```text
revalidation_queue.csv
```

They are not accepted directly.

---

## 12. Revalidation Prompt
Use for recovered / expanded terms.

```text
You are validating a recovered ontology candidate term.

Chunk evidence:
{evidence_sentence}

Candidate term:
{term}

Nearby accepted terms or cluster label:
{cluster_context}

Question:
Is this candidate term a valid ontology-relevant term for one of these categories?
- classification
- mission_usecase
- operating_condition
- environment
- scenario_operation
- system
- subsystem
- component
- part
- flow
- port
- interface

Return JSON only:
{
  "decision": "PROMOTE | CANDIDATE | BACKLOG | REJECT",
  "proposed_type": "...",
  "confidence": "high | medium | low",
  "reason": "short reason",
  "required_evidence": "what evidence is missing, if any"
}

Rules:
- PROMOTE only if the evidence sentence directly supports the term and type.
- CANDIDATE if plausible but not enough agreement.
- BACKLOG if useful but needs more evidence or type is unclear.
- REJECT if out of scope or unsupported.
```

---

## 13. Loop-back Points

### Loop 1: Candidate -> Revalidation

```text
candidate term -> evidence sentence -> LLM revalidation -> promote/candidate/backlog/reject
```

### Loop 2: Accepted terms -> Embedding expansion

```text
accepted terms -> semantic neighbours -> recovered candidates -> revalidation
```

### Loop 3: Consolidated clusters -> More recall search

```text
accepted clusters -> extractor-only terms in same semantic region -> revalidation
```

Important:

```text
No loop should directly accept a term without evidence revalidation.
```

---

## 14. Output Files

### chunk_analysis.csv

```text
chunk_id
gate_decision
gate_reason
concepts_present
out_of_scope_theme
source_doc
section
page
```

### chunk_terms.csv

```text
chunk_id
term
normalised_term
proposed_type
evidence_sentence
llm_confidence
source_methods
extractor_count
initial_state
final_state
confidence_score
```

### extractor_terms.csv

```text
chunk_id
term
normalised_term
extractor_name
extractor_score
source_sentence
```

### term_evidence.csv

```text
term
normalised_term
chunk_id
evidence_sentence
source_method
ontology_type
confidence
state
```

### semantic_neighbours.csv

```text
anchor_term
candidate_term
cosine_similarity
scope
chunk_id
cluster_id
state_after_revalidation
```

### term_clusters.csv

```text
cluster_id
canonical_term
member_terms
aliases
supporting_chunks
dominant_type
cluster_confidence
```

### revalidation_queue.csv

```text
term
evidence_sentence
cluster_context
reason_for_revalidation
similarity_score
current_state
```

### final_ontology_candidates.csv

```text
canonical_term
aliases
ontology_type
final_state
confidence_score
supporting_chunks
evidence_count
source_methods
reason
```

---

## 15. Suggested Config

```python
TOP_K_TERMS_PER_CHUNK = 10
MIN_EXTRACTOR_COUNT_FOR_CANDIDATE = 2
MIN_SIMILARITY_LOCAL = 0.72
MIN_SIMILARITY_CLUSTER = 0.68
MIN_SIMILARITY_GLOBAL = 0.78
PROMOTION_SCORE_THRESHOLD = 0.70
CANDIDATE_SCORE_THRESHOLD = 0.45
USE_GLOBAL_EXPANSION = False
REVALIDATE_ALL_EXPANSIONS = True
PASS_PREVIOUS_CONTEXT_TO_NEXT_CHUNK = False
```

---

## 16. Functions Copilot Should Implement

```python
load_chunks()
build_gate_prompt(chunk)
call_llm_gate(prompt)
build_evidence_prompt(chunk)
call_llm_evidence(prompt)
run_yake(chunk)
run_keybert(chunk)
run_tfidf(chunk)
run_noun_phrase_extraction(chunk)
normalise_terms(terms)
compute_concurrence(llm_terms, extractor_terms)
assign_initial_state(term_record)
embed_terms(term_records)
find_semantic_neighbours(anchor_terms, candidate_terms)
cluster_terms(term_embeddings)
build_revalidation_prompt(candidate)
call_llm_revalidation(prompt)
update_term_state(term, decision)
consolidate_terms()
save_outputs()
```

Restartability:

```text
If chunk_id already processed, skip unless FORCE_RERUN = True.
```

Auditability:

```text
Save all prompts and raw LLM responses.
Keep term -> sentence -> chunk -> document provenance.
```

---

## 17. Final Summary

```text
LLM decides whether the chunk is ontology-bearing.
LLM extracts grounded terms with evidence.
Extractors provide concurrence, not final truth.
Embeddings recover semantically close missed terms.
Clustering groups aliases and related terms.
Recovered terms are revalidated before promotion.
Accepted terms are always evidence-backed.
```
