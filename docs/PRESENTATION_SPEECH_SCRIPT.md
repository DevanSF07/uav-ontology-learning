# Oral Defense Speech Script & Presentation Guide
## Automated UAV Domain Ontology Generation Using Hybrid Term Extraction and Multi-Agent LLM Systems
**Candidate:** Devan Singh Faujdar (Roll No: 22CH30046)  
**Department:** Department of Chemical Engineering, IIT Kharagpur  
**Host Organization:** Synopsys Inc.  
**Company Supervisor:** Mr. Vinod Rao (with guidance from Ms. Sandhya Seshagiri)  
**Target Duration:** 12 – 14 minutes presentation + 5 minutes Q&A  

---

## Defense Strategy & Delivery Guidelines
1. **Pacing:** Speak at a calm, deliberate pace (~120–130 words per minute). Do not rush through the numbers.
2. **Body Language & Pointers:** When mentioning figures (such as the pipeline on Slide 4, the Before vs. After metrics on Slide 6, or the core taxonomy graph on Slide 10), actively point to the specific cards and nodes.
3. **Presenter View:** All talking points below are also embedded directly into the **PowerPoint Speaker Notes** in your `.pptx` file. If you use dual-screen / Presenter View, this script will be right in front of you.

---

## Slide-by-Slide Spoken Script

### Slide 01: Title Slide
* **Time:** `0:00 – 0:45` (45 seconds)
* **Visual Focus:** Title, IIT Kharagpur crest, Synopsys logo, roll number, and supervisor details.

> *"Respected members of the evaluation committee, faculty members, and fellow students: Good morning/afternoon.*
>
> *I am Devan Singh Faujdar, a third-year undergraduate in the Department of Chemical Engineering at IIT Kharagpur. Today, I am honored to present my internship defense titled: **'Automated UAV Domain Ontology Generation Using Hybrid Term Extraction and Multi-Agent LLM Systems'**.*
>
> *This work was carried out during my industrial internship at Synopsys, under the supervision of Mr. Vinod Rao, with valuable technical discussions and guidance from Ms. Sandhya Seshagiri.*
>
> *In this presentation, I will walk you through how we designed, implemented, and benchmarked an end-to-end, evidence-grounded framework that transforms unstructured technical aerospace literature into a structured, typed taxonomy of Unmanned Aerial Systems using local Large Language Models."*

---

### Slide 02: Presentation Outline
* **Time:** `0:45 – 1:15` (30 seconds)
* **Visual Focus:** Structured 10-part agenda grid.

> *"Here is the roadmap for today's presentation.*
>
> *We will begin with the industrial motivation and our explicit research goal. We will then examine the complete 4-layer system architecture.*
>
> *Next, I will detail the four sequential pipeline stages: Stage I candidate extraction, Stage II Qwen LLM validation where we improved precision by nearly five-fold, Stage III multi-agent entity typing into 25 official categories, and Stage IV multi-agent taxonomy induction where we synthesized the actual System-Subsystem hierarchy.*
>
> *Finally, we will analyze the UAV semantic neighborhoods, conclude with our technical contributions, and discuss the future production roadmap."*

---

### Slide 03: Problem Statement, Motivation & Research Goal
* **Time:** `1:15 – 2:30` (1 minute 15 seconds)
* **Visual Focus:** Top left card (Motivation), Top right card (Bottlenecks), Bottom highlighted card (Primary Research Goal).

> *"Let us first understand the industrial context and the core engineering challenge.*
>
> *In complex engineering sectors like aerospace, technical literature—ranging from flight manuals and system specifications to regulatory airworthiness standards—contains a wealth of structured domain knowledge. However, this knowledge is locked in dense, unstructured prose. Building domain ontologies manually requires weeks of subject-matter expert time, making it expensive, brittle, and difficult to scale.*
>
> *The Unmanned Aerial Vehicle (UAV) domain is a particularly demanding testbed. Its terminology is deeply heterogeneous: it mixes vehicle classes like HALE and MAVs, multi-layered avionics subsystems, mission types, operating parameters, and civil aviation regulations.*
>
> *Existing automated NLP approaches face three major bottlenecks:*
> 1. *First, statistical extractors suffer from massive noise, yielding precision as low as 7%.*
> 2. *Second, engineering concepts suffer from severe category overlap—words like 'System', 'Subsystem', and 'Interface' frequently blur together in technical text.*
> 3. *Third, free-form LLM pipelines tend to hallucinate relations without verifiable provenance.*
>
> *To solve this, our **Primary Research Goal** was to build an automated, evidence-grounded pipeline that extracts domain terminology, contextually validates every term using local LLMs, types them into 25 official categories, and induces an auditable System/Subsystem taxonomy—with zero data leakage and 100% corpus grounding."*

---

### Slide 04: End-to-End System Architecture
* **Time:** `2:30 – 3:30` (1 minute)
* **Visual Focus:** Large comprehensive flowchart (`image_16.png`), pointing to Stage 1 (top), Stage 2 (bottom), and key principles.

> *"This slide illustrates our complete end-to-end architecture, spanning four computational layers.*
>
> *Looking at the top half—**Stage 1**—raw technical corpus text is ingested. We first extract candidate terms using a multi-extractor ensemble. These candidates are passed to a local Qwen 2.5 7B model for contextual validation, followed by conservative protection rules to produce a refined inventory of 564 concepts.*
>
> *Every retained concept then enters the **Multi-Agent Entity Typing Pipeline**, where context retrieval grounds the candidate into 5 retrieved sentences, a Candidate-Type Agent shortlists five categories, a Primary Agent makes the decision, and a Critic Agent verifies edge cases.*
>
> *Looking at the lower half—**Stage 2**—terms typed as System or Subsystem are filtered, canonicalized into stable nodes, paired with candidate parents using MPNet embeddings, classified via relation agents into `is_a` and `has_subsystem` links, and cleaned through deterministic graph validation.*
>
> *Three architectural principles guided this entire design:*
> 1. *Durable modular checkpoints at every stage.*
> 2. *Evidence-grounded decisions tied to exact sentence IDs.*
> 3. *Strict evaluation separation: ground-truth was never visible during model inference; it was attached purely post-prediction."*

---

### Slide 05: Stage I: Candidate Term Extraction & Baseline Performance
* **Time:** `3:30 – 4:45` (1 minute 15 seconds)
* **Visual Focus:** Left card (Ensemble details), Top right metric cards (68% Recall vs 7% Precision), Funnel chart (`image_13.png`).

> *"Now, let us dive into Stage I: Candidate Term Extraction.*
>
> *Rather than relying on a single keyword algorithm, we deployed an ensemble of six complementary methods spanning three distinct algorithmic families:*
> * Local statistical salience via YAKE,*
> * Corpus-wide distribution via TF-IDF,*
> * Graph-based phrase centrality via TextRank and Noun Phrases,*
> * And dense transformer semantics using KeyBERT with MiniLM and MPNet.*
>
> *Through weighted voting and deduplication, this stage extracted **1,893 candidate terms** in just 9 minutes—making it exceptionally fast and computationally inexpensive.*
>
> *However, look at the benchmark numbers on the top right:*
> *Our YAKE standalone baseline achieved a high recall of approximately **68%**, meaning it captured the vast majority of domain concepts. But its standalone precision was only **~7%**, resulting in an F1-score of just **12.7%**.*
>
> *The reason is clear: unsupervised statistical extractors optimize for frequency and casing. They inevitably pull in document formatting artifacts, generic nouns, and partial sentence fragments.*
>
> *This proved our first hypothesis: extraction alone is insufficient for ontology learning. We needed an intelligent, evidence-grounded semantic validation layer."*

---

### Slide 06: Stage II: Qwen LLM Validation Layer — Before vs. After
* **Time:** `4:45 – 6:00` (1 minute 15 seconds)
* **Visual Focus:** Side-by-side comparison cards (YAKE vs Qwen Context Extended), Bar chart (`image_01.png`), Bottom protection rules box.

> *"Stage II introduces our Qwen LLM Validation Layer, and the results on this slide represent one of our primary performance breakthroughs.*
>
> *We prompted a local Qwen 2.5 7B model with each candidate term alongside five retrieved corpus context sentences. The model scored each term on domain specificity, genericity, and ontology suitability, outputting an initial retain, review, or reject verdict.*
>
> *Crucially, we implemented a **Conservative Protection Protocol**. We recognized that a vanilla LLM might prematurely reject valid technical acronyms like 'HALE' (High Altitude Long Endurance) or 'MUAV' simply because they are infrequent. Our protection rules converted uncertain rejects into a 'review' status if the term had acronym morphology, was supported by three or more extractors, or appeared repeatedly across technical contexts.*
>
> *Now, look at the Before versus After metrics:*
> * Before validation, the YAKE baseline had **7.0% precision** and **12.7% F1**.*
> * After Qwen contextual validation, our precision jumped to **33.87%**—an absolute gain of nearly 27%, representing a **4.8-fold precision boost**!*
> * Recall remained strong at **52.17%**, and the overall F1-score surged to **41.07%**—more than triple the baseline.*
>
> *Applying a score threshold of 0.60 reduced 1,893 noisy candidates down to a curated, high-confidence inventory of **564 validated domain concepts**."*

---

### Slide 07: Stage III: Multi-Agent Entity Typing Architecture & Classes
* **Time:** `6:00 – 7:15` (1 minute 15 seconds)
* **Visual Focus:** Top definition card (25 classes clustered into 4 groups), Multi-agent pipeline diagram (`image_20.png`), Bottom agent role callouts.

> *"With 564 validated concepts in hand, Stage III tackles Entity Typing.*
>
> *Before we can connect concepts into an ontology, we must know what each concept denotes. For example, in aerospace systems, 'UAV' is a physical platform or System; 'Avionics' is a Subsystem; 'Autonomous Flight' is an Operational Activity; and 'Airworthiness' is a Regulatory Concept. Conflating these leads to nonsensical taxonomies.*
>
> *We operated over a closed vocabulary of **25 official entity classes**, grouped into four functional categories shown at the top: Hardware/Structural, Operational, Quantitative, and Governance.*
>
> *To type all 564 terms accurately, we designed a cooperative **Three-Agent Multi-Agent Architecture**:*
> 1. *First, the **Candidate-Type Agent** receives the term and five MPNet-retrieved sentences. Its job is not final precision, but high recall: it shortlists the **Top-5 most plausible classes** out of the 25.*
> 2. *Second, the **Primary-Typing Agent** compares only those five shortlisted profiles, selecting the single winning type along with an explicit rationale and evidence citation.*
> 3. *Third, if the primary agent's confidence is below threshold or evidence is conflicting, the **Critic Agent** is invoked to perform an adversarial re-evaluation.*
>
> *Importantly: every one of the 564 concepts was typed through this pipeline before ground truth was ever consulted."*

---

### Slide 08: Stage III Results: Entity Typing Accuracy & Compute Profile
* **Time:** `7:15 – 8:30` (1 minute 15 seconds)
* **Visual Focus:** Typing metrics chart (`image_15.png`), Compute time breakdown (`image_07.png`), Bottom evaluation cards.

> *"This slide presents our quantitative evaluation for entity typing, benchmarked on 175 unambiguous ground-truth matches.*
>
> *Looking at the green bar at the top of the left chart:*
> *Our Candidate-Type Agent achieved a remarkable **92.00% Top-5 Candidate Recall**. In other words, in more than 9 out of 10 cases, the correct class was successfully captured in the candidate shortlist.*
> *Candidate Top-3 recall was **86.86%**, and Top-1 was **65.71%**.*
> *After primary selection and critic arbitration, our final classification reached **68.57% accuracy**, with a Macro-F1 of **62.03%** and Weighted-F1 of **63.99%** across the 25 classes.*
>
> *Now, examine the gap between the 92% shortlist recall and the 68.57% final accuracy. Our error analysis revealed that this bottleneck is primarily driven by semantic bleed between adjacent classes—such as distinguishing whether an item is best labeled a 'System' versus a 'Subsystem', or a 'Subsystem' versus an internal 'Part'.*
>
> *Turning to the right chart, we see our compute-time profile:*
> *Total end-to-end wall-clock time was **31 hours and 56 minutes** on a single workstation with local Qwen 2.5 7B.*
> *Notice that statistical extraction took under 10 minutes—less than 0.5% of total runtime. Over 99% of compute was dominated by sequential LLM inference (14.4 hours for validation, 8.2 hours for typing, and 9.1 hours for taxonomy).*
> *We intentionally retained sequential, single-term inference to ensure complete crash-resilience, checkpointing, and auditability."*

---

### Slide 09: Stage IV: Multi-Agent Taxonomy Induction Architecture & Pipeline
* **Time:** `8:30 – 9:45` (1 minute 15 seconds)
* **Visual Focus:** Full detailed taxonomy induction architecture flowchart (`pdf_p13_0_Image58.jpg`), Bottom workflow summary.

> *"Now we arrive at Stage IV: Multi-Agent Taxonomy Induction, which takes our validated, typed inventory and synthesizes the actual hierarchical knowledge graph.*
>
> *The complete pipeline is mapped out on this slide across nine modular steps:*
> 1. *First, we filter all terms predicted as **System** or **Subsystem** from our 564-concept pool.*
> 2. *The **Node Canonicalization Agent** resolves acronyms, merges singular and plural variants, and establishes canonical node identifiers.*
> 3. *To avoid an intractable combinatorial explosion of comparing thousands of node pairs, our **Candidate Parent Retrieval** module embeds each node with MPNet and retrieves only the Top-10 structurally permitted parents. For instance, a System can only have another System as an `is_a` parent; a Subsystem can have a Subsystem as an `is_a` parent, or a System as a `has_subsystem` parent.*
> 4. *The **Relation Classification Agent** evaluates each pair one at a time under strict linguistic tests: kind-of generality, substitutability, and compositional containment.*
> 5. *The **Relation Critic Agent** independently audits positive relations, verifying the directionality of the edge.*
> 6. *Finally, **Graph Validation and Transitive Reduction** strips out duplicate edges, breaks cycles, and eliminates transitive shortcuts—ensuring that if concept A connects to B, and B connects to C, the redundant direct edge from A to C is removed."*

---

### Slide 10: Stage IV Results: Generated Taxonomy Graph & Structural Metrics
* **Time:** `9:45 – 11:00` (1 minute 15 seconds)
* **Visual Focus:** Core generated graph on the left (`pdf_p16_1_Image67.png`), Node/edge composition charts on the right (`pdf_p16_0_Image66.png`), Bottom metric summary.

> *"This slide showcases the primary deliverable of our taxonomy induction stage: the actual induced graph and its structural metrics.*
>
> *Looking at the graph visualization on the left:*
> *You can observe the representative core of our generated System-Subsystem hierarchy.*
> *At the root level, the system correctly established **'Unmanned System'**, which branches into **'Aircraft'** and **'UAS'**.*
> *'UAS' branches cleanly via `is_a` edges into specialized variants like **'UAV System'**, **'Military UAS'**, **'RPAS'**, and specific platforms like **'Shadow UAS'**.*
> *Attached to these systems through orange dashed `has_subsystem` compositional edges are the key operational subsystems: **'Avionics'**, **'Flight Control'**, **'Ground Control Station (GCS)'**, **'Remote Pilot Station'**, and **'Radar Systems'**.*
>
> *Examining the quantitative deliverables:*
> * The induction stage produced **74 canonical taxonomy nodes**: 62 System nodes (83.8%) and 12 Subsystem nodes (16.2%).*
> * Out of 860 candidate pairs evaluated, the system accepted **181 verified edges**:*
>   * **58** System `is_a` System edges,*
>   * **9** Subsystem `is_a` Subsystem edges,*
>   * And **114** System `has_subsystem` Subsystem composition edges.*
> * Graph quality metrics showed excellent coherence: **2 root concepts** and only **2 orphan nodes**, giving an orphan rate of just **2.7%**.*
>
> *In terms of error analysis, relation agents occasionally interpreted operational associations—such as 'uses', 'transports', or 'communicates_with'—as physical `has_subsystem` containment. Addressing this nuance forms a key part of our future work."*

---

### Slide 11: Exploratory Analysis: UAV Semantic Neighborhoods & Radial Map
* **Time:** `11:00 – 11:45` (45 seconds)
* **Visual Focus:** Large radial semantic map (`image_09.png`), Right card (8 dimensions).

> *"In addition to direct hierarchy induction, we conducted an exploratory semantic neighborhood analysis, visualized in this radial embedding map.*
>
> *Each System concept was embedded alongside its retrieved corpus context using MPNet. Radius represents cosine distance from the central concept 'UAV', while angle reflects shared semantic PCA direction.*
>
> *This visualization reveals **eight distinct, human-interpretable classification dimensions**:*
> *Use Domain, Size and Weight Class, Platform Type, Mission Capability, System vs Vehicle Concepts, Autonomy Level, Performance Characteristics, and Regulatory Airspace Context.*
>
> *The fundamental ontological takeaway here is that technical engineering domains are inherently **polyhierarchical**. A single rigid classification tree hides valid alternative perspectives. These eight dimensions provide the blueprint for multi-parent polyhierarchies in future work."*

---

### Slide 12: Conclusions, Technical Contributions & Future Scope
* **Time:** `11:45 – 12:45` (1 minute)
* **Visual Focus:** Executive conclusion banner, Left card (Technical Contributions), Right card (Future Roadmap).

> *"To summarize our conclusions, contributions, and future scope:*
>
> *In **conclusion**, this internship delivered a fully functional, reproducible, and ground-truth-evaluated pipeline for automated aerospace ontology engineering. We proved that high-recall extraction, contextual Qwen validation, and multi-agent typing can effectively discover and structure technical concepts without catastrophic hallucination or ground-truth leakage.*
>
> *Our **key technical contributions** include:*
> 1. *A hybrid 6-extractor ensemble capturing candidates in under 10 minutes.*
> 2. *A conservative protection protocol that increased precision nearly five-fold.*
> 3. *A cooperative 3-agent typing architecture achieving 92% Top-5 recall across 25 classes.*
> 4. *An end-to-end taxonomy generator synthesizing 181 verified relations with full audit trails.*
>
> *For **future scope**, we propose:*
> * Implementing 4-bit quantization and vLLM batching to reduce the 32-hour runtime by over 5x.*
> * Introducing finer-grained relation prompts to distinguish physical composition from operational association.*
> * And exporting the validated hierarchy into OWL/RDF formats to enable formal description logic reasoning and SPARQL querying."*

---

### Slide 13: References & Academic Literature
* **Time:** `12:45 – 13:00` (15 seconds)
* **Visual Focus:** Bibliographic references card.

> *"Here are the primary literature foundations that informed our methodologies: including Campos et al. for YAKE keyword extraction, Reimers and Gurevych for Sentence-BERT, Song et al. for MPNet embeddings, Grootendorst for KeyBERT, Alibaba Cloud's technical report for Qwen 2.5, and internal Synopsys technical standards."*

---

### Slide 14: Thank You & Discussion
* **Time:** `13:00+`
* **Visual Focus:** Thank You card, institutional affiliation, supervisor acknowledgment.

> *"I would like to express my sincere gratitude to my supervisor Mr. Vinod Rao and Ms. Sandhya Seshagiri at Synopsys, and to the Department of Chemical Engineering at IIT Kharagpur for supporting this internship.*
>
> *Thank you very much for your time and attention. I am now open to your questions and comments."*

---

## Anticipated Committee Questions & High-Scoring Defense Answers

### Q1: *"Your entity-typing accuracy is 68.57%. Why is it not higher, and is this acceptable for an automated system?"*
> **Answer:**
> *"That is a very perceptive question, Sir/Madam. To put 68.57% into context: this is across **25 fine-grained, highly specialized technical classes**—not a simple binary or 3-class classification problem. Random guessing across 25 classes would yield only 4% accuracy.
>
> More importantly, our **Candidate Top-5 Recall was 92.00%**, and Top-3 recall was 86.86%. This proves that the model almost never makes catastrophic mistakes—the true domain type is virtually always in the top shortlist.
>
> Our detailed confusion matrix showed that the remaining errors are concentrated entirely on **semantically adjacent boundaries**—specifically between `System` vs `Subsystem`, and `Subsystem` vs `Part`. In engineering English, authors frequently refer to an avionics package as a 'system' in one chapter and a 'subsystem' in another. In a production pipeline, this 92% shortlist recall allows an engineer to confirm the correct class with a single click, saving ~90% of manual curation time."*

---

### Q2: *"Why did you use a local 7B open-source model (Qwen 2.5) instead of an API like GPT-4 or Claude?"*
> **Answer:**
> *"We chose local execution for three critical engineering reasons:
> 1. **Data Confidentiality & IP Security:** In industrial environments like Synopsys and aerospace clients, proprietary technical literature and design specifications cannot be uploaded to public third-party commercial APIs.
> 2. **Determinism & Reproducibility:** Commercial APIs undergo silent model updates, meaning the same prompt can yield different outputs across weeks. With local Qwen 2.5 7B, temperature=0, and fixed seeds, our entire experiment is 100% reproducible.
> 3. **Cost & Rate Limiting:** Our pipeline executed thousands of structured queries across term validation, multi-agent typing, and relation induction. Running this locally incurred zero API cost and zero rate-limiting interruptions."*

---

### Q3: *"In Slide 6, how exactly did your Conservative Protection Rules increase precision from ~7% to ~34%?"*
> **Answer:**
> *"The raw statistical extractor (YAKE) achieved 68% recall but 7% precision because it pulled in generic nouns like 'method', 'speed', and sentence fragments.
>
> When Qwen evaluated these 1,893 candidates, it correctly flagged the overwhelming majority of generic words as 'reject'. However, we observed that a raw LLM would occasionally also reject valid, niche aerospace acronyms like 'HALE' (High Altitude Long Endurance) or 'MUAV' because of their low frequency.
>
> Our conservative protection rules acted as a safety net: instead of permanently deleting low-confidence rejects, any term with acronym syntax, multi-extractor agreement (≥3 extractors), or high technical repetition was diverted into a 'review' pool. We then applied a strict confidence threshold of 0.60. This two-tier gating allowed us to purge ~70% of the extraction noise while preserving gold domain concepts, driving our precision from 7% to 33.87% and F1 from 12.7% to 41.07%."*

---

### Q4: *"How did you prevent ground-truth data leakage during the multi-agent typing and taxonomy stages?"*
> **Answer:**
> *"This was an absolute design priority. In our pipeline, **all 564 retained terms were typed**, and all terms classified as System/Subsystem entered taxonomy generation—irrespective of whether they existed in our ground-truth workbook.
>
> At no point during prompt construction, context retrieval, or multi-agent scoring were ground-truth labels injected or visible to the models. Ground-truth matching occurred **strictly post-prediction** as an offline evaluation step on the 175 unambiguous matches. This guarantees that our reported 68.57% accuracy and 92% Top-5 recall represent genuine generalization on unseen technical text."*

---

### Q5: *"Why does your compute time take ~32 hours? Can this run in a practical enterprise pipeline?"*
> **Answer:**
> *"Over 99% of the 31h 56m runtime was spent in sequential LLM inference. We intentionally chose sequential term-by-term and pair-by-pair execution during the research phase because it guaranteed complete fault tolerance: every decision was written immediately to disk with full JSON manifests, allowing safe resumption after any crash without losing state.
>
> In an enterprise deployment, this compute footprint can be easily compressed by at least 5x to 10x using three standard techniques:
> 1. **vLLM PagedAttention with batched inference**, allowing 16 to 32 candidates to be processed simultaneously.
> 2. **4-bit AWQ or GPTQ quantization**, doubling tokens-per-second throughput on commodity GPUs.
> 3. **Prompt prefix caching**, reusing cached KV-states for the static 25 type profiles rather than re-encoding them for every candidate."*

---

### Q6: *"How do you distinguish an `is_a` hierarchical relationship from a `has_subsystem` compositional relationship?"*
> **Answer:**
> *"This distinction is enforced through both our schema constraints and our relation classification prompts:
> 1. **Structural Type Filtering:** `is_a` is strictly permitted only between identical metamodel classes (`System is_a System` or `Subsystem is_a Subsystem`). A Subsystem can never be an `is_a` child of a System. Cross-class links between System and Subsystem are restricted to `has_subsystem`.
> 2. **Linguistic Entailment Prompts:** For `is_a`, the agent applies the classic substitution and kind-of test: *'Is every instance of child strictly an instance of parent?'* (e.g. 'A Military UAS is a kind of UAS' $\rightarrow$ True).
> 3. For `has_subsystem`, the agent applies the composition and functional containment test: *'Is the child a physical or functional component integrated within the parent vehicle?'* (e.g. 'Avionics is a subsystem of Aircraft' $\rightarrow$ True).
>
> Our Relation Critic then independently reviews proposed edges to ensure that operational associations (like 'transports' or 'communicates_with') do not leak into `has_subsystem`."*

---

### Q7: *"What is Transitive Reduction in Slide 9, and why is it necessary for taxonomy induction?"*
> **Answer:**
> *"When LLMs evaluate candidate pairs pairwise, they frequently validate both direct and indirect relationships. For example, the model might confirm:
> 1. `RPAS is_a UAS`,
> 2. `UAS is_a Unmanned System`,
> 3. and also directly predict `RPAS is_a Unmanned System`.
>
> If we keep all three edges, the graph becomes dense with redundant shortcut connections. **Transitive reduction** is a deterministic graph algorithm that removes the direct shortcut edge `RPAS is_a Unmanned System` because the relationship is already inherited transitively through `UAS`. This ensures our induced taxonomy is a clean, minimal, and interpretable directed acyclic graph."*

---

### Q8: *"As a Chemical Engineering student, how did you approach and master this Computer Science / NLP project?"*
> **Answer:**
> *"Chemical engineering training places immense emphasis on **systems engineering, process flowsheets, multi-stage separations, and mass balances**.
>
> I approached this NLP challenge through that exact engineering lens:
> * Candidate extraction is essentially a high-throughput raw feed separation stage.
> * LLM validation and conservative protection act as selective purification and filtration stages, rejecting generic noise while preserving high-value product.
> * Multi-agent entity typing is a multi-component distillation into 25 discrete fractions.
> * And taxonomy induction is the synthesis of the final structural flowsheet.
>
> By applying rigorous mathematical benchmarking, error analysis, and pipeline modularity, I was able to bridge natural language processing with systematic engineering methodology."*
