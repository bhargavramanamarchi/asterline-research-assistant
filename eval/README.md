# Asterline Research Assistant: Evaluation Framework

Comprehensive evaluation suite for the Asterline Research Assistant, benchmarking multi-agent routing, vector retrieval recall, evidence-grounded answer generation, citation validity, persistent memory isolation, and temporal document conflict resolution.

---

## 1. Overview & Architecture

The evaluation framework provides automated, end-to-end quantitative assessment of the Asterline multi-agent research and memory system. It validates the complete LangGraph execution graph (`PlannerNode` &rarr; `MemoryNode` / `ResearcherNode`), verifying that the agent reliably adheres to official documentation, resolves temporal policy conflicts, and strictly enforces user memory isolation.

### Key Components

| File | Purpose |
| :--- | :--- |
| [`eval/benchmark_dataset.json`](file:///d:/asterline-research-assistant/eval/benchmark_dataset.json) | 15 curated benchmark evaluation cases covering 8 core capabilities. |
| [`eval/eval_runner.py`](file:///d:/asterline-research-assistant/eval/eval_runner.py) | Automated evaluation runner executing graph workflows, computing metrics, and running the A/B improvement experiment. |
| [`eval/results.json`](file:///d:/asterline-research-assistant/eval/results.json) | Serialized execution results, containing per-case outputs, traces, citation validations, and aggregate scores. |
| [`eval/README.md`](file:///d:/asterline-research-assistant/eval/README.md) | Documentation of evaluation methodology, metric calculations, A/B experiment findings, failure analysis, and limitations. |

---

## 2. Benchmark Dataset Composition

The benchmark dataset consists of 15 carefully curated questions grounded exclusively in the 75 chunks of the official Asterline knowledge base (`corpus.jsonl`). The dataset is partitioned into document-grounding tests and user-memory operations:

| Case ID | Category | Type | Query Summary | Target Chunks / Key Facts |
| :--- | :--- | :--- | :--- | :--- |
| **BENCH-01** | Factual Retrieval | Research | Enterprise monthly uptime SLA and service credit tiers | `sla:0` (99.92%, 10%, 25%, 50% tiers) |
| **BENCH-02** | Temporal Supersession | Research | Cold recovery storage retention period after searchable window | `retention:0`, `security:1` (20 days supersedes 60 days) |
| **BENCH-03** | Version Conflict | Research | AQL query timeout limits across product versions | `query:0`, `release-23:0`, `release-20:2` (v2.3: 50s, earlier: 25s) |
| **BENCH-04** | Negative Grounding | Research | Asia-Pacific region release date / deployment timeline | `faq:1`, `regions:2` (explicitly unannounced / no date) |
| **BENCH-05** | Factual Retrieval | Research | Signals alert rule evaluation interval and anomaly history | `alerts:0` (4 minutes interval, 21 days history) |
| **BENCH-06** | Temporal Policy Update | Research | Weekly on-call stipend compensation update | `handbook:1`, `oncall:0` ($175 supersedes $140) |
| **BENCH-07** | Pricing Retrieval | Research | Monthly subscription costs for Launch and Team tiers | `pricing:0` (Launch: $59, Team: $449) |
| **BENCH-08** | Multi-Step Policy | Research | Planned maintenance exclusion & credit request window | `sla:2` (72 hours notice, 20 calendar days) |
| **BENCH-09** | Claim Refutation | Research | Refutation of false 100% uptime SLA guarantee claims | `sla:0` (refutes 100% claim; max is 99.92%) |
| **BENCH-10** | Pure Memory Store | Memory | Store preferred export format as Parquet | SQLite store (`preferred export format` = `Parquet`) |
| **BENCH-11** | Pure Memory Recall | Memory | Retrieve stored export preference for User Alpha | SQLite recall (`Parquet`) |
| **BENCH-12** | Pure Memory Update | Memory | In-place correction of preference from Parquet to CSV | SQLite update (`preferred export format` = `CSV`) |
| **BENCH-13** | User Isolation | Memory | User Beta attempts to recall User Alpha's export preference | Negative recall (no memory recorded for User Beta) |
| **BENCH-14** | Hybrid Research | Hybrid | Store tier as Enterprise and research contractual monthly SLA | Node trace: `[planner, memory_manager, researcher]` |
| **BENCH-15** | Precedence Invariance| Hybrid | Conflicting memory ("assumed SLA 100%") vs official docs | Grounded fact (official 99.92% SLA overrides memory note) |

---

## 3. Metrics & Calculation Methodology

The evaluation runner computes four primary metrics across the benchmark suite:

### 1. Agent Routing Accuracy (%)
- **Definition:** Measures whether the `PlannerNode` correctly classifies the user's intent (`research`, `memory`, or `hybrid`) and executes the exact required node sequence in LangGraph.
- **Formula:**
  $$\text{Routing Accuracy} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{actual\_trace}_i == \text{expected\_trace}_i \land \text{actual\_intent}_i == \text{expected\_intent}_i)}{N} \times 100\%$$
- **Target:** 100% (deterministic planner routing).

### 2. Retrieval Recall @ Top-5 (%)
- **Definition:** For research and hybrid queries, measures whether all necessary ground-truth evidence chunk IDs are successfully retrieved in ChromaDB's top-5 vector results.
- **Formula:**
  $$\text{Retrieval Recall} = \frac{\sum_{j=1}^{M} \mathbb{I}(\text{expected\_chunks}_j \subseteq \text{retrieved\_chunks}_j)}{M} \times 100\%$$
  where $M$ is the number of queries requiring document retrieval.

### 3. Citation Validity Rate (%)
- **Definition:** Deterministically verifies that all bracketed citations `[chunk_id]` appearing in the generated answer correspond to valid chunks actually present in the retrieved context, that no hallucinated IDs are cited, and that at least one citation is provided.
- **Validation Rules:**
  - Every citation must match `[chunk_id]` format.
  - Every cited `chunk_id` must belong to the retrieved set: $\text{cited\_ids} \subseteq \text{retrieved\_ids}$.
  - Invalid citation count must be 0: $|\text{invalid\_ids}| == 0$.
  - At least one citation must be present for research answers: $|\text{cited\_ids}| > 0$.

### 4. Factual Grounding Accuracy (%)
- **Definition:** Evaluates whether key ground-truth facts (e.g. specific numbers, dates, policy terms, or SQLite memory entries) are accurately reflected in the generated output without hallucination.
- **Formula:**
  $$\text{Fact Score}_i = \frac{|\{f \in \text{expected\_facts}_i \mid f \in \text{answer}_i\}|}{|\text{expected\_facts}_i|}$$
  $$\text{Case Passed} \iff \text{Fact Score}_i \ge 0.75$$

---

## 4. Demonstrated Measured Improvement: Temporal Supersession

### The Problem
In production enterprise search, newer operational policies frequently contradict older documentation. Standard (naive) RAG implementations retrieve relevant chunks across different publication dates and concatenate them into the prompt. When documents conflict, naive models frequently:
1. Hallucinate an arbitrary compromise between conflicting figures.
2. Quote the older document because of vector similarity rank.
3. Fail to provide chunk citations, leaving users unable to trace the source of truth.

### Experimental Design (A/B Test)

We conducted an empirical A/B experiment comparing:
- **Condition A (Baseline RAG):** Standard RAG prompt concatenating top-5 retrieved chunks with a generic instruction (`"Answer the question based only on the context below:"`), without date metadata emphasis or conflict-resolution rules.
- **Condition B (Asterline Temporal RAG):** The production Asterline prompt format incorporating document publication dates, version numbers, explicit conflict supersession rules (*"If documents conflict, newer documents with explicit supersession statements override older documents"*), and mandatory inline citations `[chunk_id] (Title)`.

Both conditions were evaluated on the same temporal conflict cases:
1. **BENCH-02:** Cold recovery storage retention (`retention:0`, dated 2026-01-15, stating 20 days supersedes `security:1`, dated 2025-03-15, stating 60 days).
2. **BENCH-03:** AQL query timeout limits (`release-23:0` & `query:0`, stating 50s in v2.3, superseding `release-20:2`, stating 25s in v2.0).

### Actual Experimental Results

| Metric | Condition A (Baseline RAG) | Condition B (Asterline Temporal RAG) | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Citation Grounding Rate** | **0.0%** (0/2) | **100.0%** (2/2) | **+100.0%** |
| **Temporal Conflict Resolution** | **50.0%** (1/2) | **100.0%** (2/2) | **+50.0%** |
| Average Valid Citations per Answer | 0.0 | 3.0 | +3.0 citations |

#### Qualitative Evidence Comparison (BENCH-02):
- **Baseline Answer:**
  > *"Raw run records are held in cold recovery storage for 60 days after searchable retention ends according to the security overview, but the data retention policy states 20 days."*
  > *(Result: Zero citations; leaves user confused with conflicting numbers; does not identify which policy governs).*
- **Asterline Temporal RAG Answer:**
  > *"Raw run records are held in cold recovery storage for **20 days** after their plan-specific searchable retention period ends. [retention:0] (Data retention policy)*  
  > ***Conflict Resolution:***  
  > *The current Data Retention Policy (dated 2026-01-15) explicitly states that the 20-day cold-storage period supersedes the earlier 60-day period described in the older Security Overview (dated 2025-03-15). [security:1] (Security overview)"*  
  > *(Result: Explicitly resolves the conflict, cites both sources, and guides the user correctly).*

---

## 5. Evaluation Results Summary

Results captured from the automated run (`eval/results.json`):

```text
===========================================================================
FINAL EVALUATION METRICS SUMMARY
===========================================================================
Total Cases Evaluated:       15
Execution Duration:          65.31 seconds
Agent Routing Accuracy:      100.0% (15/15)
Retrieval Recall (Top-5):    100.0% (11/11)
Citation Validity Rate:      100.0% (11/11)
Factual Grounding Accuracy:  100.0% (15/15)

MEASURED IMPROVEMENT COMPARISON:
  - Citation Grounding Rate: Baseline 0.0%  -->  Asterline 100.0%  (+100.0%)
  - Conflict Resolution:     Baseline 50.0%  -->  Asterline 100.0%  (+50.0%)
===========================================================================
```

---

## 6. Failure Analysis & Observed Edge Cases

During iterative evaluation development, three specific failure modes and edge cases were identified and addressed:

1. **Test Assertion Alignment with Corpus Ground Truth:**
   - *Observation:* Early draft benchmarks contained external assumptions (e.g. expecting 15-second alert intervals or arbitrary stipend figures).
   - *Finding:* In closed-domain RAG systems, evaluation assertions must be strictly derived from the verified knowledge base (`corpus.jsonl`). For instance, `alerts:0` specifies a 4-minute evaluation interval, and `handbook:1` specifies a $175 on-call stipend superseding $140. Correcting the benchmark assertions to match true corpus facts resulted in 100% grounding accuracy.

2. **Negative Recall Semantic Phrasing Sensitivity:**
   - *Observation:* In user isolation tests (e.g. `BENCH-13`), User Beta queries memories stored only by User Alpha. The memory agent responds: `"I do not have a recorded memory for your preferred export format."`
   - *Finding:* Requiring exact multi-word substrings (such as `"currently have no"`) causes false evaluation failures when the LLM produces semantically equivalent negative acknowledgments. Fact assertions for negative recall must target canonical semantic roots (`"not have"`, `"memory"`).

3. **Memory Precedence Invariance:**
   - *Observation:* In `BENCH-15`, an adversarial user stored a personal memory claiming an `"assumed SLA: 100.0%"`.
   - *Finding:* Without strict precedence instructions, hybrid agents may prioritize user-provided assumptions over official documentation. Asterline's architecture enforces a critical invariant: *personal memory represents user preferences only and never overrides official documentation*. The agent correctly cited `sla:0` (99.92%) and rejected the 100% assumption.

---

## 7. Limitations

1. **Closed Corpus Scope:** The benchmark dataset is tailored to the 75 chunks of the Asterline knowledge base. Performance characteristics may vary on multi-gigabyte corpora requiring dense hierarchical indexing or hybrid sparse-dense (BM25 + vector) search.
2. **Deterministic String Matching for Facts:** While citation validity is validated with mathematical precision against chunk metadata, factual grounding utilizes semantic keyword matching. For open-ended creative domains, model-based evaluation (e.g. LLM-as-a-judge) may offer greater flexibility, though at the expense of determinism and latency.
3. **API Rate Limits & Latency:** Live evaluations require active LLM calls to Groq (`qwen/qwen3.8-27b`). Average inference latency is ~0.4s to ~1.2s per question, requiring approximately 65 seconds for the full 15-case benchmark and A/B comparison suite.

---

## 8. Reproducibility & Exact Commands

To reproduce the benchmark evaluation and regenerate `eval/results.json`:

```powershell
# Ensure virtual environment is activated
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Run the automated benchmark evaluation suite
.\.venv\Scripts\python.exe eval/eval_runner.py
```

To run individual regression tests:
```powershell
.\.venv\Scripts\python.exe test_stage1.py   # Corpus indexing & ChromaDB retrieval
.\.venv\Scripts\python.exe test_stage2.py   # Grounded answering & citation validation
.\.venv\Scripts\python.exe test_stage3.py   # LangGraph StateGraph agent routing
.\.venv\Scripts\python.exe test_stage4.py   # Persistent SQLite memory & user isolation
```
