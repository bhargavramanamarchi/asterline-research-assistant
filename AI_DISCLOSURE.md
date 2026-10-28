# AI Use Disclosure & Verification Statement

This document transparently outlines the use of AI coding assistants during the development, debugging, evaluation, and documentation of the **Asterline Research Assistant** project, alongside the human oversight and empirical verification applied at each stage.

---

## 1. Summary of AI Assistance

Development was conducted through pair programming with an advanced AI coding assistant (**Antigravity IDE**, powered by Google DeepMind agentic models). AI assistance was utilized across five major project areas:

1. **Architecture & Multi-Agent State Machine:**
   - Designed and refined the **LangGraph StateGraph** topology (`PlannerNode`, `MemoryManagerNode`, `ResearcherNode`).
   - Implemented state schema transitions and conditional routing based on intent classification.
2. **Retrieval, Prompt Engineering & Citation Enforcement:**
   - Structured the evidence prompt format to inject document publication dates, versions, and categories.
   - Implemented regex-based citation extraction and set containment validation (`validate_citations`).
   - Formulated the multi-provider LLM fallback hierarchy (Groq primary, Gemini secondary).
3. **Persistent Memory & Isolation:**
   - Drafted the SQLite database schema (`data/memory.db`) with composite primary keys `(user_id, key)`.
   - Built parameterized CRUD queries to ensure multi-tenant user isolation.
   - Resolved Windows SQLite file locking issues by wrapping connections in a context manager with explicit closure.
4. **Streamlit Chat UI:**
   - Designed the dark glassmorphic UI layout with custom CSS styling inspired by modern AI SaaS design patterns.
   - Built execution trace rendering, citation expanders, and sidebar memory controls.
5. **Evaluation Suite & Documentation:**
   - Curated the 15-case benchmark dataset (`eval/benchmark_dataset.json`).
   - Implemented the automated evaluation runner and baseline vs. temporal RAG A/B comparison experiment (`eval/eval_runner.py`).
   - Generated system architecture diagrams (Mermaid), `README.md`, `DESIGN.md`, and this disclosure.

---

## 2. Human Oversight, Code Review & Verification

Every component, function, and test suite generated or modified with AI assistance underwent strict verification:

### 1. Corpus & Data Integrity
- **Verification:** The original knowledge base `corpus.jsonl` was cryptographically verified before and after each task using SHA-256 hashing.
- **Result:** The SHA-256 hash remained strictly `f63fc05cf43204485c98bd8fb6d89860a32c16a7e906bc90975e7c3954fe9fb1` with zero data modification or hallucinated chunks.

### 2. Security & Secret Protection
- **Verification:** All database queries were reviewed to verify parameterized execution (`?` placeholders), preventing SQL injection vulnerabilities.
- **Secret Hygiene:** Environment variables were audited to verify that `.env` is ignored by `.gitignore` and that no private API keys are ever printed in console logs, exception handlers, or test output.

### 3. Empirical Test Suite Execution
AI-assisted implementations were verified against rigorous, stage-by-stage automated test suites:
- **`test_stage1.py`:** Verified 75-chunk schema compliance, ChromaDB idempotent indexing, and vector similarity retrieval (8/8 tests passed).
- **`test_stage2.py`:** Tested deterministic citation validation and live Groq LLM grounded answering (Part A and Part B passed).
- **`test_stage3.py`:** Verified LangGraph routing for `research`, `memory`, and `hybrid` intents, and confirmed precedence invariance (7/7 tests passed).
- **`test_stage4.py`:** Verified SQLite CRUD operations, process restart survival, and multi-tenant user isolation (7/7 tests passed).
- **`eval/eval_runner.py`:** Ran all 15 benchmark questions against the live system, confirming 100% routing accuracy, 100% retrieval recall, 100% citation validity, and 100% factual accuracy.

### 4. Empirical A/B Improvement Demonstration
The claimed improvement in temporal conflict resolution was not assumed; it was empirically proven via an automated A/B test comparing baseline RAG with Asterline's temporal-grounded prompt on the same queries:
- **Baseline RAG:** 0.0% citation grounding rate, 50.0% conflict resolution.
- **Asterline Temporal RAG:** 100.0% citation grounding rate, 100.0% conflict resolution (+100.0% and +50.0% absolute gains).

---

## 3. Human Authorship & Responsibility

All final architectural decisions, prompt constraints, metric definitions, and verification assertions were reviewed, tested, and validated for strict adherence to the Asterline assignment specification.
