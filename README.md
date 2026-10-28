# Asterline Research Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/vector_store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/inference-Groq-green.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)

An enterprise-grade multi-agent research assistant for operational and policy documentation. Built with **LangGraph**, **ChromaDB**, **SQLite**, and **Groq LLMs**, Asterline features deterministic intent routing, rigorous inline citation validation, automatic temporal policy supersession, and persistent user-scoped memory with multi-tenant isolation.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Environment Configuration](#environment-configuration)
- [Running the Streamlit UI](#running-the-streamlit-ui)
- [Verification & Test Suites](#verification--test-suites)
- [Automated Evaluation Suite](#automated-evaluation-suite)
- [Known Limitations](#known-limitations)

---

## Overview

Asterline Research Assistant answers complex operational and technical inquiries grounded strictly in Asterline's 75-chunk / 25-document knowledge base (`corpus.jsonl`). It solves fundamental failure modes of naive RAG architectures:
1. **Hallucination & Ungrounded Claims:** Enforces strict inline citation validation `[chunk_id] (Document Title)`. Any claim lacking authentic evidence is rejected.
2. **Conflicting Documentation & Supersession:** Distinguishes between obsolete runbooks and superseding policies using publication dates and explicit supersession directives (e.g., verifying that 2026 20-day cold retention overrides 2025 60-day retention).
3. **Multi-User State Management:** Implements persistent SQLite memory (`data/memory.db`) that survives application restarts while strictly isolating user preferences and context across user IDs.
4. **Precedence Invariance:** Strictly guarantees that user personal memories represent context/preferences only and can **never** override official company documentation or SLAs.

---

## Key Features

- **Multi-Agent StateGraph:** Deterministic LangGraph workflow comprising `PlannerNode`, `MemoryManagerNode`, and `ResearcherNode`.
- **Zero-Hallucination Citation Engine:** Deterministic validator verifying that 100% of bracketed citations map to authentic retrieved chunks; flags and blocks hallucinated chunk identifiers.
- **Temporal Supersession Grounding:** Incorporates document publication dates, version numbers, and explicit conflict resolution directives into evidence prompts.
- **Persistent SQLite User Memory:** Multi-tenant isolated storage supporting `store`, `recall`, `update`, `forget`, and `clear` operations.
- **Multi-Provider LLM Fallback:** Fast inference via Groq (`qwen/qwen3.8-27b` / `llama-3.3-70b-versatile`), with transparent fallback to Google Gemini (`gemini-3-flash-preview`).
- **Futuristic Streamlit Chat UI (`app.py`):** Near-black glassmorphic interface with interactive user switching, LangGraph node execution trace pills, expandable evidence sources, and live memory management.
- **Comprehensive Benchmark Suite:** 15 curated benchmark evaluation cases with an empirical A/B experiment demonstrating **+100.0%** citation grounding and **+50.0%** conflict resolution gains over baseline RAG.

---

## System Architecture

```text
                                +-------------------+
                                |    User Request   |
                                +---------+---------+
                                          |
                                          v
                                +-------------------+
                                |   Planner Node    |
                                |  (Intent Routing) |
                                +----+----+----+----+
                                     |    |    |
           +-------------------------+    |    +-------------------------+
           | (intent = 'memory')          | (intent = 'hybrid')          | (intent = 'research')
           v                              v                              |
+----------------------+        +----------------------+                 |
| Memory Manager Node  |        | Memory Manager Node  |                 |
| (SQLite CRUD)        |        | (Loads User Context) |                 |
+----------+-----------+        +----------+-----------+                 |
           |                               |                             |
           |                               +------------------------+    |
           |                                                        |    |
           |                                                        v    v
           |                                            +----------------------+
           |                                            |   Researcher Node    |
           |                                            | (ChromaDB + Groq LLM)|
           |                                            +----------+-----------+
           |                                                       |
           v                                                       v
+------------------------------------------------------------------------------+
|                         Final State & Streamlit UI                           |
|       (Answer, Execution Trace, Verified Citations, Active Memories)         |
+------------------------------------------------------------------------------+
```

---

## Project Structure

```text
asterline-research-assistant/
├── app.py                     # Streamlit interactive chat UI
├── corpus.jsonl               # Official Asterline knowledge base (75 chunks)
├── requirements.txt           # Python dependencies
├── .env.example               # Template for private API credentials
├── DESIGN.md                  # Detailed architectural design & Mermaid diagrams
├── AI_DISCLOSURE.md           # Transparent AI coding assistant disclosure
├── data/
│   ├── chroma_db/             # Persistent ChromaDB vector store
│   └── memory.db              # Persistent SQLite memory database
├── src/
│   ├── corpus.py              # Schema validation & corpus loader
│   ├── indexer.py             # ChromaDB indexing & semantic vector retrieval
│   ├── researcher.py          # Grounded synthesis, citation validation, LLM client
│   ├── memory.py              # SQLiteMemoryStore with user isolation
│   └── agents/
│       ├── state.py           # TypedDict AgentState definition
│       ├── planner.py         # Intent classification & routing logic
│       ├── memory_agent.py    # Memory manager node implementation
│       ├── researcher_agent.py# Researcher node implementation
│       └── graph.py           # Compiled LangGraph StateGraph workflow
├── eval/
│   ├── benchmark_dataset.json # 15 curated benchmark evaluation questions
│   ├── eval_runner.py         # Automated evaluation & A/B experiment runner
│   ├── results.json           # Serialized benchmark metrics & test results
│   └── README.md              # Evaluation methodology, metrics & failure analysis
└── test_stage1.py ... test_stage4.py # Stage verification test suites
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 to 3.13
- Git

### 1. Clone & Create Virtual Environment
```powershell
# Clone the repository
git clone <repo-url>
cd asterline-research-assistant

# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # On Windows (PowerShell)
# source .venv/bin/activate    # On Linux / macOS
```

### 2. Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in your private API credentials:

```powershell
cp .env.example .env
```

### Required Configuration (`.env`)
> [!IMPORTANT]
> Use placeholders only when sharing configuration files. **Never commit `.env` or expose private API keys.**

```ini
# Primary LLM Provider: Groq (Ultra-fast inference)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile  # Default; automatically falls back to qwen/qwen3.8-27b if unavailable

# Fallback LLM Provider: Google Gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview

# LangSmith Tracing & Observability (Optional)
# Modern LangSmith SDK variables:
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=asterline-research-assistant
# Legacy aliases also supported by SDK: LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
```

---

## Running the Streamlit UI

Launch the interactive chat interface:

```powershell
streamlit run app.py
```

The application will be accessible at `http://localhost:8501`.

### UI Capabilities:
1. **User Persona Selector:** Switch between `alex_sre`, `sarah_pm`, `jordan_lead`, or enter a custom user ID to test multi-tenant memory isolation.
2. **LangGraph Execution Trace:** Displays real-time node progression pills (e.g. `planner` &rarr; `memory_manager` &rarr; `researcher`).
3. **Expandable Evidence Sources:** View full retrieved chunk texts, publication dates, and similarity scores.
4. **Memory Sidebar Panel:** Live table of active memories stored in SQLite for the selected user, with quick-store and memory clearing controls.

---

## Verification & Test Suites

The project features comprehensive automated test suites for each implementation stage:

### 1. Stage 1: Corpus Loading & Vector Indexing
Validates schema integrity of all 75 chunks in `corpus.jsonl`, idempotent ChromaDB indexing, and vector similarity retrieval:
```powershell
python test_stage1.py
```
*Result: 8/8 tests passed.*

### 2. Stage 2: Grounded Answering & Citation Validation
Validates deterministic citation validator, prompt formatting, and live Groq LLM generation across factual, conflict, and negative queries:
```powershell
python test_stage2.py
```
*Result: Part A (Deterministic) & Part B (Live Groq) passed.*

### 3. Stage 3: Multi-Agent StateGraph Workflow
Tests LangGraph topology, planner intent classification (`research`, `memory`, `hybrid`), state transitions, and memory-documentation precedence invariance:
```powershell
python test_stage3.py
```
*Result: 7/7 tests passed.*

### 4. Stage 4: Persistent SQLite User Memory
Verifies SQLite CRUD operations, process restart persistence, case-insensitive key recall, and strict user isolation:
```powershell
python test_stage4.py
```
*Result: 7/7 tests passed.*

### 5. Stage 5: Streamlit UI Smoke Test
Verifies UI syntax, component imports, and headless rendering:
```powershell
python -m py_compile app.py
streamlit run app.py --server.headless true
```
*Result: Clean compilation; HTTP 200 on port 8501.*

---

## Automated Evaluation Suite

The evaluation suite runs 15 curated benchmark questions covering factual retrieval, temporal supersession, version conflicts, negative grounding, memory CRUD, user isolation, and precedence invariance. It also executes an empirical A/B test against a baseline RAG condition.

### Run Benchmark Evaluation
```powershell
python eval/eval_runner.py
```

### Benchmark Results (`eval/results.json`)

| Metric | Target | Actual Score |
| :--- | :---: | :---: |
| **Agent Routing Accuracy** | 100.0% | **100.0%** (15/15) |
| **Retrieval Recall @ Top-5** | $\ge 90.0\%$ | **100.0%** (11/11) |
| **Citation Validity Rate** | 100.0% | **100.0%** (11/11) |
| **Factual Grounding Accuracy** | $\ge 80.0\%$ | **100.0%** (15/15) |

### Demonstrated Measured Improvement (A/B Test)

| Metric | Baseline RAG | Asterline Temporal RAG | Absolute Gain |
| :--- | :---: | :---: | :---: |
| **Citation Grounding Rate** | 0.0% (0/2) | **100.0%** (2/2) | **+100.0%** |
| **Temporal Conflict Resolution** | 50.0% (1/2) | **100.0%** (2/2) | **+50.0%** |
| **Average Valid Citations / Answer** | 0.0 | **3.0** | **+3.0 citations** |

For complete evaluation methodology, failure analysis, and benchmark details, see [`eval/README.md`](file:///d:/asterline-research-assistant/eval/README.md).

---

## Known Limitations

1. **Closed Domain Corpus:** The assistant is grounded specifically on the 75 chunks of `corpus.jsonl`. Queries completely outside this domain will be rejected or report that no documentation exists.
2. **Single-Node SQLite Storage:** Memory persistence is managed via a local SQLite database (`data/memory.db`). For horizontally scaled multi-container deployments, a centralized relational database (e.g., PostgreSQL) would be required.
3. **LLM API Latency:** Answer generation depends on external API calls to Groq Cloud. Average response latency is 0.4s to 1.2s per query.
4. **LangSmith Tracing:** LangSmith integration is supported via environment variables (`LANGSMITH_TRACING=true` or legacy `LANGCHAIN_TRACING_V2=true`), but active web trace recording requires a valid LangSmith API key.
