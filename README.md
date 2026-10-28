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
  - [Component Breakdown](#component-breakdown)
  - [Example Information Flow](#example-information-flow)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
  - [PowerShell Script Execution Troubleshooting](#powershell-script-execution-troubleshooting)
- [Environment Configuration](#environment-configuration)
  - [Provider Requirements & Fallback Strategy](#provider-requirements--fallback-strategy)
- [Running the Streamlit UI](#running-the-streamlit-ui)
  - [Step-by-Step Usage Walkthrough](#step-by-step-usage-walkthrough)
- [Troubleshooting](#troubleshooting)
- [Verification & Test Suites](#verification--test-suites)
- [Automated Evaluation Suite](#automated-evaluation-suite)
- [Known Limitations](#known-limitations)

---

## Overview

Asterline Research Assistant answers complex operational and technical inquiries grounded strictly in Asterline's 75-chunk / 25-document knowledge base (`corpus.jsonl`). It addresses common failure modes in naive RAG architectures:

1. **Grounded Answering & Citation Integrity:** Enforces inline citation formatting `[chunk_id] (Document Title)` and validates that cited chunks exist within the retrieved context set. Prompt constraints instruct the model to answer based only on retrieved evidence and decline claims lacking documentary support.
2. **Conflicting Documentation & Supersession:** Distinguishes between obsolete runbooks and superseding policies using publication dates and explicit supersession directives (for example, verifying that the 2026 20-day cold retention policy supersedes the earlier 2025 60-day policy).
3. **Multi-User State Management:** Implements persistent SQLite memory (`data/memory.db`) that survives application restarts while isolating user preferences and context across user IDs via parameterized queries.
4. **Precedence Invariance:** System prompts and multi-agent routing enforce an architectural hierarchy where user personal memories provide context/preferences only, while official company documentation and SLAs strictly govern research synthesis.

---

## Key Features

- **Multi-Agent StateGraph:** Deterministic LangGraph workflow comprising `PlannerNode`, `MemoryManagerNode`, and `ResearcherNode`.
- **Deterministic Citation Validation:** Verifies that bracketed citations map to authentic retrieved chunks, detecting and flagging unretrieved or hallucinated chunk identifiers.
- **Temporal Supersession Grounding:** Incorporates document publication dates, version numbers, and explicit conflict resolution directives into evidence prompts.
- **Persistent SQLite User Memory:** Multi-tenant isolated storage supporting `store`, `recall`, `update`, `forget`, and `clear` operations.
- **Multi-Provider LLM Fallback:** Fast primary inference via Groq (`llama-3.3-70b-versatile`, with automatic fallback to `qwen/qwen3.8-27b`), with secondary fallback to Google Gemini (`gemini-3-flash-preview`).
- **Futuristic Streamlit Chat UI (`app.py`):** Near-black glassmorphic interface with interactive user switching, LangGraph node execution trace pills, expandable evidence sources, and live memory management.
- **Comprehensive Benchmark Suite:** 15 curated benchmark evaluation cases with an empirical A/B experiment demonstrating **+100.0%** citation grounding and **+50.0%** conflict resolution gains over a baseline RAG prompt under test conditions.

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

### Component Breakdown

In plain English, the system divides responsibilities across specialized components:

- **Planner (`src/agents/planner.py`):** The orchestrator's front door. It analyzes incoming user prompts and decides the execution path: pure research against documentation (`research`), personal memory management (`memory`), or a combined question (`hybrid`).
- **Memory Manager (`src/agents/memory_agent.py`):** Handles saving, recalling, updating, and clearing user-specific memories in SQLite. It is **conditional**: pure research questions bypass this node entirely.
- **Researcher (`src/agents/researcher_agent.py` & `src/researcher.py`):** The evidence synthesis engine. It converts the query into a vector search, pulls the top-5 chunks from ChromaDB, constructs a dated evidence prompt, calls the LLM, and verifies citation integrity.
- **LangGraph (`src/agents/graph.py`):** The coordination framework. It defines the state machine, links nodes with conditional edges, and tracks the execution trace.
- **ChromaDB (`data/chroma_db/`):** A persistent vector database storing dense mathematical embeddings (`all-MiniLM-L6-v2`) for all 75 chunks of the knowledge base.
- **SQLite (`data/memory.db`):** A lightweight local relational database providing persistent, user-scoped memory storage that survives application and process restarts.
- **LangSmith:** An observability platform that records execution traces, node latency, and LLM inputs/outputs when tracing is enabled in `.env`.

### Example Information Flow

Consider a user asking: **"What is the contractual monthly uptime SLA for Enterprise customers?"**

1. **Submission:** The user sends the prompt via the Streamlit UI.
2. **Planner Evaluation:** The Planner scans the query, detects an inquiry about company terms without personal memory commands, and sets `intent = 'research'`.
3. **Conditional Routing:** Because the intent is `research`, the graph routes directly to the **Researcher**. The Memory Manager is bypassed.
4. **Evidence Retrieval:** The Researcher queries ChromaDB and retrieves chunk `sla:0` (*Service and support terms*, published 2026-03-10, stating 99.92% SLA and credit tiers).
5. **LLM Synthesis & Citation Check:** The LLM generates the grounded response. The citation validator scans the text and confirms that `[sla:0]` matches an authentically retrieved chunk.
6. **Delivery:** The user receives the answer with citations (`99.92% [sla:0] (Service and support terms)`) and an execution trace pill: `planner` &rarr; `researcher`.

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

Before starting, ensure your development environment has:
- **Python:** Version **3.10 to 3.13** installed (verified with Python 3.13 on Windows). Check with `python --version`.
- **Git:** Installed and available on your PATH. Check with `git --version`.
- **Internet Access:** Required for downloading dependencies, embedding models, and communicating with the Groq/Gemini APIs.
- **LLM API Key:** A free API key from [Groq Cloud Console](https://console.groq.com/) (recommended primary provider).

### Step-by-Step Setup Instructions

Open your terminal (**Windows PowerShell** recommended on Windows) and run the following numbered commands:

#### 1. Clone the repository
Downloads the complete project repository to your local machine:
```powershell
git clone https://github.com/bhargavramanamarchi/asterline-research-assistant.git
```

#### 2. Navigate into the project folder
Changes your active working directory:
```powershell
cd asterline-research-assistant
```

#### 3. Create an isolated virtual environment
Creates a dedicated local Python environment in `.venv/` to prevent conflicts with other system packages:
```powershell
python -m venv .venv
```

#### 4. Activate the virtual environment
Activates the virtual environment so subsequent `pip` and `python` commands run within this isolated container:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(On macOS / Linux, the equivalent command is: `source .venv/bin/activate`)*

#### 5. Upgrade pip and install dependencies
Installs the exact packages required across all 5 stages from `requirements.txt`:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### PowerShell Script Execution Troubleshooting

If Windows PowerShell displays an error such as:
> `File ...\Activate.ps1 cannot be loaded because running scripts is disabled on this system. (PSSecurityException)`

This is Windows' default script execution policy for PowerShell. You can resolve it safely using either of two methods **without permanently weakening system security**:

- **Option A (Recommended — Current Window Only):** Temporarily bypass the execution policy for your current PowerShell terminal session only:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\.venv\Scripts\Activate.ps1
  ```
- **Option B (Direct Invocation):** You do not need to activate the environment at all. You can run any command directly using the virtual environment's executable:
  ```powershell
  .\.venv\Scripts\python.exe -m streamlit run app.py
  .\.venv\Scripts\python.exe test_stage1.py
  ```

---

## Environment Configuration

The application reads configuration parameters from a root `.env` file. A template with placeholders is provided in `.env.example`.

### Setting Up Your Environment File

On Windows PowerShell, copy the example template:
```powershell
Copy-Item .env.example .env
```
*(On Command Prompt: `copy .env.example .env` | On macOS/Linux: `cp .env.example .env`)*

Open `.env` in your text editor (VS Code, Notepad, etc.) and add your API keys:

```ini
# Primary LLM Provider: Groq (Ultra-fast inference)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile  # Default; automatically falls back to qwen/qwen3.8-27b if unavailable

# Fallback LLM Provider: Google Gemini (Optional)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3-flash-preview

# LangSmith Tracing & Observability (Optional)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGSMITH_PROJECT=asterline-research-assistant
# Legacy SDK aliases also supported: LANGCHAIN_TRACING_V2, LANGCHAIN_API_KEY, LANGCHAIN_PROJECT
```

> [!CAUTION]
> **Security Notice:** `.env` contains private secrets and is excluded by `.gitignore`. **Never commit `.env` to version control or share its contents in public forums or pull requests.**

### Provider Requirements & Fallback Strategy

- **`GROQ_API_KEY` (Required for live generation):** Groq is the primary live inference provider due to its sub-second token generation speeds. If your account lacks access to `llama-3.3-70b-versatile`, the backend in `src/researcher.py` automatically falls back to `qwen/qwen3.8-27b`.
- **`GEMINI_API_KEY` (Optional fallback):** If Groq encounters rate limits or service unavailability, the system falls back to Google Gemini (`gemini-3-flash-preview`).
- **Offline / Deterministic Testing:** If no API keys are provided, all deterministic retrieval, vector search, citation validation, LangGraph routing, and SQLite memory tests still pass offline.
- **LangSmith Tracing (Optional):** Setting `LANGSMITH_TRACING=true` and providing a `LANGSMITH_API_KEY` is optional for running the application locally, but is required if you want to record and inspect hosted trace runs in your LangSmith dashboard.

---

## Running the Streamlit UI

Launch the interactive chat interface:

```powershell
streamlit run app.py
```
*(Or directly: `.\.venv\Scripts\python.exe -m streamlit run app.py`)*

Open your browser and navigate to the local address displayed in the console:
```text
http://localhost:8501
```

### Step-by-Step Usage Walkthrough

1. **Ask a research question:** In the chat input, type:
   > `What is the contractual monthly uptime SLA for Enterprise customers, and what are the service credit tiers?`
   
   Observe that the assistant returns a grounded answer citing `[sla:0] (Service and support terms)` and displays an execution trace pill: `planner` &rarr; `researcher`.

2. **Inspect evidence sources:** Click the **Retrieved Evidence Chunks** expander beneath the assistant's answer to view the source chunk IDs, document titles, publication dates, and relevance scores.

3. **Store a personal user preference:** In the chat input, type:
   > `Remember that my preferred export format is Parquet`
   
   Notice that the execution trace shows `planner` &rarr; `memory_manager`. The memory is saved into `data/memory.db` and immediately appears in the **Active User Memories** sidebar table.

4. **Recall a personal preference:** In the chat input, type:
   > `What is my preferred export format?`
   
   The assistant queries SQLite and responds: *"Your preferred export format is Parquet."*

5. **Test a hybrid query:** In the chat input, type:
   > `Remember that my tier is Enterprise, and tell me our monthly SLA uptime.`
   
   Notice the execution trace shows `planner` &rarr; `memory_manager` &rarr; `researcher`. The assistant stores your tier in SQLite and researches the documentation with your active tier in mind.

6. **Verify user isolation:** In the left sidebar, change the **Active User ID** from `alex_sre` to `sarah_pm`. In the chat input, ask:
   > `What is my preferred export format?`
   
   The assistant replies that it has no recorded memory for `sarah_pm`. Switch back to `alex_sre`, and your `Parquet` preference is restored, demonstrating strict multi-tenant isolation.

7. **Stop the application:** When finished, click inside your terminal window and press `Ctrl + C` to stop the Streamlit web server.

---

## Troubleshooting

| Problem | Diagnostic | Safe Resolution |
| :--- | :--- | :--- |
| **Python command not found** | Terminal returns: `'python' is not recognized as an internal or external command`. | Install Python 3.10+ from [python.org](https://www.python.org/). During installation on Windows, ensure the box **"Add python.exe to PATH"** is checked. |
| **PowerShell script execution disabled** | Error: `PSSecurityException: Execution of scripts is disabled on this system`. | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the current terminal, or invoke the environment directly using `.\.venv\Scripts\python.exe`. |
| **Missing dependencies** | Error: `ModuleNotFoundError: No module named 'langgraph'`. | Ensure your virtual environment is active, then run `pip install -r requirements.txt`. |
| **Missing or invalid API key** | Application or tests warn: `Active generation provider detected: NONE` or HTTP `401 Unauthorized`. | Verify that `.env` exists in the project root, contains `GROQ_API_KEY=...` without extra quotes or spaces, and that you have saved the file. |
| **Model access error (Groq 404)** | Error: `NotFoundError: The model llama-3.3-70b-versatile does not exist or you do not have access to it.` | The researcher automatically falls back to `qwen/qwen3.8-27b`. You can also explicitly set `GROQ_MODEL=qwen/qwen3.8-27b` in your `.env`. |
| **Port 8501 already in use** | Streamlit warns: `Port 8501 is already in use.` | Another instance is running. Launch on a different port: `streamlit run app.py --server.port 8502`. |
| **No LangSmith traces recorded** | Runs execute but do not appear in the LangSmith dashboard. | Confirm that `LANGSMITH_TRACING=true` and a valid `LANGSMITH_API_KEY` are present in `.env`. Restart the app so environment variables are reloaded. |
| **ChromaDB uninitialized** | Semantic retrieval returns 0 hits or collection missing. | Run `python test_stage1.py`. This idempotently indexes the 75 chunks into `data/chroma_db/` without duplicating records. |

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

> [!NOTE]
> Benchmark results reflect performance on the 15 curated evaluation questions in [`eval/benchmark_dataset.json`](eval/benchmark_dataset.json). The A/B measured improvement was evaluated on two specific temporal-conflict cases (`BENCH-02` and `BENCH-03`), demonstrating an observed empirical gain in citation grounding and conflict resolution under those test conditions rather than an absolute or universal guarantee across all possible unseen documents.

For complete evaluation methodology, failure analysis, and benchmark details, see [eval/README.md](eval/README.md). For detailed architecture specifications, see [DESIGN.md](DESIGN.md). For AI usage disclosure, see [AI_DISCLOSURE.md](AI_DISCLOSURE.md).

---

## Known Limitations

1. **Closed Domain Corpus:** The assistant is grounded specifically on the 75 chunks of `corpus.jsonl`. Queries completely outside this domain will be rejected or report that no documentation exists.
2. **Single-Node SQLite Storage:** Memory persistence is managed via a local SQLite database (`data/memory.db`). For horizontally scaled multi-container deployments, a centralized relational database (e.g., PostgreSQL) would be required.
3. **LLM API Latency:** Answer generation depends on external API calls to Groq Cloud. In benchmark testing, observed response latencies ranged between ~0.4s and 1.2s per generation call depending on network traffic and prompt length.
4. **LangSmith Tracing:** LangSmith integration is supported via environment variables (`LANGSMITH_TRACING=true` or legacy `LANGCHAIN_TRACING_V2=true`), but active web trace recording requires a valid LangSmith API key.
