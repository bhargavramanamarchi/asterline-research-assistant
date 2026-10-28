# Asterline Research Assistant

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![ChromaDB](https://img.shields.io/badge/vector_store-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/inference-Groq-green.svg)](https://groq.com/)
[![Streamlit](https://img.shields.io/badge/ui-Streamlit-red.svg)](https://streamlit.io/)

An enterprise-grade multi-agent research assistant for operational and policy documentation. Built with **LangGraph**, **ChromaDB**, **SQLite**, and **Groq LLMs**, Asterline features deterministic intent routing, rigorous inline citation validation, automatic temporal policy supersession, and persistent user-scoped memory with multi-tenant isolation.

---

## Live Demo

Try Asterline Research Assistant: [Open Live Application](https://asterline-research-assistant-pyjqd6vnhyxkpz5qsry7sp.streamlit.app/)

Repository: [GitHub Source Code](https://github.com/bhargavramanamarchi/asterline-research-assistant)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
  - [Component Breakdown](#component-breakdown)
  - [Beginner-Friendly Architecture Guide](#beginner-friendly-architecture-guide)
  - [Example Information Flow](#example-information-flow)
- [Project Structure](#project-structure)
  - [Directory Tree](#directory-tree)
  - [File & Directory Responsibilities](#file--directory-responsibilities)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Setup Instructions](#step-by-step-setup-instructions)
  - [PowerShell Script Execution Troubleshooting](#powershell-script-execution-troubleshooting)
- [Environment Configuration](#environment-configuration)
  - [Provider Requirements & Fallback Strategy](#provider-requirements--fallback-strategy)
  - [Streamlit Cloud Secrets Compatibility](#streamlit-cloud-secrets-compatibility)
- [Running the Streamlit UI](#running-the-streamlit-ui)
  - [Step-by-Step Usage Walkthrough](#step-by-step-usage-walkthrough)
- [Troubleshooting](#troubleshooting)
- [Verification & Test Suites](#verification--test-suites)
- [Automated Evaluation Suite](#automated-evaluation-suite)
- [Documentation Index](#documentation-index)
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

### Beginner-Friendly Architecture Guide

To understand how Asterline Research Assistant works from the ground up, consider these seven core mechanisms:

1. **How the Agents Work Together (Planner, Memory Manager, Researcher):**
   - The application is orchestrated with **LangGraph** as a finite state machine. Every query starts at the **Planner Node**.
   - The Planner evaluates the query's intent:
     - **Pure Research (`research`):** Inquiries about company policies, runbooks, or SLAs route directly to the **Researcher Node**, completely bypassing memory management to eliminate latency.
     - **Pure Memory (`memory`):** Commands to remember, recall, update, or clear personal notes route to the **Memory Manager Node**, bypassing document retrieval entirely.
     - **Hybrid (`hybrid`):** Context-dependent questions (e.g. *"Remember that my tier is Enterprise, and tell me our monthly SLA uptime"*) route first to the **Memory Manager Node** to store or retrieve user preferences, and then pass that context forward into the **Researcher Node** to condition the search and answer synthesis.

2. **How ChromaDB & Local Embeddings Support Document Retrieval:**
   - Asterline stores its 75-chunk knowledge base in a local, persistent **ChromaDB** vector database (`data/chroma_db/`).
   - Text chunks are converted into dense mathematical vectors using Chroma's built-in `all-MiniLM-L6-v2` ONNX embedding function.
   - This embedding process executes **100% locally on CPU**, requiring zero GPU resources, zero PyTorch installation, and zero external API calls.
   - When a research query is processed, ChromaDB calculates cosine distances between the query embedding and chunk embeddings, retrieving the top-5 most semantically relevant chunks.

3. **How Citations and Grounded Answers Are Produced:**
   - The Researcher formats retrieved chunks into a chronological evidence block containing chunk IDs, document titles, categories, publication dates, and versions.
   - The LLM prompt strictly instructs the model:
     - Cite every factual claim using the explicit syntax: `[chunk_id] (Document Title)`.
     - Decline unannounced or unsupported features honestly (*"The retrieved documentation does not establish..."*).
     - Resolve temporal conflicts by comparing publication dates and version numbers (e.g. verifying that `retention:0` published in 2026 explicitly supersedes `security:1` published in 2025).
   - Before presenting the response, Asterline runs a deterministic regex citation validator (`validate_citations`). Any hallucinated chunk ID not present in the authentically retrieved context is rejected.

4. **How SQLite Stores User-Specific Memory and Protects User Isolation:**
   - User memories (preferences, role, plan tier) are stored in a persistent SQLite database (`data/memory.db`).
   - The database schema enforces multi-tenancy using a composite primary key `(user_id, key)`.
   - All SQL operations use parameterized queries (`WHERE user_id = ? AND key = ?`), preventing SQL injection and guaranteeing that `user_alice` can never read, modify, or leak into `user_bob`'s memories.
   - **Precedence Invariant:** User memories serve as personal context and preferences *only*. Personal memories can **never** override official Asterline documentation, SLAs, or security policies.

5. **Primary LLM Provider (Groq) & Optional Fallback (Gemini):**
   - **Groq** is the primary inference engine, using `llama-3.3-70b-versatile` (with automatic runtime fallback to `qwen/qwen3.8-27b`), achieving ultra-low inference latency (~0.4s to 0.8s).
   - **Google Gemini** (`gemini-3-flash-preview`) is configured as an optional secondary fallback if Groq is unavailable.
   - If `GROQ_API_KEY` is present, the application runs entirely on Groq and does **not** require a Gemini API key.
   - All deterministic tests (vector search, citation validation, LangGraph routing, SQLite memory) pass offline without any API keys.

6. **How LangSmith Tracing Helps Inspect Agent Execution:**
   - When `LANGSMITH_TRACING=true` and a `LANGSMITH_API_KEY` are configured, every execution step, routing decision, retrieved chunk list, prompt template, token count, and latency metric is logged to LangSmith.
   - Publicly inspectable trace evidence links for successful SLA runs and temporal supersession runs are documented in [LANGSMITH_EVIDENCE.md](LANGSMITH_EVIDENCE.md). Tracing remains entirely optional and non-blocking.

7. **How the Application Initializes Its Vector Database on Cold Start:**
   - In cloud container environments (such as Streamlit Community Cloud) or on fresh clones where `data/chroma_db/` is absent or empty, `ensure_indexed()` in `src/indexer.py` automatically checks `collection.count() == 0`.
   - If empty, it loads `corpus.jsonl` and populates ChromaDB via `collection.upsert()` with stable chunk IDs before any search executes.
   - Indexing is strictly idempotent: re-initialization or subsequent queries see `count() > 0` and skip indexing instantly, preventing duplicate records, circular imports, and query latency overhead.

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

### Directory Tree

```text
asterline-research-assistant/
├── app.py                     # Streamlit user interface & interactive chat entry point
├── corpus.jsonl               # Official 75-chunk research knowledge base (preserved unchanged)
├── requirements.txt           # Python package dependencies
├── .env.example               # Template environment configuration with placeholder keys
├── .gitignore                 # Version control exclusion rules (virtual environments, local DBs, .env)
├── README.md                  # Project overview, setup, usage, and benchmark summary
├── DESIGN.md                  # Architectural specifications, trade-offs, and Mermaid diagrams
├── AI_DISCLOSURE.md           # Transparent AI coding tools disclosure and verification statement
├── LANGSMITH_EVIDENCE.md      # Publicly inspectable LangSmith trace run links and analysis
├── data/
│   ├── chroma_db/             # Persistent local ChromaDB vector store
│   └── memory.db              # Persistent SQLite memory database
├── src/
│   ├── corpus.py              # Read-only corpus loading & schema validation
│   ├── indexer.py             # ChromaDB vector indexing, search & cold-start auto-init
│   ├── researcher.py          # Grounded synthesis, citation validation, LLM client
│   ├── memory.py              # SQLiteMemoryStore with multi-tenant user isolation
│   └── agents/
│       ├── state.py           # TypedDict AgentState state schema definition
│       ├── planner.py         # Deterministic regex & intent classification routing
│       ├── memory_agent.py    # Memory manager node implementation (CRUD)
│       ├── researcher_agent.py# Researcher node implementation (retrieval & synthesis)
│       └── graph.py           # Compiled LangGraph StateGraph workflow
├── eval/
│   ├── benchmark_dataset.json # 15 curated benchmark evaluation questions
│   ├── eval_runner.py         # Automated evaluation & empirical A/B experiment runner
│   ├── results.json           # Serialized benchmark metrics, latencies & test scores
│   └── README.md              # Evaluation methodology, metric definitions & failure analysis
├── test_stage1.py             # Stage 1 test: Corpus validation, indexing, and vector search
├── test_stage2.py             # Stage 2 test: Grounded answering, citation check & live Groq
├── test_stage3.py             # Stage 3 test: LangGraph multi-agent routing & state transitions
├── test_stage4.py             # Stage 4 test: SQLite persistent memory & user isolation
└── test_cloud_fixes.py        # Cloud fixes test: Cold-start indexing & Streamlit Secrets
```

### File & Directory Responsibilities

Every file and directory in the project has a clearly defined responsibility verified against the source code:

- **`app.py`:** The primary application entry point and user interface. Implemented with Streamlit, it provides a futuristic dark-glass chat interface, live user session switcher, persistent SQLite memory viewer, LangGraph execution trace pills, expandable evidence drawers, and Streamlit Community Cloud Secrets synchronization.
- **`src/`:** Core backend logic and architecture modules:
  - **`src/agents/`:** LangGraph multi-agent implementation containing agent nodes, state definitions, and graph compilation:
    - **`src/agents/planner.py`:** Analyzes incoming user queries using deterministic regex patterns to classify intents into `research`, `memory`, or `hybrid` workflows.
    - **`src/agents/memory_agent.py`:** Executes SQLite memory operations (`store`, `recall`, `update`, `forget`, `clear`) based on parsed user intent.
    - **`src/agents/researcher_agent.py`:** Manages contextual search query formation, ChromaDB retrieval, prompt formatting, LLM generation, and citation validation.
    - **`src/agents/graph.py`:** Compiles the LangGraph `StateGraph`, defining nodes and conditional edges that coordinate multi-agent execution.
    - **`src/agents/state.py`:** Defines the `AgentState` TypedDict schema passing state (query, intent, user_id, memories, retrieved_chunks, answer, citations, execution_trace) between nodes.
  - **`src/indexer.py`:** Manages the persistent ChromaDB vector database. Configures local ONNX `all-MiniLM-L6-v2` embeddings, provides idempotent indexing via `collection.upsert()`, executes semantic vector search, and provides `ensure_indexed()` for cold-start auto-initialization.
  - **`src/corpus.py`:** Read-only loader and validator for `corpus.jsonl`. Confirms all 9 required fields, validates published date formats (YYYYMMDD), enforces unique chunk IDs, and preserves original source text.
  - **`src/researcher.py`:** Core evidence verification engine. Manages LLM provider clients (Groq primary, Gemini secondary), formats grounded prompts with temporal conflict directives, and validates inline citations using regex matching against retrieved chunks.
  - **`src/memory.py`:** Implements `SQLiteMemoryStore` for persistent key-value memory storage with parameterized SQL queries, case-insensitive key recall, and strict user-scoped isolation.
- **`corpus.jsonl`:** The immutable official knowledge base containing 75 records across 25 documents. Preserved byte-for-byte unchanged (verified via SHA-256 hash).
- **`eval/`:** Comprehensive automated evaluation framework:
  - **`eval/benchmark_dataset.json`:** 15 carefully curated benchmark evaluation questions covering factual retrieval, temporal supersession, version conflicts, negative feature grounding, and memory isolation.
  - **`eval/eval_runner.py`:** Automated benchmark runner measuring routing accuracy, retrieval recall @ top-5, citation validity, factual accuracy, and empirical A/B improvement over baseline RAG.
  - **`eval/results.json`:** Serialized test execution outputs, latencies, and evaluation scores.
  - **`eval/README.md`:** Detailed evaluation documentation, methodology, metric formulas, and failure analysis.
- **`test_stage1.py` through `test_stage4.py`:** Existing stage-specific automated test suites covering corpus schema validation, indexing, vector retrieval, citation checking, live Groq inference, LangGraph routing, and SQLite memory operations.
- **`test_cloud_fixes.py`:** Focused unit test suite verifying ChromaDB cold-start auto-indexing, strict count of 75, idempotency, search trigger, Streamlit Secrets compatibility, and safe error handling.
- **`requirements.txt`:** Specifies pinned Python dependencies required to run the application, test suites, and evaluation runner.
- **`.env.example`:** Example environment variable configuration containing placeholder keys for Groq, Gemini, and LangSmith. Contains zero secrets.
- **`README.md`:** Primary project documentation, installation instructions, usage guide, test commands, and architectural breakdown.
- **`DESIGN.md`:** In-depth architectural specification, Mermaid workflow diagrams, state schema details, precedence rules, and trade-off analyses.
- **`AI_DISCLOSURE.md`:** Formal disclosure of AI coding assistants used during implementation and testing, along with human review and empirical verification evidence.
- **`LANGSMITH_EVIDENCE.md`:** Verified public LangSmith trace links for inspectable execution runs with detailed node-by-node analysis.
- **`.gitignore`:** Standard Git exclusion file preventing accidental commits of virtual environments (`.venv/`), local secrets (`.env`), persistent databases (`data/`), and runtime cache files.

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

### Streamlit Cloud Secrets Compatibility

When deploying Asterline to **Streamlit Community Cloud**, you do not need to create or upload a `.env` file. You can configure your credentials directly in the Streamlit Cloud management dashboard under **App Settings &rarr; Secrets**:

```toml
# Streamlit Community Cloud Secrets (TOML format)
GROQ_API_KEY = "gsk_your_actual_groq_key_here"
GROQ_MODEL = "qwen/qwen3.8-27b"

# Optional: Google Gemini fallback
GEMINI_API_KEY = "your_gemini_key_here"

# Optional: LangSmith Tracing
LANGSMITH_TRACING = "false"
LANGSMITH_API_KEY = "your_langsmith_key_here"
LANGSMITH_PROJECT = "asterline-research-assistant"
```

The application automatically checks `st.secrets` for `GROQ_API_KEY` and `GEMINI_API_KEY` when environment variables are unset, and safely synchronizes string secrets into `os.environ` at startup so downstream libraries and background agents can access them transparently without exposing credentials.

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

### 6. Cloud-Deployment Fixes Verification
Verifies ChromaDB cold-start auto-indexing on empty storage, strict 75-record population, idempotency, search auto-indexing trigger, and Streamlit Community Cloud Secrets compatibility:
```powershell
python test_cloud_fixes.py
```
*Result: 6/6 tests passed.*

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

For complete evaluation methodology, failure analysis, and benchmark details, see [eval/README.md](eval/README.md). For detailed architecture specifications, see [DESIGN.md](DESIGN.md). For AI usage disclosure, see [AI_DISCLOSURE.md](AI_DISCLOSURE.md). For inspectable execution traces, see [LANGSMITH_EVIDENCE.md](LANGSMITH_EVIDENCE.md).

---

## Documentation Index

Comprehensive project documentation is maintained across dedicated Markdown files:

- **[`DESIGN.md`](DESIGN.md):** Complete system architecture specifications, Mermaid state machine diagrams, agent responsibilities, information flow, memory approach, and architectural trade-offs.
- **[`AI_DISCLOSURE.md`](AI_DISCLOSURE.md):** Detailed disclosure of AI coding assistants used during development, debugging, testing, and evaluation, accompanied by human oversight and empirical verification statements.
- **[`LANGSMITH_EVIDENCE.md`](LANGSMITH_EVIDENCE.md):** Verified public LangSmith trace links for inspectable execution runs (Enterprise SLA research run and Cold Recovery Retention conflict resolution run).
- **[`eval/README.md`](eval/README.md):** Benchmark methodology, evaluation metric formulas, empirical A/B measured improvement analysis, and failure mode documentation.
- **[`eval/benchmark_dataset.json`](eval/benchmark_dataset.json):** The 15 curated benchmark evaluation cases across document grounding, temporal supersession, and memory operations.
- **[`eval/results.json`](eval/results.json):** The serialized output metrics and latencies produced by `eval/eval_runner.py`.

---

## Known Limitations

1. **Closed Domain Corpus:** The assistant is grounded specifically on the 75 chunks of `corpus.jsonl`. Queries completely outside this domain will be rejected or report that no documentation exists.
2. **Single-Node SQLite Storage:** Memory persistence is managed via a local SQLite database (`data/memory.db`). For horizontally scaled multi-container deployments, a centralized relational database (e.g., PostgreSQL) would be required.
3. **LLM API Latency:** Answer generation depends on external API calls to Groq Cloud. In benchmark testing, observed response latencies ranged between ~0.4s and 1.2s per generation call depending on network traffic and prompt length.
4. **LangSmith Tracing:** LangSmith integration is supported via environment variables (`LANGSMITH_TRACING=true` or legacy `LANGCHAIN_TRACING_V2=true`), but active web trace recording requires a valid LangSmith API key.