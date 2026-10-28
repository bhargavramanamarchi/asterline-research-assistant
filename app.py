"""
app.py - Asterline Research Assistant
Interactive Streamlit Chat UI for Multi-Agent Research & Persistent Memory.

Styled in a premium futuristic AI SaaS aesthetic inspired by next-gen agent orchestration:
  - Deep pitch-black background with luminous turquoise / deep teal radial glow
  - Dark glassmorphism surfaces with fine neon-accented borders
  - Crisp white headings, muted silvery text, and luminous cyan badges
  - Compact LangGraph agent trace visualization
  - Expandable dark cards for retrieved evidence and grounded citations
  - Live user-scoped SQLite persistent memory management
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv(override=True)

from src.agents.graph import run_agent
from src.memory import SQLiteMemoryStore
from src.researcher import get_active_provider, sanitize_error

# Page configuration
st.set_page_config(
    page_title="Asterline Research Assistant",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Futuristic AI SaaS Styling (Near-black, deep teal aurora, glassmorphism)
st.markdown(
    """
    <style>
    /* Global App Canvas & Background */
    .stApp {
        background-color: #06090e;
        background-image: 
            radial-gradient(circle at 50% -10%, rgba(13, 148, 136, 0.28) 0%, rgba(6, 182, 212, 0.10) 35%, rgba(6, 9, 14, 0) 75%),
            radial-gradient(circle at 90% 40%, rgba(20, 184, 166, 0.08) 0%, rgba(6, 9, 14, 0) 50%),
            radial-gradient(circle at 10% 80%, rgba(6, 182, 212, 0.06) 0%, rgba(6, 9, 14, 0) 50%);
        background-attachment: fixed;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
    }

    /* Sidebar Glassmorphic Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(9, 14, 22, 0.92) !important;
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(45, 212, 191, 0.16) !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(45, 212, 191, 0.14) !important;
    }

    /* Headings and Branding */
    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 9999px;
        background: rgba(13, 148, 136, 0.15);
        border: 1px solid rgba(45, 212, 191, 0.35);
        color: #2dd4bf;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.75rem;
    }
    .main-header {
        font-size: 2.35rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.025em;
        line-height: 1.15;
        margin-bottom: 0.35rem;
        text-shadow: 0 0 35px rgba(45, 212, 191, 0.28);
    }
    .sub-header {
        font-size: 1.02rem;
        color: #94a3b8;
        margin-bottom: 1.8rem;
        letter-spacing: 0.01em;
    }

    /* Chat Messages Glassmorphism */
    [data-testid="stChatMessage"] {
        background: rgba(13, 20, 33, 0.65) !important;
        border: 1px solid rgba(45, 212, 191, 0.15) !important;
        border-radius: 14px !important;
        padding: 16px 20px !important;
        margin-bottom: 16px !important;
        backdrop-filter: blur(14px);
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.5), 0 0 15px rgba(6, 182, 212, 0.04);
        color: #f1f5f9 !important;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {
        color: #e2e8f0 !important;
        font-size: 0.98rem;
        line-height: 1.65;
    }

    /* Agent Execution Trace Pipeline */
    .trace-wrapper {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin: 12px 0 10px 0;
        padding: 8px 14px;
        border-radius: 10px;
        background: rgba(6, 11, 20, 0.85);
        border: 1px solid rgba(45, 212, 191, 0.2);
    }
    .trace-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(20, 184, 166, 0.12);
        color: #5eead4;
        border: 1px solid rgba(45, 212, 191, 0.3);
        font-size: 0.82rem;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        letter-spacing: 0.02em;
    }
    .intent-badge {
        display: inline-flex;
        align-items: center;
        padding: 4px 10px;
        border-radius: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 0.8rem;
        font-weight: 700;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .trace-arrow {
        color: #0d9488;
        font-weight: 700;
        font-size: 0.9rem;
    }

    /* Citation Tags */
    .citation-tag {
        display: inline-flex;
        align-items: center;
        padding: 3px 9px;
        margin: 3px 4px 3px 0;
        border-radius: 6px;
        background: rgba(13, 148, 136, 0.16);
        color: #2dd4bf;
        border: 1px solid rgba(45, 212, 191, 0.38);
        font-size: 0.82rem;
        font-weight: 600;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        box-shadow: 0 0 12px rgba(20, 184, 166, 0.15);
    }

    /* Evidence Drawer Cards */
    .evidence-card {
        border-left: 3px solid #2dd4bf;
        padding: 12px 16px;
        margin: 10px 0;
        background: rgba(8, 14, 24, 0.85);
        border-top: 1px solid rgba(255, 255, 255, 0.06);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 0 10px 10px 0;
        font-size: 0.88rem;
        color: #cbd5e1;
    }
    .evidence-header {
        color: #ffffff;
        font-weight: 700;
        font-size: 0.95rem;
    }
    .evidence-meta {
        color: #06b6d4;
        font-size: 0.8rem;
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
        margin-bottom: 6px;
    }

    /* Status Card in Sidebar */
    .status-card {
        background: rgba(6, 11, 20, 0.8);
        border: 1px solid rgba(45, 212, 191, 0.16);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
        font-size: 0.84rem;
        color: #94a3b8;
    }
    .status-card strong {
        color: #f1f5f9;
    }

    /* Expanders & Accordions */
    [data-testid="stExpander"] {
        background: rgba(9, 15, 25, 0.65) !important;
        border: 1px solid rgba(45, 212, 191, 0.18) !important;
        border-radius: 10px !important;
        margin-top: 12px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stExpander"] summary {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    [data-testid="stExpander"] summary:hover {
        color: #2dd4bf !important;
    }

    /* Input Fields & Buttons */
    [data-testid="stChatInput"] textarea {
        background: rgba(13, 20, 33, 0.85) !important;
        border: 1px solid rgba(45, 212, 191, 0.3) !important;
        color: #f8fafc !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #2dd4bf !important;
        box-shadow: 0 0 20px rgba(45, 212, 191, 0.3) !important;
    }
    .stButton > button {
        background: rgba(13, 148, 136, 0.14) !important;
        color: #f1f5f9 !important;
        border: 1px solid rgba(45, 212, 191, 0.3) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: rgba(13, 148, 136, 0.28) !important;
        border-color: #2dd4bf !important;
        color: #ffffff !important;
        box-shadow: 0 0 16px rgba(45, 212, 191, 0.25) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_memory_store() -> SQLiteMemoryStore:
    """Returns persistent SQLite memory store instance."""
    return SQLiteMemoryStore()


memory_store = get_memory_store()

# ----------------------------------------------------------------------
# SIDEBAR: User Session, Memory Context, System Architecture
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="brand-pill">⚡ Asterline Labs</div>', unsafe_allow_html=True)
    st.markdown("### Control Center")

    # User ID Switcher (Demonstrating Multi-User Isolation)
    st.caption("USER SCOPE & CONTEXT")
    default_users = ["user_alice", "user_bob", "user_demo", "custom"]
    selected_option = st.selectbox(
        "Active User Session:",
        options=default_users,
        index=0,
        help="Memories are strictly isolated per User ID in the SQLite database.",
    )

    if selected_option == "custom":
        user_id = st.text_input("Enter custom User ID:", value="user_charlie").strip()
    else:
        user_id = selected_option

    if not user_id:
        user_id = "default_user"

    st.markdown(f"<small>Connected user: <b><code style='color: #2dd4bf;'>{user_id}</code></b></small>", unsafe_allow_html=True)

    # Display Active User's SQLite Memories
    st.divider()
    st.markdown("### 🧠 Persistent Memories")
    user_memories = memory_store.get_memories(user_id)

    if user_memories:
        for k, v in user_memories.items():
            st.markdown(
                f"""
                <div class="status-card">
                  <strong>{k}:</strong> <span style="color: #2dd4bf;">{v}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("No saved personal context. Try: *'Remember that my tier is Enterprise'*.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            memory_store.clear_memories(user_id)
            st.success(f"Cleared memories for {user_id}")
            st.rerun()

    # System Architecture Status
    st.divider()
    st.markdown("### 📡 System Architecture")
    provider_name, _ = get_active_provider()
    provider_label = provider_name.upper() if provider_name != "none" else "OFFLINE"
    model_name = (
        os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
        if provider_name == "groq"
        else os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    )

    st.markdown(
        f"""
        <div class="status-card">
          &bull; <b>Provider:</b> <span style="color: #2dd4bf;">{provider_label}</span><br/>
          &bull; <b>Model:</b> <span style="color: #38bdf8;">{model_name}</span><br/>
          &bull; <b>Vector DB:</b> <span>Chroma (75 chunks)</span><br/>
          &bull; <b>Memory DB:</b> <span>SQLite (data/memory.db)</span><br/>
          &bull; <b>Orchestrator:</b> <span>LangGraph StateGraph</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ----------------------------------------------------------------------
# MAIN CHAT INTERFACE
# ----------------------------------------------------------------------
st.markdown('<div class="brand-pill">⚡ Multi-Agent Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="main-header">Asterline Research Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Documentary Evidence Verification &bull; Temporal Supersession &bull; User-Scoped Memory</div>',
    unsafe_allow_html=True,
)

# Initialize session chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display metadata for assistant responses
        if msg["role"] == "assistant":
            # 1. Execution Trace Badges
            trace = msg.get("trace", [])
            intent = msg.get("intent", "")
            if trace:
                arrow = '<span class="trace-arrow">&nbsp;➔&nbsp;</span>'
                trace_nodes = arrow.join(
                    f'<span class="trace-badge">{node}</span>' for node in trace
                )
                intent_html = f'<span class="intent-badge">{intent}</span>'
                st.markdown(
                    f'<div class="trace-wrapper">{intent_html} {trace_nodes}</div>',
                    unsafe_allow_html=True,
                )

            # 2. Citations Badges
            citations = msg.get("citations", [])
            if citations:
                citations_html = "".join(
                    f'<span class="citation-tag">[{c}]</span>' for c in citations
                )
                st.markdown(f"**Verified Documentary Sources:** {citations_html}", unsafe_allow_html=True)

            # 3. Expandable Evidence Chunks
            evidence = msg.get("evidence", [])
            if evidence:
                with st.expander(f"🔍 View Retrieved Documentary Evidence ({len(evidence)} chunks)"):
                    for idx, chunk in enumerate(evidence, start=1):
                        st.markdown(
                            f"""
                            <div class="evidence-card">
                              <div class="evidence-header">[{chunk.get('chunk_id')}] {chunk.get('title')}</div>
                              <div class="evidence-meta">Published: {chunk.get('published')} &bull; Version: {chunk.get('version')} &bull; Score: {chunk.get('score', 0):.4f}</div>
                              <small style="color: #64748B;">Source URL: {chunk.get('source_url')}</small>
                              <p style="margin-top: 6px; white-space: pre-wrap;">{chunk.get('text')}</p>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            # 4. Routing Plan / Rationale
            plan = msg.get("plan")
            if plan:
                with st.expander("📋 Planner Routing Rationale"):
                    st.write(plan)


# Example Quick Prompts when chat is empty
if len(st.session_state.messages) == 0:
    st.markdown("<p style='color: #94A3B8; margin-top: 1rem;'>Select a sample research query or personal context prompt:</p>", unsafe_allow_html=True)
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("📜 Enterprise Uptime SLA", use_container_width=True):
            st.session_state.preset_prompt = "What is the contractual monthly uptime SLA for Enterprise, and what are the service credit tiers?"
    with col_b:
        if st.button("⏳ Cold Recovery Conflict", use_container_width=True):
            st.session_state.preset_prompt = "How many days are raw run records held in cold recovery storage after searchable retention ends?"
    with col_c:
        if col_c.button("🧠 Hybrid (Context + SLA)", use_container_width=True):
            st.session_state.preset_prompt = "Remember that my tier is Enterprise, and tell me our monthly SLA uptime."


# Handle preset prompt submission or direct text input
user_input = st.chat_input("Ask an Asterline research question or manage personal preferences...")
prompt_to_run = user_input or st.session_state.pop("preset_prompt", None)

if prompt_to_run:
    # 1. Render user message
    st.session_state.messages.append({"role": "user", "content": prompt_to_run})
    with st.chat_message("user"):
        st.markdown(prompt_to_run)

    # 2. Invoke LangGraph Workflow
    with st.chat_message("assistant"):
        with st.spinner("Multi-agent workflow executing (Planner ➔ Memory / Researcher)..."):
            try:
                result = run_agent(query=prompt_to_run, user_id=user_id)
            except Exception as exc:
                sanitized = sanitize_error(str(exc))
                result = {
                    "status": "error",
                    "answer": f"Workflow execution error: {sanitized}",
                    "execution_trace": ["error"],
                    "intent": "unknown",
                    "plan": "Error occurred during execution.",
                    "citations": [],
                    "retrieved_chunks": [],
                }

        answer_text = result.get("answer", "No response generated.")
        st.markdown(answer_text)

        trace = result.get("execution_trace", [])
        intent = result.get("intent", "")
        plan = result.get("plan", "")
        raw_citations = result.get("citations", [])
        citations = [c["chunk_id"] if isinstance(c, dict) else str(c) for c in raw_citations]
        evidence = result.get("retrieved_chunks", [])

        # Display trace badges
        if trace:
            arrow = '<span class="trace-arrow">&nbsp;➔&nbsp;</span>'
            trace_nodes = arrow.join(
                f'<span class="trace-badge">{node}</span>' for node in trace
            )
            intent_html = f'<span class="intent-badge">{intent}</span>'
            st.markdown(
                f'<div class="trace-wrapper">{intent_html} {trace_nodes}</div>',
                unsafe_allow_html=True,
            )

        # Display citation badges
        if citations:
            citations_html = "".join(
                f'<span class="citation-tag">[{c}]</span>' for c in citations
            )
            st.markdown(f"**Verified Documentary Sources:** {citations_html}", unsafe_allow_html=True)

        # Expandable evidence chunks
        if evidence:
            with st.expander(f"🔍 View Retrieved Documentary Evidence ({len(evidence)} chunks)"):
                for idx, chunk in enumerate(evidence, start=1):
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                          <div class="evidence-header">[{chunk.get('chunk_id')}] {chunk.get('title')}</div>
                          <div class="evidence-meta">Published: {chunk.get('published')} &bull; Version: {chunk.get('version')} &bull; Score: {chunk.get('score', 0):.4f}</div>
                          <small style="color: #64748B;">Source URL: {chunk.get('source_url')}</small>
                          <p style="margin-top: 6px; white-space: pre-wrap;">{chunk.get('text')}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Expandable planning rationale
        if plan:
            with st.expander("📋 Planner Routing Rationale"):
                st.write(plan)

    # 3. Store assistant message in session history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer_text,
        "trace": trace,
        "intent": intent,
        "plan": plan,
        "citations": citations,
        "evidence": evidence,
    })

    # Rerun to update sidebar memories live if a memory command was processed
    if "memory" in intent or result.get("memory_action") != "none":
        st.rerun()
