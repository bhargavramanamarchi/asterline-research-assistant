"""
src/agents/researcher_agent.py - Asterline Research Assistant
Researcher Agent Node for LangGraph.

Reuses Stage 1 persistent ChromaDB retrieval (search_corpus) and
Stage 2 evidence-grounded generation with citation validation.
Supports Groq (default provider) and Google Gemini (fallback).

CRITICAL INVARIANT:
Personal memories represent user context/preferences ONLY.
Asterline documentation is the SOLE source of truth for company facts.
Personal memories NEVER override company documentation.
"""

from typing import Any, Dict, List, Optional

from src.agents.state import AgentState
from src.indexer import search_corpus
from src.researcher import (
    format_evidence_prompt,
    generate_grounded_answer,
    validate_citations,
)


def enhance_query_with_context(query: str, user_memories: Dict[str, str]) -> str:
    """
    Synthesizes effective retrieval query by incorporating user context if referenced.
    For example, if query asks 'What is my SLA?' and memory has {'tier': 'Enterprise'},
    enhances query to include 'Enterprise SLA'.
    """
    if not user_memories:
        return query

    context_hints: List[str] = []
    q_lower = query.lower()

    for k, v in user_memories.items():
        k_lower = k.lower()
        if k_lower in q_lower or any(w in q_lower for w in ["my", "i am", "our", "based on"]):
            context_hints.append(f"{k}: {v}")

    if context_hints:
        return f"{query} (Context: {', '.join(context_hints)})"
    return query


def format_contextual_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    user_memories: Optional[Dict[str, str]] = None,
) -> str:
    """
    Formats the evidence prompt, explicitly segregating user context from company ground truth.
    Strictly instructs the LLM that personal memory NEVER overrides company documentation.
    """
    base_prompt = format_evidence_prompt(query, retrieved_chunks)

    if not user_memories:
        return base_prompt

    mem_lines = "\n".join(f"- {k}: {v}" for k, v in user_memories.items())

    context_block = f"""
USER PERSONAL CONTEXT (Preferences / Attributes only):
{mem_lines}

CRITICAL PRECEDENCE RULE:
- The user personal context above describes who the user is (e.g. tier, role, preferred format).
- Official Asterline documentation below is the SOLE GROUND TRUTH for company policies, SLAs, technical specifications, and procedures.
- PERSONAL MEMORIES MUST NEVER OVERRIDE COMPANY DOCUMENTATION.
- If a user memory contradicts official documentation, explain the official policy with citations and note that company documentation takes precedence over personal notes.
"""
    if "QUESTION:" in base_prompt:
        parts = base_prompt.split("QUESTION:", 1)
        return f"{parts[0]}{context_block}\nQUESTION:{parts[1]}"
    return f"{context_block}\n{base_prompt}"


def researcher_node(state: AgentState) -> Dict[str, Any]:
    """
    Researcher Node implementation for LangGraph.
    
    Reads: state['query'], state.get('user_memories')
    Updates: retrieved_chunks, answer, citations, citations_valid, status, execution_trace
    """
    raw_query = state.get("query", "").strip()
    user_memories = state.get("user_memories", {})

    # 1. Enhance query with user context if relevant
    search_query = enhance_query_with_context(raw_query, user_memories)

    # 2. Retrieve documentary evidence from ChromaDB
    retrieved_chunks = search_corpus(search_query, top_k=5)

    trace = list(state.get("execution_trace", []))
    trace.append("researcher")

    if not retrieved_chunks:
        return {
            "status": "no_evidence",
            "retrieved_chunks": [],
            "answer": "The retrieved documentation does not contain evidence to answer this question.",
            "citations": [],
            "citations_valid": False,
            "execution_trace": trace,
        }

    # 3. Format contextual prompt
    prompt = format_contextual_prompt(raw_query, retrieved_chunks, user_memories)

    # 4. Generate answer using active provider (Groq or Gemini)
    gen_result = generate_grounded_answer(prompt)

    if gen_result["status"] != "success":
        err_msg = gen_result.get("text") or gen_result.get("error_detail", "Generation failed")
        return {
            "status": gen_result["status"],
            "error": err_msg,
            "retrieved_chunks": retrieved_chunks,
            "answer": err_msg,
            "citations": [],
            "citations_valid": False,
            "execution_trace": trace,
        }

    answer_text = gen_result["text"]

    # 5. Validate citations
    validation = validate_citations(answer_text, retrieved_chunks)
    cited_ids = [c["chunk_id"] if isinstance(c, dict) else str(c) for c in validation["valid_citations"]]

    return {
        "status": "success",
        "retrieved_chunks": retrieved_chunks,
        "answer": answer_text,
        "citations": cited_ids,
        "citation_details": validation["valid_citations"],
        "citations_valid": validation["is_valid"],
        "execution_trace": trace,
    }
