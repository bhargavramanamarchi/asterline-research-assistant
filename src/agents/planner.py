"""
src/agents/planner.py - Asterline Research Assistant
Planner Agent for Request Classification and Routing.

Classifies incoming queries into:
  - 'research': Questions strictly about Asterline Labs, products, policies, SLAs, docs.
  - 'memory': Personal memory operations (store, recall, forget, inspect preferences).
  - 'hybrid': Involves personal user context/preferences AND Asterline documentary research.
"""

import re
from typing import Any, Dict, Literal, Optional, Tuple

from src.agents.state import AgentState


def extract_memory_details(query: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extracts memory action, key, and value from a memory-bearing query.
    Returns: (action, key, value)
    """
    q_lower = query.strip().lower()

    # Forget / Delete / Clear patterns
    if re.search(r"\b(?:forget\s+all|clear\s+(?:all\s+)?(?:my\s+)?memories|delete\s+(?:all\s+)?(?:my\s+)?memories|reset\s+memories)\b", q_lower):
        return ("clear", None, None)

    m_forget = re.search(r"\b(?:forget|delete|remove)\s+(?:my\s+)?([a-z0-9_\s\-]+)", q_lower)
    if m_forget:
        key = m_forget.group(1).strip()
        key = re.sub(r"[?!.]+$", "", key).strip()
        return ("forget", key, None)

    # Store patterns
    # e.g., "Remember that my preferred export format is Parquet"
    # e.g., "Remember that my name is Alice and my role is Site Reliability Engineer"
    # e.g., "Remember that I am on Enterprise tier"
    m_remember = re.search(
        r"\b(?:remember(?:\s+that)?|save|store)\s+(?:my\s+)?([^:]+?)\s+(?:is|to|as|=|:)\s+(.+)",
        query,
        re.IGNORECASE,
    )
    if m_remember:
        raw_key = m_remember.group(1).strip()
        raw_val = m_remember.group(2).strip()
        raw_val = re.sub(r"[?!.]+$", "", raw_val).strip()
        if "," in raw_val:
            parts = raw_val.split(",", 1)
            raw_val = parts[0].strip()
        elif " and " in raw_val.lower():
            parts = re.split(r"\s+and\s+", raw_val, maxsplit=1, flags=re.IGNORECASE)
            if any(w in parts[1].lower() for w in ["what", "tell", "how", "sla", "does", "is"]):
                raw_val = parts[0].strip()
        return ("store", raw_key, raw_val)

    # "Remember I am on the Enterprise tier" / "Remember that my tier is Enterprise"
    m_remember_am = re.search(
        r"\bremember(?:\s+that)?\s+i\s+am\s+(?:on\s+)?(?:the\s+)?([a-z0-9_\s\-]+?)(?:tier|plan)?(?=[,.?]|$)",
        query,
        re.IGNORECASE,
    )
    if m_remember_am:
        val = m_remember_am.group(1).strip().capitalize()
        return ("store", "tier", val)

    # "My name is Alice" / "My preferred region is APAC"
    m_my = re.search(r"\bmy\s+([a-z0-9_\s\-]+?)\s+(?:is|=|:)\s+([^,.?!]+)", query, re.IGNORECASE)
    if m_my:
        k = m_my.group(1).strip()
        v = m_my.group(2).strip()
        return ("store", k, v)

    # Recall patterns
    if re.search(r"\b(?:what do you remember|show memories|list memories|inspect memories)\b", q_lower):
        return ("recall", None, None)

    m_recall = re.search(r"\b(?:what is|what's|tell me)\s+my\s+([a-z0-9_\s\-]+)", q_lower)
    if m_recall:
        k = m_recall.group(1).strip()
        k = re.sub(r"[?!.]+$", "", k).strip()
        return ("recall", k, None)

    return (None, None, None)


def classify_intent(
    query: str,
    user_memories: Optional[Dict[str, str]] = None,
) -> Tuple[Literal["research", "memory", "hybrid"], str, Optional[str], Optional[str], Optional[str]]:
    """
    Analyzes query and classifies into 'research', 'memory', or 'hybrid'.
    
    Returns:
      (intent, plan_explanation, memory_action, memory_key, memory_value)
    """
    q_lower = query.strip().lower()
    mem_action, mem_key, mem_val = extract_memory_details(query)

    # Research topic indicators from Asterline corpus
    company_research_keywords = [
        "sla", "uptime", "service credit", "cold recovery", "cold storage",
        "retention", "aql", "timeout", "release", "version", "asia-pacific",
        "apac", "region", "on-call", "stipend", "handbook", "signals",
        "alert", "incident", "asterline", "policy", "features", "workflow",
        "worker", "architecture", "audit", "compliance", "searchable",
        "run records", "tier", "enterprise", "standard"
    ]

    has_research_keyword = any(ind in q_lower for ind in company_research_keywords)

    # Indicators that a research question is actively being asked
    has_question_phrase = bool(
        "?" in query
        or re.search(r"\b(?:what is|what's|what are|tell me|how long|how do|how does|does|can|has|when will|show me)\b", q_lower)
        or re.search(r"\b(?:and tell me|and what|and how|and does)\b", q_lower)
    )

    # Explicit memory triggers
    memory_triggers = [
        "remember", "remind", "save my", "store my", "forget", "clear all my memories",
        "clear my memories", "what do you remember", "what is my", "what's my", "my name is",
        "my role is", "my preferred", "i prefer", "as i told you", "based on my",
        "given my role", "given my", "i am an", "i am a", "for my account"
    ]
    has_memory_trigger = any(trig in q_lower for trig in memory_triggers) or (mem_action is not None)

    # Classification Logic:
    # 1. Pure Memory:
    #    - Actions like forget or clear (without asking a research question)
    #    - Pure recall: "what do you remember about me?", "what is my preferred export format?"
    #    - Pure store: "remember that my preferred export format is Parquet" (no question attached)
    # 2. Hybrid:
    #    - Memory trigger/action present AND an active research question is being asked
    #    - Or existing user_memories context referenced in a research inquiry
    # 3. Pure Research:
    #    - Company research question without personal memory triggers
    
    is_hybrid = False
    if has_memory_trigger:
        if mem_action in ("clear", "forget") and not has_question_phrase:
            is_hybrid = False
        elif mem_action == "recall":
            # If asking about memory ("what do you remember about me", "what is my preferred export format")
            # without referencing company research docs, it's pure memory
            if not has_research_keyword:
                is_hybrid = False
            else:
                is_hybrid = True
        elif mem_action == "store":
            # If storing something AND also asking a question (e.g. "?", "and tell me", research topic)
            if has_question_phrase and has_research_keyword:
                is_hybrid = True
            else:
                is_hybrid = False
        else:
            # Memory trigger without explicit store/recall action (e.g. "Given my role as backend engineer, what is...")
            if has_research_keyword or has_question_phrase:
                is_hybrid = True
    elif user_memories and has_research_keyword:
        if any(w in q_lower for w in ["what is my", "what's my", "my tier", "my sla", "my stipend"]):
            is_hybrid = True

    # Final Classification
    if is_hybrid:
        intent: Literal["research", "memory", "hybrid"] = "hybrid"
        plan = (
            "Hybrid request detected: Query combines personal user context/memory with Asterline "
            "documentary research. Routing first to Memory Manager to load or update user context, "
            "then to Researcher Agent to synthesize a grounded answer using Asterline documentation."
        )
    elif has_memory_trigger:
        intent = "memory"
        action_name = mem_action or "inspect"
        plan = (
            f"Personal memory management request detected ({action_name}). "
            "Routing directly to Memory Manager. Note: Personal memories reflect user context only "
            "and are not Asterline company documentation."
        )
    else:
        intent = "research"
        plan = (
            "Asterline company research query detected. Routing directly to Researcher Agent "
            "to search persistent ChromaDB and generate a grounded, citation-backed answer."
        )

    return (intent, plan, mem_action, mem_key, mem_val)


def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Planner Node implementation for LangGraph.
    
    Reads: state["query"], state.get("user_memories")
    Updates: intent, plan, memory_action, memory_key, memory_value, execution_trace
    """
    query = state.get("query", "").strip()
    user_memories = state.get("user_memories", {})

    intent, plan, mem_action, mem_key, mem_val = classify_intent(query, user_memories)

    trace = list(state.get("execution_trace", []))
    trace.append("planner")

    return {
        "intent": intent,
        "plan": plan,
        "memory_action": mem_action or "none",
        "memory_key": mem_key,
        "memory_value": mem_val,
        "execution_trace": trace,
    }
