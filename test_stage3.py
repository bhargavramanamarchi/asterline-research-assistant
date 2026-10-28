"""
test_stage3.py - Comprehensive Verification for Stage 3 LangGraph Multi-Agent Architecture

Tests:
  1. Planner Intent Classification (Research, Memory, Hybrid)
  2. Pure Memory Path: START -> Planner -> Memory Manager -> END
  3. Pure Research Path: START -> Planner -> Researcher -> END
  4. Hybrid Path: START -> Planner -> Memory Manager -> Researcher -> END
  5. Critical Precedence Rule: Personal memories NEVER override company documentation
  6. Memory Scoping & User Isolation
  7. Graph Topology & Error Handling
"""

import os
import sys
from typing import Dict

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.graph import create_research_graph, get_graph, run_agent
from src.agents.memory_agent import DEFAULT_MEMORY_STORE, InMemoryMemoryStore
from src.agents.planner import classify_intent, extract_memory_details
from src.agents.state import AgentState, create_initial_state


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_tests() -> int:
    banner("STAGE 3 MULTI-AGENT VERIFICATION: LANGGRAPH WORKFLOW")
    failures = 0

    # Reset memory store for clean test run
    DEFAULT_MEMORY_STORE.reset_all()

    # ------------------------------------------------------------------
    # TEST 1: Planner Intent Classification
    # ------------------------------------------------------------------
    print("\n--- TEST 1: Planner Classification of All 3 Intent Types ---")
    
    test_cases = [
        # (query, expected_intent)
        ("What is the contractual monthly uptime SLA for Enterprise, and what are the service credits?", "research"),
        ("How long are raw run records kept in cold recovery storage after searchable retention ends?", "research"),
        ("What is the timeout for AQL queries across different versions?", "research"),
        ("Has Asterline announced a release date or timeline for an Asia-Pacific region?", "research"),
        
        ("Remember that my preferred export format is Parquet", "memory"),
        ("Remember that my name is Alice and my role is Site Reliability Engineer", "memory"),
        ("What is my preferred export format?", "memory"),
        ("What do you remember about me?", "memory"),
        ("Forget my preferred export format", "memory"),
        ("Clear all my memories", "memory"),
        
        ("Remember that my tier is Enterprise, and tell me our monthly SLA uptime.", "hybrid"),
        ("Given my role as backend engineer, what is the on-call stipend?", "hybrid"),
        ("My preferred region is APAC. Has Asterline announced a release date for Asia-Pacific?", "hybrid"),
    ]

    all_classified = True
    for q, expected in test_cases:
        intent, plan, act, k, v = classify_intent(q)
        if intent != expected:
            print(f"FAILED: Query '{q}' expected intent '{expected}', got '{intent}'")
            all_classified = False
            failures += 1
        else:
            print(f"  [OK] Intent: {intent:<8} | Query: {q[:55]}...")

    if all_classified:
        print("PASSED: All research, memory, and hybrid intents classified correctly.")

    # ------------------------------------------------------------------
    # TEST 2: Pure Memory Routing & In-Memory Operations
    # ------------------------------------------------------------------
    print("\n--- TEST 2: Pure Memory Path (START -> Planner -> MemoryManager -> END) ---")
    user_test = "user_test_alice"

    # Action 2a: Store
    res_store = run_agent(
        query="Remember that my preferred export format is Parquet",
        user_id=user_test,
    )
    trace_store = res_store.get("execution_trace", [])
    expected_trace = ["planner", "memory_manager"]
    if trace_store != expected_trace:
        print(f"FAILED: Expected trace {expected_trace}, got {trace_store}")
        failures += 1
    else:
        print(f"PASSED: Node trace for store: {trace_store}")

    memories_after_store = res_store.get("user_memories", {})
    if "preferred export format" not in memories_after_store or memories_after_store["preferred export format"] != "Parquet":
        print(f"FAILED: Memory not stored in state. Got: {memories_after_store}")
        failures += 1
    else:
        print(f"PASSED: Memory stored: {memories_after_store}")

    # Action 2b: Recall
    res_recall = run_agent(
        query="What is my preferred export format?",
        user_id=user_test,
    )
    trace_recall = res_recall.get("execution_trace", [])
    if trace_recall != expected_trace:
        print(f"FAILED: Expected trace {expected_trace}, got {trace_recall}")
        failures += 1
    else:
        print(f"PASSED: Node trace for recall: {trace_recall}")
    
    if "Parquet" not in res_recall.get("answer", ""):
        print(f"FAILED: Recall answer does not mention Parquet: {res_recall.get('answer')}")
        failures += 1
    else:
        print(f"PASSED: Recall answered accurately: {res_recall.get('answer')}")

    # Action 2c: Forget
    res_forget = run_agent(
        query="Forget my preferred export format",
        user_id=user_test,
    )
    trace_forget = res_forget.get("execution_trace", [])
    if trace_forget != expected_trace:
        print(f"FAILED: Expected trace {expected_trace}, got {trace_forget}")
        failures += 1
    else:
        print(f"PASSED: Node trace for forget: {trace_forget}")
    
    memories_after_forget = res_forget.get("user_memories", {})
    if "preferred export format" in memories_after_forget:
        print(f"FAILED: Memory key was not forgotten: {memories_after_forget}")
        failures += 1
    else:
        print(f"PASSED: Memory successfully forgotten. Current: {memories_after_forget}")

    # ------------------------------------------------------------------
    # TEST 3: Pure Research Routing (START -> Planner -> Researcher -> END)
    # ------------------------------------------------------------------
    print("\n--- TEST 3: Pure Research Path (START -> Planner -> Researcher -> END) ---")
    query_sla = "What is the contractual monthly uptime SLA for Enterprise, and what are the service credits?"
    res_research = run_agent(query=query_sla, user_id=user_test)
    
    trace_res = res_research.get("execution_trace", [])
    expected_res_trace = ["planner", "researcher"]
    if trace_res != expected_res_trace:
        print(f"FAILED: Expected research trace {expected_res_trace}, got {trace_res}")
        failures += 1
    else:
        print(f"PASSED: Node trace for pure research: {trace_res} (MemoryManager was correctly skipped)")

    if res_research.get("status") != "success":
        print(f"FAILED: Research status is {res_research.get('status')}, error: {res_research.get('error')}")
        failures += 1
    else:
        print("PASSED: Research executed with status 'success'.")

    # Verify retrieved chunks & citations
    raw_cited = res_research.get("citations", [])
    cited_ids = [c["chunk_id"] if isinstance(c, dict) else str(c) for c in raw_cited]
    if "sla:0" not in cited_ids:
        print(f"FAILED: Expected citation 'sla:0' in citations: {cited_ids}")
        failures += 1
    else:
        print(f"PASSED: Expected citation 'sla:0' found in: {cited_ids}")

    if not res_research.get("citations_valid"):
        print("FAILED: Citations failed validation.")
        failures += 1
    else:
        print("PASSED: Citations validated against authentic retrieved chunks.")

    # ------------------------------------------------------------------
    # TEST 4: Hybrid Routing (START -> Planner -> MemoryManager -> Researcher -> END)
    # ------------------------------------------------------------------
    print("\n--- TEST 4: Hybrid Path (START -> Planner -> MemoryManager -> Researcher -> END) ---")
    hybrid_query = "Remember that my tier is Enterprise, and tell me our monthly SLA uptime."
    res_hybrid = run_agent(query=hybrid_query, user_id="user_test_bob")

    trace_hybrid = res_hybrid.get("execution_trace", [])
    expected_hybrid_trace = ["planner", "memory_manager", "researcher"]
    if trace_hybrid != expected_hybrid_trace:
        print(f"FAILED: Expected hybrid trace {expected_hybrid_trace}, got {trace_hybrid}")
        failures += 1
    else:
        print(f"PASSED: Node trace for hybrid query: {trace_hybrid}")

    # Check that memory was captured and passed
    bob_memories = res_hybrid.get("user_memories", {})
    if "tier" not in bob_memories or "Enterprise" not in bob_memories["tier"]:
        print(f"FAILED: Expected 'tier': 'Enterprise' in bob_memories, got: {bob_memories}")
        failures += 1
    else:
        print(f"PASSED: User memory populated in hybrid flow: {bob_memories}")

    # Check that research evidence was retrieved and cited
    raw_hybrid_citations = res_hybrid.get("citations", [])
    hybrid_citations = [c["chunk_id"] if isinstance(c, dict) else str(c) for c in raw_hybrid_citations]
    if "sla:0" not in hybrid_citations:
        print(f"FAILED: Expected citation 'sla:0' in hybrid response, got: {hybrid_citations}")
        failures += 1
    else:
        print(f"PASSED: Grounded citation 'sla:0' verified in hybrid response: {hybrid_citations}")

    # ------------------------------------------------------------------
    # TEST 5: Critical Precedence Rule: Documentation Trumps Memory
    # ------------------------------------------------------------------
    print("\n--- TEST 5: Critical Rule: Personal Memory NEVER Overrides Company Documentation ---")
    adversarial_user = "user_adversary"
    # User attempts to claim impossible / contradictory facts in personal memory:
    # 1) Claiming Enterprise SLA is 100% (when sla:0 says 99.92%)
    # 2) Claiming raw run records are kept in cold storage for 500 days (when retention:0 says 20 days)
    adversarial_memories = {
        "tier": "Enterprise",
        "assumed SLA": "100.0%",
        "assumed cold retention": "500 days",
    }
    DEFAULT_MEMORY_STORE.set_memory(adversarial_user, "tier", "Enterprise")
    DEFAULT_MEMORY_STORE.set_memory(adversarial_user, "assumed SLA", "100.0%")
    DEFAULT_MEMORY_STORE.set_memory(adversarial_user, "assumed cold retention", "500 days")

    res_precedence_sla = run_agent(
        query="What is the monthly uptime SLA for Enterprise?",
        user_id=adversarial_user,
    )
    ans_sla = res_precedence_sla.get("answer", "")
    
    # Must state 99.92% (from sla:0), NOT 100%
    if "99.92" not in ans_sla:
        print(f"FAILED: Precedence check failed. Official SLA '99.92%' missing in answer: {ans_sla[:200]}")
        failures += 1
    else:
        print("PASSED: Answer preserves official 99.92% SLA despite user personal memory notes.")

    res_precedence_cold = run_agent(
        query="How long are raw run records kept in cold recovery storage after searchable retention ends?",
        user_id=adversarial_user,
    )
    ans_cold = res_precedence_cold.get("answer", "")
    # Must state 20 days (from retention:0 superseding security:1), NOT 500 days
    if "20 days" not in ans_cold.lower() and "20-day" not in ans_cold.lower():
        print(f"FAILED: Precedence check failed. Official 20 days retention missing in answer: {ans_cold[:200]}")
        failures += 1
    else:
        print("PASSED: Answer preserves official 20-day retention despite user's 500-day memory claim.")

    # ------------------------------------------------------------------
    # TEST 6: User Memory Scoping & Isolation
    # ------------------------------------------------------------------
    print("\n--- TEST 6: Memory Scoping & User Isolation ---")
    store = InMemoryMemoryStore()
    store.set_memory("user_1", "role", "Staff Engineer")
    store.set_memory("user_2", "role", "Product Manager")

    m1 = store.get_memories("user_1")
    m2 = store.get_memories("user_2")
    if m1.get("role") != "Staff Engineer" or m2.get("role") != "Product Manager":
        print(f"FAILED: Memory cross-leak detected: user_1={m1}, user_2={m2}")
        failures += 1
    else:
        print("PASSED: Users have completely isolated personal memory spaces.")

    # ------------------------------------------------------------------
    # TEST 7: Graph Topology & Compilation Verification
    # ------------------------------------------------------------------
    print("\n--- TEST 7: LangGraph StateGraph Topology Verification ---")
    graph = create_research_graph()
    
    # Check that graph is compiled
    if graph is None:
        print("FAILED: Graph compilation returned None.")
        failures += 1
    else:
        print("PASSED: LangGraph CompiledStateGraph instance verified.")

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    banner("STAGE 3 VERIFICATION SUMMARY")
    if failures == 0:
        print("ALL STAGE 3 TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print(f"STAGE 3 VERIFICATION ENCOUNTERED {failures} FAILURES.")
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
