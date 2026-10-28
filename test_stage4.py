"""
test_stage4.py - Comprehensive Verification for Stage 4 Persistent SQLite Memory

Tests:
  1. Memory Creation & Recall (case-insensitive lookup)
  2. Memory Listing (retrieving all user-scoped memories)
  3. Memory In-Place Update & Correction (updating without duplication)
  4. Memory Deletion & Clearing
  5. Persistence across Reconnection (surviving process/connection resets)
  6. Strict User Isolation (zero cross-user data leakage)
  7. LangGraph Multi-Agent Integration with Persistent SQLite Storage
"""

import os
from pathlib import Path
import sys
import tempfile

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.graph import run_agent
from src.agents.memory_agent import set_default_memory_store
from src.memory import SQLiteMemoryStore


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def run_stage4_tests() -> int:
    banner("STAGE 4 VERIFICATION: PERSISTENT SQLITE USER MEMORY")
    failures = 0

    with tempfile.TemporaryDirectory() as temp_dir:
        test_db_path = str(Path(temp_dir) / "test_stage4_isolated.db")
        print(f"Isolated test database created at: {test_db_path}")

        store = SQLiteMemoryStore(db_path=test_db_path)

        # --------------------------------------------------------------
        # TEST 1: Creation & Recall
        # --------------------------------------------------------------
        print("\n--- TEST 1: Memory Creation & Case-Insensitive Recall ---")
        store.set_memory("user_alice", "tier", "Enterprise")
        store.set_memory("user_alice", "preferred export format", "Parquet")

        val1 = store.get_memory("user_alice", "tier")
        val1_upper = store.get_memory("user_alice", "TIER")
        val2 = store.get_memory("user_alice", "preferred export format")

        if val1 != "Enterprise" or val1_upper != "Enterprise":
            print(f"FAILED: Expected 'Enterprise', got: '{val1}', upper: '{val1_upper}'")
            failures += 1
        else:
            print("PASSED: Memory created and recalled accurately (case-insensitive match verified).")

        if val2 != "Parquet":
            print(f"FAILED: Expected 'Parquet', got: '{val2}'")
            failures += 1
        else:
            print("PASSED: Multi-word key 'preferred export format' stored and recalled accurately.")

        # --------------------------------------------------------------
        # TEST 2: Listing
        # --------------------------------------------------------------
        print("\n--- TEST 2: Listing User-Scoped Memories ---")
        store.set_memory("user_alice", "role", "Lead SRE")
        alice_memories = store.get_memories("user_alice")

        expected_keys = {"tier", "preferred export format", "role"}
        if set(alice_memories.keys()) != expected_keys:
            print(f"FAILED: Expected keys {expected_keys}, got: {set(alice_memories.keys())}")
            failures += 1
        else:
            print(f"PASSED: Listed all {len(alice_memories)} memories for user_alice: {alice_memories}")

        # --------------------------------------------------------------
        # TEST 3: Update & In-Place Correction
        # --------------------------------------------------------------
        print("\n--- TEST 3: Update & In-Place Correction ---")
        # Update existing 'tier' from Enterprise to Team
        store.set_memory("user_alice", "tier", "Team")
        updated_val = store.get_memory("user_alice", "tier")
        updated_memories = store.get_memories("user_alice")

        if updated_val != "Team":
            print(f"FAILED: Expected updated tier 'Team', got: '{updated_val}'")
            failures += 1
        elif len(updated_memories) != 3:
            print(f"FAILED: Updating duplicated key instead of updating in-place. Count: {len(updated_memories)}")
            failures += 1
        else:
            print(f"PASSED: In-place update verified: tier is now '{updated_val}' (count remains 3).")

        # --------------------------------------------------------------
        # TEST 4: Deletion & Clearing
        # --------------------------------------------------------------
        print("\n--- TEST 4: Deletion & Clearing ---")
        deleted_ok = store.delete_memory("user_alice", "preferred export format")
        deleted_again = store.delete_memory("user_alice", "preferred export format")

        if not deleted_ok:
            print("FAILED: delete_memory returned False for existing key.")
            failures += 1
        elif deleted_again:
            print("FAILED: delete_memory returned True for already deleted key.")
            failures += 1
        else:
            print("PASSED: Single memory deleted successfully; reported accurate deletion status.")

        # Clear remaining
        store.clear_memories("user_alice")
        cleared_memories = store.get_memories("user_alice")
        if cleared_memories != {}:
            print(f"FAILED: Memories not cleared: {cleared_memories}")
            failures += 1
        else:
            print("PASSED: clear_memories removed all memories for user_alice.")

        # --------------------------------------------------------------
        # TEST 5: Persistence Across Reopening (Simulating Process Restart)
        # --------------------------------------------------------------
        print("\n--- TEST 5: Persistence Across Database Reopening ---")
        store.set_memory("user_persist", "on-call zone", "US-East")
        store.set_memory("user_persist", "theme", "Dark Glass")

        # Dereference existing store and open a completely new instance pointing to same file
        del store
        reopened_store = SQLiteMemoryStore(db_path=test_db_path)

        reopened_memories = reopened_store.get_memories("user_persist")
        if reopened_memories.get("on-call zone") != "US-East" or reopened_memories.get("theme") != "Dark Glass":
            print(f"FAILED: Persistence failed upon reopening database: {reopened_memories}")
            failures += 1
        else:
            print(f"PASSED: Data survived connection closure and reopen: {reopened_memories}")

        # --------------------------------------------------------------
        # TEST 6: Strict User Isolation
        # --------------------------------------------------------------
        print("\n--- TEST 6: Strict User Isolation ---")
        reopened_store.set_memory("user_alpha", "confidential_project", "Project A")
        reopened_store.set_memory("user_beta", "confidential_project", "Project B")

        alpha_memories = reopened_store.get_memories("user_alpha")
        beta_memories = reopened_store.get_memories("user_beta")
        gamma_memories = reopened_store.get_memories("user_gamma")

        if alpha_memories.get("confidential_project") != "Project A":
            print(f"FAILED: User alpha memory corrupted: {alpha_memories}")
            failures += 1
        elif beta_memories.get("confidential_project") != "Project B":
            print(f"FAILED: User beta memory corrupted: {beta_memories}")
            failures += 1
        elif gamma_memories != {}:
            print(f"FAILED: Nonexistent user gamma read other users' memories: {gamma_memories}")
            failures += 1
        else:
            print("PASSED: Zero cross-user memory leakage. Strict user-scoped isolation verified.")

        # --------------------------------------------------------------
        # TEST 7: LangGraph Multi-Agent Integration with Persistent Store
        # --------------------------------------------------------------
        print("\n--- TEST 7: LangGraph Multi-Agent Integration with SQLite ---")
        # Inject the isolated test store into the Memory Manager node
        set_default_memory_store(reopened_store)

        try:
            # 7a. LangGraph Store Command
            user_graph_test = "user_graph_stage4"
            res_store = run_agent(
                query="Remember that my preferred export format is Parquet",
                user_id=user_graph_test,
            )
            trace_store = res_store.get("execution_trace", [])
            if trace_store != ["planner", "memory_manager"]:
                print(f"FAILED: Expected trace ['planner', 'memory_manager'], got: {trace_store}")
                failures += 1
            else:
                print(f"PASSED: Node trace for store in StateGraph: {trace_store}")

            # Verify saved on disk in SQLite
            disk_val = reopened_store.get_memory(user_graph_test, "preferred export format")
            if disk_val != "Parquet":
                print(f"FAILED: Memory not found in SQLite database on disk: {disk_val}")
                failures += 1
            else:
                print(f"PASSED: Memory confirmed persisted directly in SQLite database: '{disk_val}'")

            # 7b. LangGraph Recall Command
            res_recall = run_agent(
                query="What is my preferred export format?",
                user_id=user_graph_test,
            )
            trace_recall = res_recall.get("execution_trace", [])
            if trace_recall != ["planner", "memory_manager"]:
                print(f"FAILED: Expected trace ['planner', 'memory_manager'], got: {trace_recall}")
                failures += 1
            elif "Parquet" not in res_recall.get("answer", ""):
                print(f"FAILED: Recall answer does not mention Parquet: {res_recall.get('answer')}")
                failures += 1
            else:
                print(f"PASSED: StateGraph recalled memory from SQLite: {res_recall.get('answer')}")

            # 7c. LangGraph Hybrid Command (Memory Update + Research Answering)
            res_hybrid = run_agent(
                query="Remember that my tier is Enterprise, and tell me our monthly SLA uptime.",
                user_id=user_graph_test,
            )
            trace_hybrid = res_hybrid.get("execution_trace", [])
            expected_hybrid_trace = ["planner", "memory_manager", "researcher"]
            if trace_hybrid != expected_hybrid_trace:
                print(f"FAILED: Expected hybrid trace {expected_hybrid_trace}, got: {trace_hybrid}")
                failures += 1
            else:
                print(f"PASSED: Hybrid trace verified: {trace_hybrid}")

            # Verify tier was persisted to SQLite
            persisted_tier = reopened_store.get_memory(user_graph_test, "tier")
            if persisted_tier != "Enterprise":
                print(f"FAILED: Tier not persisted in SQLite: {persisted_tier}")
                failures += 1
            else:
                print(f"PASSED: Hybrid memory write persisted in SQLite: tier = '{persisted_tier}'")

            # Verify grounded SLA answer
            hybrid_answer = res_hybrid.get("answer", "")
            if "99.92" not in hybrid_answer:
                print(f"FAILED: Hybrid answer missing official 99.92% SLA: {hybrid_answer[:200]}")
                failures += 1
            else:
                print("PASSED: Grounded research answer produced with persisted user context.")

        finally:
            # Clean up injection
            set_default_memory_store(None)

    banner("STAGE 4 VERIFICATION SUMMARY")
    if failures == 0:
        print("ALL STAGE 4 PERSISTENT SQLITE MEMORY TESTS PASSED!")
        return 0
    else:
        print(f"STAGE 4 VERIFICATION ENCOUNTERED {failures} FAILURES.")
        return 1


if __name__ == "__main__":
    sys.exit(run_stage4_tests())
