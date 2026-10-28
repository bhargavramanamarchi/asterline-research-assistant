"""
src/agents/memory_agent.py - Asterline Research Assistant
Memory Manager Agent and In-Memory Interface.

Handles personal memory requests (store, recall, forget, clear).
STAGE 3 NOTE: Uses an in-memory interface. Persistent SQLite storage is Stage 4.

CRITICAL INVARIANT:
Personal memories represent user context/preferences ONLY.
Personal memories NEVER override official Asterline documentation or company policies.
"""

from typing import Any, Dict, Optional

from src.agents.state import AgentState
from src.memory import SQLiteMemoryStore


class InMemoryMemoryStore:
    """
    Temporary thread-safe in-memory store for user personal context and preferences.
    Retained for lightweight testing and backward compatibility.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, Dict[str, str]] = {}

    def get_memories(self, user_id: str) -> Dict[str, str]:
        """Retrieves a copy of all memories for a user."""
        return dict(self._storage.get(user_id, {}))

    def get_memory(self, user_id: str, key: str) -> Optional[str]:
        """Retrieves a single memory value by key (case-insensitive key lookup)."""
        user_dict = self._storage.get(user_id, {})
        for k, v in user_dict.items():
            if k.lower() == key.lower():
                return v
        return None

    def set_memory(self, user_id: str, key: str, value: str) -> None:
        """Stores or updates a memory key-value pair for a user."""
        if user_id not in self._storage:
            self._storage[user_id] = {}
        clean_key = " ".join(key.strip().split())
        self._storage[user_id][clean_key] = value.strip()

    def delete_memory(self, user_id: str, key: str) -> bool:
        """Deletes a specific memory key. Returns True if deleted, False if not found."""
        if user_id in self._storage:
            clean_key = " ".join(key.strip().split())
            to_remove = None
            for k in self._storage[user_id]:
                if k.lower() == clean_key.lower():
                    to_remove = k
                    break
            if to_remove:
                del self._storage[user_id][to_remove]
                return True
        return False

    def clear_memories(self, user_id: str) -> None:
        """Clears all memories for a specific user."""
        if user_id in self._storage:
            self._storage[user_id].clear()

    def reset_all(self) -> None:
        """Clears all storage across all users (used for test isolation)."""
        self._storage.clear()


# Default persistent SQLite store instance
_DEFAULT_SQLITE_STORE: Optional[SQLiteMemoryStore] = None
_INJECTED_STORE: Optional[Any] = None


def get_default_memory_store() -> Any:
    """Returns the active memory store (injected test store or default SQLite store)."""
    global _DEFAULT_SQLITE_STORE, _INJECTED_STORE
    if _INJECTED_STORE is not None:
        return _INJECTED_STORE
    if _DEFAULT_SQLITE_STORE is None:
        _DEFAULT_SQLITE_STORE = SQLiteMemoryStore()
    return _DEFAULT_SQLITE_STORE


def set_default_memory_store(store: Optional[Any]) -> None:
    """Allows injecting an alternate memory store (e.g. a temporary test SQLite database)."""
    global _INJECTED_STORE
    _INJECTED_STORE = store


# Backward compatibility reference for Stage 3 imports
DEFAULT_MEMORY_STORE = SQLiteMemoryStore()


def memory_manager_node(
    state: AgentState,
    store: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Memory Manager Node implementation for LangGraph.
    
    Responsibilities:
      1. Inspects memory_action ('store', 'recall', 'forget', 'clear', 'none').
      2. Reads/writes to the persistent SQLite store for state['user_id'].
      3. For pure 'memory' intent: sets final state['answer'] with confirmation/details.
      4. For 'hybrid' intent: prepares state['user_memories'] and hands off to Researcher.
      5. Appends 'memory_manager' to state['execution_trace'].
    """
    memory_store = store or get_default_memory_store()
    user_id = state.get("user_id", "default_user")
    intent = state.get("intent", "memory")
    action = state.get("memory_action") or "none"
    key = state.get("memory_key")
    val = state.get("memory_value")

    # Sync state memories with existing store memories
    existing_store_memories = memory_store.get_memories(user_id)
    # If state already had memories passed in, merge them into the store
    for k, v in state.get("user_memories", {}).items():
        if k not in existing_store_memories:
            memory_store.set_memory(user_id, k, v)

    answer_text = ""

    if action == "store" and key and val:
        # Save new memory
        memory_store.set_memory(user_id, key, val)
        if intent == "memory":
            answer_text = f"I have remembered that your {key} is {val}."

    elif action == "forget" and key:
        deleted = memory_store.delete_memory(user_id, key)
        if intent == "memory":
            if deleted:
                answer_text = f"I have forgotten your stored memory for '{key}'."
            else:
                answer_text = f"No stored memory was found for '{key}'."

    elif action == "clear":
        memory_store.clear_memories(user_id)
        if intent == "memory":
            answer_text = "All of your personal memories have been cleared."

    elif action == "recall" or intent == "memory":
        current = memory_store.get_memories(user_id)
        if key:
            found_val = memory_store.get_memory(user_id, key)
            if found_val:
                answer_text = f"Your {key} is {found_val}."
            else:
                answer_text = f"I do not have a recorded memory for your {key}."
        else:
            if current:
                items_str = ", ".join(f"{k}: {v}" for k, v in current.items())
                answer_text = f"Here are your stored personal preferences: {items_str}."
            else:
                answer_text = "You currently have no saved personal memories or preferences."

    # Refresh current memories after action
    current_memories = memory_store.get_memories(user_id)

    trace = list(state.get("execution_trace", []))
    trace.append("memory_manager")

    result: Dict[str, Any] = {
        "user_memories": current_memories,
        "execution_trace": trace,
    }

    # Only set answer for pure memory requests; hybrid requests will have the
    # final answer synthesized by the Researcher Agent.
    if intent == "memory":
        result["answer"] = answer_text
        result["status"] = "success"
        result["citations"] = []
        result["citations_valid"] = True

    return result
