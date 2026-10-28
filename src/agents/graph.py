"""
src/agents/graph.py - Asterline Research Assistant
LangGraph Multi-Agent StateGraph Definition and Routing.

Assembles Planner, Memory Manager, and Researcher into a coordinated StateGraph.
"""

from typing import Any, Dict, Literal, Optional

from langgraph.graph import END, START, StateGraph

from src.agents.memory_agent import memory_manager_node
from src.agents.planner import planner_node
from src.agents.researcher_agent import researcher_node
from src.agents.state import AgentState, create_initial_state


def route_intent(state: AgentState) -> Literal["researcher", "memory_manager"]:
    """
    Conditional routing function from the Planner node.
    
    Routing rules:
      - 'research': Go directly to researcher.
      - 'memory': Go to memory_manager.
      - 'hybrid': Go to memory_manager first, then memory_manager will route to researcher.
    """
    intent = state.get("intent", "research")
    if intent in ("memory", "hybrid"):
        return "memory_manager"
    return "researcher"


def route_after_memory(state: AgentState) -> Literal["researcher", "__end__"]:
    """
    Conditional routing function after the Memory Manager node.
    
    Routing rules:
      - 'hybrid': Continue to researcher with populated user_memories.
      - 'memory': Pure memory action complete; transition to END.
    """
    intent = state.get("intent", "memory")
    if intent == "hybrid":
        return "researcher"
    return END


def create_research_graph():
    """
    Builds and compiles the Asterline Multi-Agent StateGraph.
    
    Graph Topology:
      START
        │
        ▼
      planner ────────(intent == 'research')────────► researcher ──► END
        │                                                  ▲
        │ (intent in ['memory', 'hybrid'])                 │
        ▼                                                  │
      memory_manager ───────(intent == 'hybrid')───────────┘
        │
        └──────────────(intent == 'memory')────────────────────────► END
    """
    builder = StateGraph(AgentState)

    # Add the three core agent nodes
    builder.add_node("planner", planner_node)
    builder.add_node("memory_manager", memory_manager_node)
    builder.add_node("researcher", researcher_node)

    # Edge from START to Planner
    builder.add_edge(START, "planner")

    # Conditional routing from Planner
    builder.add_conditional_edges(
        "planner",
        route_intent,
        {
            "researcher": "researcher",
            "memory_manager": "memory_manager",
        },
    )

    # Conditional routing from Memory Manager
    builder.add_conditional_edges(
        "memory_manager",
        route_after_memory,
        {
            "researcher": "researcher",
            END: END,
        },
    )

    # Terminal edge from Researcher to END
    builder.add_edge("researcher", END)

    return builder.compile()


# Lazy-loaded compiled graph singleton
_COMPILED_GRAPH = None


def get_graph():
    """Returns the compiled LangGraph singleton."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_research_graph()
    return _COMPILED_GRAPH


def run_agent(
    query: str,
    user_id: str = "default_user",
    user_memories: Optional[Dict[str, str]] = None,
) -> AgentState:
    """
    Convenience entrypoint to execute the multi-agent graph on a query.
    
    Args:
      query: The user's input prompt or command.
      user_id: User identifier for personal memory scoping.
      user_memories: Optional initial user context.
      
    Returns:
      The final populated AgentState after all routed nodes complete.
    """
    graph = get_graph()
    initial_state = create_initial_state(
        query=query,
        user_id=user_id,
        user_memories=user_memories,
    )
    result = graph.invoke(initial_state)
    return result
