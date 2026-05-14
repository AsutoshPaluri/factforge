"""LangGraph wiring: decompose -> retrieve -> nli -> synthesize -> summarize.

The graph is linear for v1. Each node uses a different specialized
capability (Gemini vision, DDG retrieval, DeBERTa NLI, aggregation,
Gemini narrative synthesis).

Usage:
    graph = build_graph()
    final_state = await graph.ainvoke({"claim": "vaccines cause autism"})
    print(final_state["final_verdict"], final_state["summary"])
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from factforge.agent.nodes.decomposer import decomposer_node
from factforge.agent.nodes.nli import nli_node
from factforge.agent.nodes.retriever import retriever_node
from factforge.agent.nodes.summarizer import summarizer_node
from factforge.agent.nodes.synthesizer import synthesizer_node
from factforge.agent.state import AgentState


def build_graph():
    """Build and compile the fact-checking agent graph."""
    builder = StateGraph(AgentState)

    builder.add_node("decompose", decomposer_node)
    builder.add_node("retrieve", retriever_node)
    builder.add_node("nli", nli_node)
    builder.add_node("synthesize", synthesizer_node)
    builder.add_node("summarize", summarizer_node)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "nli")
    builder.add_edge("nli", "synthesize")
    builder.add_edge("synthesize", "summarize")
    builder.add_edge("summarize", END)

    return builder.compile()


# Module-level compiled graph for reuse
_graph = None


def get_graph():
    """Get the process-wide compiled graph (built on first call)."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
