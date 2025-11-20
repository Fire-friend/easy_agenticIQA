"""
LangGraph StateGraph setup for AgenticIQA pipeline.
Orchestrates the Planner→Executor→Summarizer workflow.
"""

import logging
from typing import Optional, Dict, Any, Literal

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from src.agentic.state import AgenticIQAState
from src.agentic.nodes.planner import planner_node
from src.agentic.nodes.executor import executor_node
from src.agentic.nodes.summarizer import summarizer_node

# Configure logging
logger = logging.getLogger(__name__)


def decide_next_node(state: AgenticIQAState) -> Literal["planner", "__end__"]:
    """
    Conditional edge after Summarizer.

    Determines whether replanning is needed based on evidence sufficiency.
    Returns "planner" to trigger replanning or "__end__" to finish.

    Args:
        state: Current AgenticIQA state

    Returns:
        "planner" if replanning needed and iterations < max
        "__end__" otherwise
    """
    summarizer_result = state.get("summarizer_result")
    if not summarizer_result:
        logger.warning("No summarizer_result in state, ending pipeline")
        return "__end__"

    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_replan_iterations", 2)

    logger.info(f"Replanning decision: need_replan={summarizer_result.need_replan}, "
                f"iteration={iteration}/{max_iterations}")

    if summarizer_result.need_replan and iteration < max_iterations:
        logger.info(f"Replanning triggered: {summarizer_result.replan_reason}")
        logger.info(f"Iteration {iteration}/{max_iterations}")
        return "planner"

    if summarizer_result.need_replan and iteration >= max_iterations:
        logger.warning(f"Max replanning iterations ({max_iterations}) reached")
        logger.warning("Continuing with current evidence despite need_replan=True")

    return "__end__"


def create_agentic_graph() -> StateGraph:
    """
    Create the AgenticIQA StateGraph.

    Phase 4: Includes Planner, Executor, and Summarizer nodes with replanning loop

    Returns:
        StateGraph instance
    """
    logger.info("Creating AgenticIQA StateGraph")

    # Initialize graph with state type
    graph = StateGraph(AgenticIQAState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_node("summarizer", summarizer_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Define edges
    graph.add_edge("planner", "executor")
    graph.add_edge("executor", "summarizer")

    # Conditional edge from Summarizer for replanning
    graph.add_conditional_edges(
        "summarizer",
        decide_next_node,
        {
            "planner": "planner",
            "__end__": END
        }
    )

    logger.info("StateGraph created with Planner, Executor, and Summarizer nodes")

    return graph


def compile_graph(
    graph: Optional[StateGraph] = None,
    checkpointer: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Compile the StateGraph for execution.

    Args:
        graph: StateGraph to compile (creates new one if None)
        checkpointer: Optional checkpointer for state persistence
        config: Optional configuration dict

    Returns:
        Compiled graph ready for execution
    """
    if graph is None:
        graph = create_agentic_graph()

    config = config or {}

    # Use MemorySaver if no checkpointer provided
    if checkpointer is None and config.get("enable_checkpointing", False):
        checkpointer = MemorySaver()
        logger.info("Using MemorySaver for checkpointing")

    # Compile graph
    if checkpointer:
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()

    logger.info("StateGraph compiled successfully")

    return compiled


def run_pipeline(
    query: str,
    image_path: str,
    reference_path: Optional[str] = None,
    max_replan_iterations: int = 2,
    config: Optional[Dict[str, Any]] = None
) -> AgenticIQAState:
    """
    Run the AgenticIQA pipeline end-to-end.

    Args:
        query: User's natural language query
        image_path: Path to image to assess
        reference_path: Optional path to reference image
        max_replan_iterations: Maximum replanning iterations allowed (default=2)
        config: Optional configuration override

    Returns:
        Final state after pipeline execution
    """
    logger.info(f"Running pipeline: query='{query}', image='{image_path}'")

    # Create initial state with iteration tracking
    initial_state: AgenticIQAState = {
        "query": query,
        "image_path": image_path,
        "iteration_count": 0,
        "max_replan_iterations": max_replan_iterations,
        "replan_history": []
    }

    if reference_path:
        initial_state["reference_path"] = reference_path

    logger.info(f"Max replanning iterations: {max_replan_iterations}")

    # Create and compile graph
    compiled_graph = compile_graph(config=config)

    # Execute graph
    try:
        final_state = compiled_graph.invoke(initial_state)
        logger.info("Pipeline execution completed successfully")

        # Log replanning history if any
        if final_state.get("replan_history"):
            logger.info(f"Replanning occurred {len(final_state['replan_history'])} time(s)")
            for entry in final_state["replan_history"]:
                logger.info(f"  {entry}")

        return final_state

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


def visualize_graph(graph: Optional[StateGraph] = None, output_path: Optional[str] = None) -> str:
    """
    Generate Mermaid diagram for graph visualization.

    Args:
        graph: StateGraph to visualize (creates new one if None)
        output_path: Optional file path to save diagram

    Returns:
        Mermaid diagram string
    """
    if graph is None:
        graph = create_agentic_graph()

    # Generate Mermaid diagram with Phase 4 replanning loop
    mermaid = """graph TD
    START([START]) --> planner[Planner]
    planner --> executor[Executor]
    executor --> summarizer[Summarizer]
    summarizer -->|need_replan=True & iter<max| planner
    summarizer -->|need_replan=False or iter>=max| END([END])

    style START fill:#e1f5e1
    style END fill:#ffe1e1
    style planner fill:#e3f2fd
    style executor fill:#fff3e0
    style summarizer fill:#f3e5f5
"""

    if output_path:
        with open(output_path, 'w') as f:
            f.write(mermaid)
        logger.info(f"Graph visualization saved to {output_path}")

    return mermaid
