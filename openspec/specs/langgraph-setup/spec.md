# langgraph-setup Specification

## Purpose
TBD - created by archiving change implement-phase2-planner-module. Update Purpose after archive.
## Requirements
### Requirement: StateGraph Initialization
The system SHALL create and configure the LangGraph StateGraph with AgenticIQAState.

#### Scenario: Graph creation with typed state
- **Given** the `AgenticIQAState` TypedDict definition
- **When** initializing the StateGraph
- **Then** it creates a `StateGraph(AgenticIQAState)` instance
- **And** validates that state keys match TypedDict fields
- **And** prepares for node registration

#### Scenario: Graph naming and metadata
- **Given** the need to identify the graph instance
- **When** creating the StateGraph
- **Then** it includes a descriptive name: "AgenticIQA-Pipeline"
- **And** optionally stores metadata (version, creation timestamp)

### Requirement: Planner Node Registration
The system SHALL register the Planner node function with the StateGraph.

#### Scenario: Planner node addition
- **Given** a compiled Planner node function `planner_node(state: AgenticIQAState) -> AgenticIQAState`
- **When** adding it to the StateGraph
- **Then** it calls `graph.add_node("planner", planner_node)`
- **And** the node is registered with unique ID "planner"

#### Scenario: Planner node function signature
- **Given** the Planner node function
- **When** defining its signature
- **Then** it accepts `state: AgenticIQAState` as input
- **And** returns a dict with state updates (partial `AgenticIQAState`)
- **And** uses type hints for IDE/tooling support

#### Scenario: Planner node execution
- **Given** a state with `query`, `image_path`, and optionally `reference_path`
- **When** the Planner node executes
- **Then** it updates the state with `plan: PlannerOutput`
- **And** optionally updates `error` field if an error occurs
- **And** returns the updated state dict

### Requirement: Graph Entry and Exit Points
The system SHALL configure the graph entry point and conditional termination logic.

#### Scenario: Entry point definition
- **Given** the StateGraph with Planner node
- **When** defining the entry point
- **Then** it calls `graph.set_entry_point("planner")`
- **And** the graph starts execution at the Planner node

#### Scenario: Exit point definition (Phase 2 simple termination)
- **Given** Phase 2 only implements the Planner
- **When** the Planner node completes
- **Then** the graph terminates (no subsequent nodes yet)
- **And** the final state includes the generated plan
- **Note**: Full pipeline with Executor→Summarizer added in Phase 3

#### Scenario: Conditional edges placeholder
- **Given** future need for replanning logic (Summarizer → Planner)
- **When** documenting the graph structure
- **Then** include comments indicating future conditional edges:
  ```python
  # Phase 3+: Add conditional edge from Summarizer
  # graph.add_conditional_edges(
  #     "summarizer",
  #     should_replan,
  #     {True: "planner", False: END}
  # )
  ```

### Requirement: Graph Compilation and Execution
The system SHALL compile the StateGraph and provide an execution interface.

#### Scenario: Graph compilation
- **Given** a configured StateGraph with nodes and edges
- **When** calling `graph.compile()`
- **Then** it returns a compiled `CompiledGraph` instance
- **And** validates that all nodes are reachable
- **And** checks for cycles (allowed with conditional edges)

#### Scenario: Graph invocation
- **Given** a compiled graph and initial state
- **When** calling `compiled_graph.invoke(initial_state)`
- **Then** it executes the graph from the entry point
- **And** returns the final state after all nodes complete
- **And** raises exceptions if nodes fail (unless error handling configured)

#### Scenario: Streaming execution
- **Given** the need to observe intermediate states
- **When** using `compiled_graph.stream(initial_state)`
- **Then** it yields state updates after each node execution
- **And** allows for progress monitoring or early termination

### Requirement: State Persistence and Checkpointing
The system SHALL configure checkpointing for long-running pipeline executions.

#### Scenario: Checkpointer configuration
- **Given** the pipeline may process large batches
- **When** configuring the StateGraph
- **Then** optionally enable checkpointing via `MemorySaver` or persistent backend
- **And** save state after each node completion
- **And** allow resumption from checkpoint on failure

#### Scenario: Checkpoint storage location
- **Given** checkpointing is enabled
- **When** specifying storage location
- **Then** use `configs/pipeline.yaml` setting: `pipeline.checkpoint.save_dir`
- **And** create checkpoint files with format: `{timestamp}_{sample_id}.json`

#### Scenario: Resume from checkpoint
- **Given** a previous execution failed mid-pipeline
- **When** resuming with the same input
- **Then** load the last successful checkpoint
- **And** skip already-completed nodes
- **And** continue from the failed node

### Requirement: Graph Visualization and Debugging
The system SHALL provide tools for visualizing and debugging the graph structure.

#### Scenario: Graph structure visualization
- **Given** the need to understand the pipeline flow
- **When** requesting graph visualization
- **Then** the graph can be exported to Mermaid diagram format
- **And** optionally saved to a file: `logs/graph_structure.mmd`
- **Example output**:
  ```mermaid
  graph TD
      START --> planner
      planner --> END
  ```

#### Scenario: Debug mode execution
- **Given** development or troubleshooting scenario
- **When** enabling `debug_mode: true` in `configs/pipeline.yaml`
- **Then** the graph logs verbose details for each node:
  - Input state snapshot
  - Node execution duration
  - Output state changes
  - Any warnings or errors
- **And** writes debug logs to `logs/graph_debug.log`

### Requirement: Error Propagation and Handling
The system SHALL define how errors are propagated through the graph.

#### Scenario: Node exception handling
- **Given** a node raises an exception during execution
- **When** the error occurs
- **Then** by default, the graph halts and re-raises the exception
- **And** the error message includes node name and input state

#### Scenario: Graceful error handling mode
- **Given** `graceful_errors: true` configuration
- **When** a node raises an exception
- **Then** the graph captures the error in state `error` field
- **And** optionally continues to next node (or terminates)
- **And** logs the error with full traceback

#### Scenario: Retry logic integration
- **Given** transient errors (e.g., API rate limits)
- **When** a node fails
- **Then** the graph automatically retries up to `max_retry_attempts` times
- **And** uses exponential backoff between retries
- **And** logs each retry attempt

### Requirement: Graph Configuration Loading
The system SHALL load graph-specific settings from configuration files.

#### Scenario: Graph settings from YAML
- **Given** `configs/pipeline.yaml` contains `langgraph` section
- **When** initializing the StateGraph
- **Then** load settings:
  - `max_iterations: int` (prevent infinite loops)
  - `recursion_limit: int` (LangGraph recursion depth)
  - `debug_mode: bool`
  - `save_graph_visualization: bool`
- **And** apply these settings to the compiled graph

#### Scenario: Per-node timeout configuration
- **Given** nodes may have different time complexity
- **When** configuring node timeouts
- **Then** load from `pipeline.timeout` section:
  - `planner: 60` seconds
  - `executor: 300` seconds (future)
  - `summarizer: 60` seconds (future)
- **And** enforce timeouts during node execution
- **And** raise `TimeoutError` if exceeded

