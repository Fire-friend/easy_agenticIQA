# Design: Executor Module Architecture

## Overview
The Executor module bridges VLM reasoning with traditional IQA tool measurements. It decomposes evidence collection into four conditional subtasks orchestrated by the Planner's control flags.

## Architectural Decisions

### 1. Subtask Modularization
**Decision**: Implement four independent subtask functions instead of a monolithic executor.

**Rationale**:
- Each subtask can be tested and debugged independently
- Control flags enable conditional execution based on Planner output
- Supports incremental development and validation
- Aligns with paper's modular design (Figure 2)

**Trade-offs**:
- More functions to maintain vs. single unified executor
- Need careful state management between subtasks
- **Chosen**: Modularity wins for testability and clarity

### 2. Tool Registry Pattern
**Decision**: Centralized tool registry with metadata-driven execution.

**Rationale**:
- Single source of truth for tool capabilities and parameters
- Enables dynamic tool selection based on distortion types
- Supports adding new tools without modifying executor logic
- Facilitates tool output caching by tool name + image hash

**Implementation**:
```python
class ToolRegistry:
    def __init__(self, metadata_path: Path):
        self.tools = self._load_metadata(metadata_path)
        self.cache = {}

    def execute_tool(self, tool_name: str, image_path: str,
                     reference_path: Optional[str] = None) -> float:
        # Check cache, run tool, normalize score
        pass

    def normalize_score(self, tool_name: str, raw_score: float) -> float:
        # Apply logistic function with tool-specific parameters
        pass
```

**Trade-offs**:
- Abstraction overhead vs. direct IQA-PyTorch calls
- **Chosen**: Registry pattern for maintainability and extensibility

### 3. Score Normalization Strategy
**Decision**: Five-parameter logistic function for all tool outputs.

**Formula**:
```
f(x) = (β1 - β2) / (1 + exp(-(x - β3)/|β4|)) + β2
```

**Rationale**:
- Monotonic transformation preserves tool ranking
- Maps heterogeneous ranges to unified [1, 5] scale
- Parameters from paper Appendix A.3 ensure paper-aligned behavior
- Handles both "higher is better" and "lower is better" tools

**Fallback**: Linear scaling if parameters unavailable, with documentation note.

**Trade-offs**:
- Complex parameterization vs. simple linear scaling
- **Chosen**: Logistic for paper alignment, linear as fallback

### 4. Prompt Template Location
**Decision**: Store prompt templates as Python string constants in executor.py.

**Rationale**:
- Same pattern as Planner module (consistency)
- Avoids external file management overhead
- Easy to version control and review alongside code
- Templates from `docs/03_module_executor.md` are stable

**Trade-offs**:
- Inline strings vs. external template files
- **Chosen**: Inline for simplicity (Phase 2 precedent)

### 5. VLM Client Reuse
**Decision**: Reuse VLM client abstraction from Phase 2 for all subtasks.

**Rationale**:
- Consistent API across Planner and Executor
- Configuration loading already implemented
- Supports multiple VLM backends (OpenAI, Anthropic, Google)
- Leverages retry logic and error handling

**Implementation**: Load executor VLM config from `configs/model_backends.yaml`:
```yaml
executor:
  backend: openai.gpt-4o
  temperature: 0.0
```

### 6. State Model Extensions
**Decision**: Add `executor_evidence: NotRequired[ExecutorOutput]` to `AgenticIQAState`.

**Rationale**:
- Maintains type safety with Pydantic
- Optional field allows Planner-only execution (Phase 2 compatibility)
- Structured output supports Summarizer integration (Phase 4)
- Aligns with LangGraph state management patterns

**Structure**:
```python
class ExecutorOutput(BaseModel):
    distortion_set: Optional[Dict[str, List[str]]]
    distortion_analysis: Optional[Dict[str, List[DistortionAnalysis]]]
    selected_tools: Optional[Dict[str, Dict[str, str]]]
    quality_scores: Optional[Dict[str, Dict[str, Tuple[str, float]]]]
    tool_logs: List[ToolExecutionLog] = Field(default_factory=list)
```

### 7. Control Flag Orchestration
**Decision**: Executor node checks each control flag and conditionally calls subtasks.

**Logic Flow**:
```python
def executor_node(state: AgenticIQAState, config: Optional[RunnableConfig] = None):
    plan = state["plan"]
    output = ExecutorOutput()

    if plan.plan.distortion_detection:
        output.distortion_set = distortion_detection_subtask(...)

    if plan.plan.distortion_analysis:
        output.distortion_analysis = distortion_analysis_subtask(...)

    if plan.plan.tool_selection:
        output.selected_tools = tool_selection_subtask(...)

    if plan.plan.tool_execution:
        output.quality_scores = tool_execution_subtask(...)

    return {"executor_evidence": output}
```

**Rationale**:
- Explicit control flow matching Planner's intent
- Each subtask can fail independently without blocking others
- Aligns with paper's conditional execution model

### 8. Caching Strategy
**Decision**: Simple image hash + tool name keyed cache for tool execution.

**Rationale**:
- Avoids redundant tool calls during development/debugging
- Tool outputs are deterministic for same inputs
- Cache key: `hash(image_bytes) + tool_name + hash(reference_bytes)`

**Implementation**:
- In-memory cache for single-run sessions
- Optional disk cache (JSON) for cross-run persistence
- Cache invalidation not needed (deterministic outputs)

**Trade-offs**:
- Memory usage vs. computation time
- **Chosen**: In-memory cache sufficient for Phase 3

### 9. Error Handling Strategy
**Decision**: Three-tier approach mirroring Planner:
1. **Validation**: Pydantic schema validation for all subtask outputs
2. **Retry**: Up to 3 attempts for VLM-based subtasks with stricter prompts
3. **Fallback**: Tool failures → fallback to generic NR tools (BRISQUE, NIQE)

**Logging**: All errors logged with context (subtask, attempt, error type).

**State**: Errors recorded in `ExecutorOutput.tool_logs` with failure flag.

### 10. Tool Metadata Format
**Decision**: JSON file in `iqa_tools/metadata/tools.json` with structure:
```json
{
  "TOPIQ_FR": {
    "type": "FR",
    "strengths": ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"],
    "logistic_params": {"beta1": 5.0, "beta2": 1.0, "beta3": 0.5, "beta4": 0.1}
  },
  "QAlign": {
    "type": "NR",
    "strengths": ["Blurs", "Color distortions", "Noise", "Brightness change", "Spatial distortions", "Sharpness"],
    "logistic_params": {"beta1": 5.0, "beta2": 1.0, "beta3": 3.0, "beta4": 0.5}
  }
}
```

**Rationale**:
- Human-readable and version-controllable
- Easy to extend with new tools
- Logistic parameters embedded for self-contained execution

## Implementation Sequence
1. **ExecutorOutput models** (state.py) - Type safety foundation
2. **Tool registry** (tool_registry.py) - Metadata loading and execution
3. **Subtask functions** (nodes/executor.py) - Four independent subtasks
4. **Executor node** (nodes/executor.py) - Orchestration logic
5. **Graph integration** (graph.py) - Add Executor node to StateGraph
6. **Tests** (tests/test_executor.py, tests/test_tool_registry.py) - Validation

## Validation Approach
- Mock tool outputs for unit tests (deterministic)
- Real tool execution for integration tests (requires weights)
- Example from `docs/03_module_executor.md` as regression test
- Verify Planner→Executor→state correctness
