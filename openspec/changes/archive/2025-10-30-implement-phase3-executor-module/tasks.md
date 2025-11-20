# Implementation Tasks: Phase 3 - Executor Module

## Prerequisites
- [x] Phase 2 (Planner module) completed and archived ✓
- [x] IQA-PyTorch installed and accessible
- [x] Tool weights downloaded (if needed for real tool execution)
- [x] Test images prepared from `docs/03_module_executor.md` examples

## Task 1: Executor State Models
**Owner**: Implementation team
**Estimated effort**: 2-3 hours
**Dependencies**: None (extends existing state.py)

### Subtasks
- [x] Add `DistortionAnalysis` model with type, severity, explanation fields
- [x] Add `ToolExecutionLog` model for tool call recording
- [x] Add `ExecutorOutput` model with all four subtask output fields
- [x] Extend `AgenticIQAState` TypedDict with `executor_evidence: NotRequired[ExecutorOutput]`
- [x] Add field validators for distortion types (limited to valid categories)
- [x] Add field validators for severity levels (none/slight/moderate/severe/extreme)

### Validation
- [x] Unit tests for each new model with valid/invalid inputs
- [x] Test JSON serialization/deserialization round-trip
- [x] Verify state merging works with new executor_evidence field

**Files**:
- `src/agentic/state.py` - Add models

## Task 2: Tool Registry Implementation
**Owner**: Implementation team
**Estimated effort**: 3-4 hours
**Dependencies**: Task 1 (ExecutorOutput models)

### Subtasks
- [x] Create `src/agentic/tool_registry.py` module
- [x] Implement `ToolRegistry` class with metadata loading
- [x] Create `iqa_tools/metadata/tools.json` with tool definitions
- [x] Implement `execute_tool()` method with IQA-PyTorch integration
- [x] Implement `normalize_score()` with five-parameter logistic function
- [x] Add image hash-based caching for tool outputs
- [x] Add logistic parameter defaults for fallback

### Validation
- [x] Unit tests for metadata loading
- [x] Unit tests for logistic normalization with known inputs
- [x] Integration tests with mocked IQA-PyTorch calls
- [x] Test caching behavior (cache hits/misses)

**Files**:
- `src/agentic/tool_registry.py` - New file
- `iqa_tools/metadata/tools.json` - Tool metadata

## Task 3: Distortion Detection Subtask
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Task 1, Task 2

### Subtasks
- [x] Define `DISTORTION_DETECTION_PROMPT_TEMPLATE` with template from docs
- [x] Implement `distortion_detection_subtask()` function
- [x] Add JSON parsing and validation for distortion_set output
- [x] Implement retry logic (up to 3 attempts) for invalid JSON
- [x] Add distortion type validation against valid categories list

### Validation
- [x] Unit tests with mocked VLM responses
- [x] Test with valid and invalid JSON outputs
- [x] Verify distortion types are limited to valid categories
- [x] Test with query_scope alignment

**Files**:
- `src/agentic/nodes/executor.py` - New file (subtask function)

## Task 4: Distortion Analysis Subtask
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Task 1, Task 3

### Subtasks
- [x] Define `DISTORTION_ANALYSIS_PROMPT_TEMPLATE` with template from docs
- [x] Implement `distortion_analysis_subtask()` function
- [x] Add JSON parsing and validation for distortion_analysis output
- [x] Implement retry logic for invalid JSON
- [x] Add severity level validation (none/slight/moderate/severe/extreme)

### Validation
- [x] Unit tests with mocked VLM responses
- [x] Test severity level validation
- [x] Verify explanation text is present and non-empty
- [x] Test with distortion_set input from Task 3

**Files**:
- `src/agentic/nodes/executor.py` - Add subtask function

## Task 5: Tool Selection Subtask
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Task 1, Task 2

### Subtasks
- [x] Define `TOOL_SELECTION_PROMPT_TEMPLATE` with template from docs
- [x] Implement `tool_selection_subtask()` function
- [x] Add tool metadata formatting for VLM prompt
- [x] Add JSON parsing and validation for selected_tools output
- [x] Implement FR/NR matching logic (prioritize FR tools when reference_mode="Full-Reference")
- [x] Implement retry logic for invalid JSON

### Validation
- [x] Unit tests with mocked VLM responses
- [x] Test FR/NR prioritization logic
- [x] Verify tool names match metadata
- [x] Test with distortion_set from Task 3

**Files**:
- `src/agentic/nodes/executor.py` - Add subtask function

## Task 6: Tool Execution Subtask
**Owner**: Implementation team
**Estimated effort**: 3 hours
**Dependencies**: Task 2, Task 5

### Subtasks
- [x] Implement `tool_execution_subtask()` function
- [x] Add image loading and preprocessing for tools
- [x] Integrate with ToolRegistry.execute_tool()
- [x] Add error handling for tool failures (fallback to BRISQUE/NIQE)
- [x] Record tool execution logs with timing and scores
- [x] Handle NaN/Inf outputs from tools

### Validation
- [x] Unit tests with mocked tool registry
- [x] Integration tests with real IQA-PyTorch tools
- [x] Test fallback behavior on tool failures
- [x] Verify score normalization to [1, 5] range

**Files**:
- `src/agentic/nodes/executor.py` - Add subtask function

## Task 7: Executor Node Orchestration
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Tasks 3, 4, 5, 6

### Subtasks
- [x] Implement `executor_node()` LangGraph node function
- [x] Add control flag checking logic for conditional subtask execution
- [x] Add configuration loading from `configs/model_backends.yaml`
- [x] Add VLM client creation (reuse from Planner)
- [x] Add error handling for subtask failures
- [x] Implement logging for all subtask executions

### Validation
- [x] Unit tests for control flag logic
- [x] Test with all control flags enabled
- [x] Test with partial control flags (only some enabled)
- [x] Verify error handling doesn't block independent subtasks

**Files**:
- `src/agentic/nodes/executor.py` - Add orchestration function

## Task 8: Graph Integration
**Owner**: Implementation team
**Estimated effort**: 1-2 hours
**Dependencies**: Task 7

### Subtasks
- [x] Add Executor node to `create_agentic_graph()` in graph.py
- [x] Add edge from Planner to Executor
- [x] Update graph visualization (Mermaid diagram)
- [x] Update `run_pipeline()` to handle executor_evidence in state
- [x] For Phase 3, add temporary edge from Executor to END (until Summarizer exists)

### Validation
- [x] Test graph creation with Executor node
- [x] Test Planner→Executor flow with mocked responses
- [x] Verify state contains both plan and executor_evidence
- [x] Test graph compilation succeeds

**Files**:
- `src/agentic/graph.py` - Update graph definition

## Task 9: Unit Tests
**Owner**: Implementation team
**Estimated effort**: 3-4 hours
**Dependencies**: Tasks 1-8

### Test Coverage
- [x] `tests/test_executor_state.py` - Executor state models
- [x] `tests/test_tool_registry.py` - Tool registry and normalization
- [x] `tests/test_executor_subtasks.py` - Four subtask functions (integrated into test_executor_integration.py)
- [x] `tests/test_executor_node.py` - Orchestration logic (integrated into test_executor_integration.py)
- [x] `tests/test_executor_integration.py` - Planner→Executor flow

### Test Data
- [x] Create fixture examples from `docs/03_module_executor.md`
- [x] Mock VLM responses for each subtask
- [x] Mock tool outputs with known normalization values
- [x] Create test images (or use placeholders)

**Files**:
- `tests/test_executor_state.py` - New file
- `tests/test_tool_registry.py` - New file
- `tests/test_executor_subtasks.py` - New file
- `tests/test_executor_node.py` - New file
- `tests/test_executor_integration.py` - New file

## Task 10: Documentation and Examples
**Owner**: Implementation team
**Estimated effort**: 1-2 hours
**Dependencies**: Tasks 1-9

### Deliverables
- [x] Update IMPLEMENTATION_SUMMARY.md with Phase 3 status
- [x] Document tool metadata format in iqa_tools/metadata/README (documented in IMPLEMENTATION_SUMMARY.md)
- [x] Add example usage in docstrings
- [x] Update graph visualization Mermaid diagram

**Files**:
- `IMPLEMENTATION_SUMMARY.md` - Update summary

## Completion Checklist
- [x] All 10 tasks completed
- [x] All unit tests passing (99 passed, 11 skipped - target: >80% coverage for new code)
- [x] No Pydantic or LangGraph warnings
- [x] Integration test with Planner→Executor succeeds
- [x] Tool registry successfully loads metadata and executes tools
- [x] Example from `docs/03_module_executor.md` produces expected output
- [x] Code follows project conventions (minimal files, reuse existing patterns)
- [x] Documentation updated

## Validation Commands

### Run all Executor tests
```bash
pytest tests/test_executor*.py tests/test_tool_registry.py -v
```

### Run integration test
```bash
pytest tests/test_executor_integration.py -v
```

### Run full test suite
```bash
pytest tests/ -v
```

### Check test coverage
```bash
pytest tests/ --cov=src/agentic --cov-report=html
```

### Validate environment
```bash
python scripts/check_env.py
```

## Notes
- Tasks can be parallelized: Task 3, 4, 5, 6 are independent after Task 1 and 2
- Tool execution (Task 6) requires IQA-PyTorch; use mocks if weights unavailable
- Executor node (Task 7) ties everything together
- Graph integration (Task 8) enables end-to-end testing
- Test suite (Task 9) ensures correctness and regression prevention
