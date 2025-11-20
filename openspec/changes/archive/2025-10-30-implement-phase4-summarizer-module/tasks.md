# Implementation Tasks: Phase 4 - Summarizer Module

## Prerequisites
- [x] Phase 2 (Planner module) completed and archived ✓
- [x] Phase 3 (Executor module) completed and archived ✓
- [ ] Review `docs/04_module_summarizer.md` for specifications
- [ ] Test images and expected outputs prepared

## Task 1: Summarizer State Models
**Owner**: Implementation team
**Estimated effort**: 2-3 hours
**Dependencies**: None (extends existing state.py)

### Subtasks
- [ ] Add `SummarizerOutput` model with final_answer, quality_reasoning, need_replan fields
- [ ] Add field validators for quality_reasoning (non-empty) and final_answer (stripped)
- [ ] Add optional fields: replan_reason, used_evidence
- [ ] Extend `AgenticIQAState` TypedDict with:
  - `summarizer_result: NotRequired[SummarizerOutput]`
  - `iteration_count: NotRequired[int]`
  - `max_replan_iterations: NotRequired[int]`
  - `replan_history: NotRequired[List[str]]`
- [ ] Add JSON schema examples to model_config

### Validation
- [ ] Unit tests for SummarizerOutput with valid/invalid inputs
- [ ] Test JSON serialization/deserialization round-trip
- [ ] Test field validators (empty reasoning, whitespace trimming)
- [ ] Verify state merging with new fields

**Files**:
- `src/agentic/state.py` - Add models

## Task 2: Score Fusion Utility
**Owner**: Implementation team
**Estimated effort**: 3-4 hours
**Dependencies**: Task 1 (SummarizerOutput models)

### Subtasks
- [ ] Create `src/agentic/score_fusion.py` module
- [ ] Implement `ScoreFusion` class with configurable eta and quality_levels
- [ ] Implement `compute_perceptual_weights(tool_scores)` - Gaussian weights
- [ ] Implement `extract_vlm_probabilities(vlm_output, mode)` - logits/classification/uniform
- [ ] Implement `fuse_scores(tool_scores, vlm_probs)` - weighted fusion formula
- [ ] Implement `map_to_level(score)` - continuous to discrete mapping
- [ ] Implement `map_to_letter(level)` - level to letter grade (A-E)
- [ ] Add numerical stability handling (subtract max before exp)

### Validation
- [ ] Unit tests for perceptual weight calculation with known inputs
- [ ] Unit tests for VLM probability extraction (all 3 modes)
- [ ] Unit tests for fusion formula with manual calculations
- [ ] Test mapping functions (score→level→letter)
- [ ] Test edge cases: empty tool scores, single score, numerical stability
- [ ] Validate against test cases from docs/04_module_summarizer.md

**Files**:
- `src/agentic/score_fusion.py` - New file

## Task 3: Explanation/QA Mode Prompt
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Task 1

### Subtasks
- [ ] Define `EXPLANATION_PROMPT_TEMPLATE` constant with exact template from docs
- [ ] Implement `format_evidence_for_explanation(executor_output)` function
- [ ] Format distortion_analysis as JSON for prompt
- [ ] Format quality_scores as JSON for prompt
- [ ] Implement `render_explanation_prompt(query, evidence, images)` function

### Validation
- [ ] Unit tests with mocked Executor outputs
- [ ] Verify prompt includes all evidence sections
- [ ] Test with missing evidence (graceful handling)
- [ ] Compare rendered prompt with docs example

**Files**:
- `src/agentic/nodes/summarizer.py` - New file (prompt template)

## Task 4: Scoring Mode Prompt
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Task 1, Task 2

### Subtasks
- [ ] Define `SCORING_PROMPT_TEMPLATE` constant with exact template from docs
- [ ] Implement `format_evidence_for_scoring(executor_output)` function
- [ ] Integrate score fusion: call ScoreFusion utility
- [ ] Format fusion results for VLM prompt (optional guidance)
- [ ] Implement `render_scoring_prompt(query, evidence, fusion_score, images)` function

### Validation
- [ ] Unit tests with mocked Executor outputs and tool scores
- [ ] Verify score fusion is called correctly
- [ ] Test with missing tool scores (fallback to uniform weights)
- [ ] Compare rendered prompt with docs example

**Files**:
- `src/agentic/nodes/summarizer.py` - Add scoring prompt function

## Task 5: Replanning Decision Logic
**Owner**: Implementation team
**Estimated effort**: 2-3 hours
**Dependencies**: Task 1

### Subtasks
- [ ] Implement `check_evidence_sufficiency(executor_output, query_scope)` function
- [ ] Check distortion_analysis coverage for all query_scope objects
- [ ] Check quality_scores availability for key distortions
- [ ] Detect contradictory evidence (severity vs scores)
- [ ] Return (need_replan: bool, reason: str) tuple
- [ ] Implement max iteration check (respect max_replan_iterations)

### Validation
- [ ] Unit tests for evidence sufficiency checks
- [ ] Test with complete evidence (need_replan=false)
- [ ] Test with missing evidence (need_replan=true, appropriate reason)
- [ ] Test with contradictory evidence
- [ ] Test max iteration enforcement

**Files**:
- `src/agentic/nodes/summarizer.py` - Add replanning logic

## Task 6: Summarizer Node Implementation
**Owner**: Implementation team
**Estimated effort**: 4-5 hours
**Dependencies**: Tasks 1, 2, 3, 4, 5

### Subtasks
- [ ] Implement `summarizer_node(state, config)` LangGraph node function
- [ ] Load VLM client from `configs/model_backends.yaml` (summarizer section)
- [ ] Load images from state (test image + optional reference)
- [ ] Select prompt mode based on plan.query_type (IQA → scoring, else → explanation)
- [ ] Render appropriate prompt with evidence formatting
- [ ] Call VLM client with images and prompt
- [ ] Parse VLM JSON response
- [ ] Validate with SummarizerOutput Pydantic model
- [ ] Implement retry logic (up to 3 attempts) for invalid JSON
- [ ] Check evidence sufficiency and determine need_replan
- [ ] Increment iteration_count if replanning
- [ ] Append to replan_history if replanning
- [ ] Return state update with summarizer_result

### Validation
- [ ] Unit tests with mocked VLM responses
- [ ] Test both prompt modes (explanation/QA and scoring)
- [ ] Test retry logic with invalid JSON
- [ ] Test replanning logic (sufficient vs insufficient evidence)
- [ ] Test iteration tracking
- [ ] Test error handling (missing plan, missing executor_evidence, image load failures)

**Files**:
- `src/agentic/nodes/summarizer.py` - Add orchestration function

## Task 7: Graph Updates for Replanning
**Owner**: Implementation team
**Estimated effort**: 2-3 hours
**Dependencies**: Task 6

### Subtasks
- [ ] Import summarizer_node in `src/agentic/graph.py`
- [ ] Add Summarizer node: `graph.add_node("summarizer", summarizer_node)`
- [ ] Update edge: `graph.add_edge("executor", "summarizer")`
- [ ] Remove Phase 3 temporary edge: `graph.add_edge("executor", END)`
- [ ] Implement `decide_next_node(state)` conditional edge function
- [ ] Check need_replan and iteration_count in conditional function
- [ ] Add conditional edge from Summarizer:
```python
graph.add_conditional_edges(
    "summarizer",
    decide_next_node,
    {"planner": "planner", "__end__": END}
)
```
- [ ] Initialize iteration_count and max_replan_iterations in run_pipeline

### Validation
- [ ] Test graph creation with Summarizer node
- [ ] Test conditional edge function with various states
- [ ] Test replanning flow: Planner→Executor→Summarizer→Planner→Executor→Summarizer→END
- [ ] Test no-replan flow: Planner→Executor→Summarizer→END
- [ ] Test max iteration enforcement
- [ ] Test graph compilation succeeds

**Files**:
- `src/agentic/graph.py` - Update graph definition

## Task 8: Configuration Updates
**Owner**: Implementation team
**Estimated effort**: 1 hour
**Dependencies**: Task 6

### Subtasks
- [ ] Add summarizer section to `configs/model_backends.yaml`:
```yaml
summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 512
```
- [ ] Document max_replan_iterations parameter in `configs/pipeline.yaml`
- [ ] Add default values for iteration tracking

### Validation
- [ ] Test config loading in summarizer_node
- [ ] Verify default values are applied correctly

**Files**:
- `configs/model_backends.yaml` - Add summarizer config
- `configs/pipeline.yaml` - Document max_replan_iterations

## Task 9: Unit Tests
**Owner**: Implementation team
**Estimated effort**: 4-5 hours
**Dependencies**: Tasks 1-8

### Test Coverage
- [ ] `tests/test_summarizer_state.py` - Summarizer state models (20 tests)
  - SummarizerOutput validation
  - Field validators
  - JSON serialization
  - State extensions

- [ ] `tests/test_score_fusion.py` - Score fusion algorithm (20 tests)
  - Perceptual weight calculation
  - VLM probability extraction (all modes)
  - Fusion formula
  - Mapping functions
  - Edge cases and numerical stability

- [ ] `tests/test_summarizer_node.py` - Summarizer node logic (15 tests)
  - Prompt mode selection
  - Evidence formatting
  - VLM interaction and retry
  - Replanning decision
  - Iteration tracking

- [ ] `tests/test_replanning_logic.py` - Replanning flow (10 tests)
  - Conditional edge function
  - Iteration limit enforcement
  - State propagation

### Test Data
- [ ] Create fixture examples from `docs/04_module_summarizer.md`
- [ ] Mock VLM responses for both modes
- [ ] Prepare evidence scenarios (complete, incomplete, contradictory)

**Files**:
- `tests/test_summarizer_state.py` - New file
- `tests/test_score_fusion.py` - New file
- `tests/test_summarizer_node.py` - New file
- `tests/test_replanning_logic.py` - New file

## Task 10: Integration Tests
**Owner**: Implementation team
**Estimated effort**: 3-4 hours
**Dependencies**: Task 9

### Test Coverage
- [ ] `tests/test_full_pipeline.py` - End-to-end pipeline tests (10 tests)
  - Planner→Executor→Summarizer→END (no replan)
  - Planner→Executor→Summarizer→Planner→Executor→Summarizer→END (1 replan)
  - Max iteration limit enforcement
  - Both prompt modes (explanation/QA and scoring)
  - Error handling in each node

### Scenarios
- [ ] Test with complete evidence (no replanning)
- [ ] Test with incomplete evidence (trigger replanning)
- [ ] Test with max iterations reached
- [ ] Test scoring mode with score fusion
- [ ] Test explanation/QA mode without fusion

**Files**:
- `tests/test_full_pipeline.py` - New file

## Task 11: Visualization Updates
**Owner**: Implementation team
**Estimated effort**: 1 hour
**Dependencies**: Task 7

### Subtasks
- [ ] Update `visualize_graph()` Mermaid diagram
- [ ] Show Summarizer node
- [ ] Show conditional edges with labels (need_replan check)
- [ ] Document iteration limit in diagram
- [ ] Update docstring with Phase 4 architecture

### Validation
- [ ] Verify Mermaid syntax is valid
- [ ] Render diagram and check visual correctness

**Files**:
- `src/agentic/graph.py` - Update visualization function

## Task 12: Documentation
**Owner**: Implementation team
**Estimated effort**: 2 hours
**Dependencies**: Tasks 1-11

### Deliverables
- [ ] Update IMPLEMENTATION_SUMMARY.md with Phase 4 status
- [ ] Document score fusion algorithm with examples
- [ ] Document replanning mechanism and iteration limits
- [ ] Add example usage for both prompt modes
- [ ] Update graph visualization in documentation
- [ ] Document configuration options (max_replan_iterations, eta, etc.)

**Files**:
- `IMPLEMENTATION_SUMMARY.md` - Update summary

## Completion Checklist
- [ ] All 12 tasks completed
- [ ] All unit tests passing (target: >80% coverage for new code)
- [ ] No Pydantic or LangGraph warnings
- [ ] Integration test with Planner→Executor→Summarizer succeeds
- [ ] Replanning loop tested and works correctly
- [ ] Max iteration limit enforced
- [ ] Score fusion algorithm validated against manual calculations
- [ ] Example from `docs/04_module_summarizer.md` produces expected output
- [ ] Code follows project conventions (minimal files, reuse existing patterns)
- [ ] Documentation updated

## Validation Commands

### Run all Summarizer tests
```bash
pytest tests/test_summarizer*.py tests/test_score_fusion.py tests/test_replanning_logic.py -v
```

### Run integration test
```bash
pytest tests/test_full_pipeline.py -v
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

### Visualize graph
```python
from src.agentic.graph import create_agentic_graph, visualize_graph
graph = create_agentic_graph()
print(visualize_graph(graph))
```

## Notes
- Tasks can be parallelized: Task 2 (fusion), Task 3 (explanation prompt), Task 4 (scoring prompt) are independent
- Task 5 (replanning logic) can be done in parallel with Tasks 3-4
- Task 6 (Summarizer node) ties everything together
- Task 7 (graph updates) enables end-to-end testing
- Test suite (Tasks 9-10) ensures correctness and regression prevention
- Score fusion requires NumPy; already in dependencies
- Summarizer completes the three-agent architecture
