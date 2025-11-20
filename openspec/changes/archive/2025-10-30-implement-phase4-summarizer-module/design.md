# Design: Summarizer Module Architecture

## Overview
The Summarizer module is the final synthesis component in AgenticIQA that integrates Executor evidence with VLM visual understanding to produce final answers and quality reasoning. It supports two modes (explanation/QA and scoring), implements weighted score fusion for scoring tasks, and enables self-correction through conditional replanning.

## Architectural Decisions

### 1. Two-Mode Prompt Strategy
**Decision**: Implement separate prompt templates for explanation/QA mode and scoring mode.

**Rationale**:
- Different task types require different output formats and reasoning strategies
- Explanation/QA mode: Open-ended answers with flexible evidence usage
- Scoring mode: Discrete quality levels (A-E) with structured evidence integration
- Explicit mode selection based on Planner's `query_type` field

**Implementation**:
```python
EXPLANATION_PROMPT_TEMPLATE = """
System:
You are a visual quality assessment assistant. Your task is to select the most appropriate answer...
[Full template from docs/04_module_summarizer.md]
"""

SCORING_PROMPT_TEMPLATE = """
System:
You are a visual quality assessment assistant. Given the question and the analysis...
Must select one single answer from: A. Excellent, B. Good, C. Fair, D. Poor, E. Bad
[Full template from docs/04_module_summarizer.md]
"""
```

**Trade-offs**:
- Two templates vs. unified dynamic template
- **Chosen**: Separate templates for clarity and paper alignment

### 2. Score Fusion Algorithm Design
**Decision**: Implement fusion as standalone utility module with clear separation of concerns.

**Formula** (from paper):
```
1. Tool score average: q̄ = (1/n) Σ q̂_i
2. Perceptual weights: α_c = exp(-η(q̄ - c)²) / Σ_j exp(-η(q̄ - j)²), η=1
3. VLM probabilities: p_c = exp(log p̂_c) / Σ_j exp(log p̂_j)
4. Final score: q = Σ_c α_c · p_c · c
```

**Rationale**:
- Gaussian weights center around tool score mean, reducing influence of outliers
- VLM probabilities provide semantic understanding complementing tool scores
- Fusion balances objective measurements with perceptual judgment
- Standalone module enables independent testing and reuse

**Implementation**:
```python
class ScoreFusion:
    def __init__(self, eta: float = 1.0, quality_levels: List[int] = [1, 2, 3, 4, 5]):
        self.eta = eta
        self.quality_levels = quality_levels

    def compute_perceptual_weights(self, tool_scores: List[float]) -> Dict[int, float]:
        """Compute Gaussian weights centered at tool score mean"""
        pass

    def extract_vlm_probabilities(self, vlm_output: Union[Dict, str]) -> Dict[int, float]:
        """Extract probabilities from VLM logits or classification"""
        pass

    def fuse_scores(self, tool_scores: List[float], vlm_probabilities: Dict[int, float]) -> float:
        """Apply fusion formula: q = Σ α_c · p_c · c"""
        pass

    def map_to_level(self, score: float) -> str:
        """Map continuous score to discrete level (A-E)"""
        pass
```

**Trade-offs**:
- Simple averaging vs. weighted fusion
- **Chosen**: Weighted fusion for paper alignment and better accuracy

### 3. Replanning Decision Logic
**Decision**: Summarizer evaluates evidence sufficiency and sets `need_replan` flag.

**Criteria for replanning**:
- Distortion analysis missing for query_scope objects
- Tool scores missing for key distortions
- Contradictory evidence between analysis and scores
- Explicit uncertainty in Executor outputs

**Rationale**:
- Agent self-awareness improves robustness
- Planner can refine strategy with Summarizer feedback
- Prevents low-quality outputs from incomplete evidence
- Limits iterations to prevent infinite loops

**Implementation**:
```python
def should_replan(executor_output: ExecutorOutput, query_scope: List[str]) -> Tuple[bool, str]:
    """
    Determine if replanning is needed based on evidence quality.

    Returns:
        (need_replan, reason)
    """
    # Check coverage of query_scope
    if executor_output.distortion_analysis:
        covered_objects = set(executor_output.distortion_analysis.keys())
        required_objects = set(query_scope) if query_scope != "Global" else {"Global"}
        if not required_objects.issubset(covered_objects):
            return True, f"Missing analysis for {required_objects - covered_objects}"

    # Check tool scores availability
    if executor_output.quality_scores is None or len(executor_output.quality_scores) == 0:
        return True, "No tool scores available"

    return False, ""
```

**Trade-offs**:
- Aggressive replanning vs. conservative (only clear gaps)
- **Chosen**: Conservative approach to reduce API costs and latency

### 4. State Model Extensions
**Decision**: Add `summarizer_result: NotRequired[SummarizerOutput]` to `AgenticIQAState`.

**Rationale**:
- Maintains consistency with Executor pattern (optional field)
- Allows Planner-only or Planner-Executor execution without Summarizer
- Structured output supports result tracking and evaluation
- Aligns with LangGraph state management patterns

**Structure**:
```python
class SummarizerOutput(BaseModel):
    final_answer: str = Field(..., description="Final answer (MCQ letter or quality level)")
    quality_reasoning: str = Field(..., min_length=1, description="Evidence-based explanation")
    need_replan: bool = Field(default=False, description="Whether replanning is needed")
    replan_reason: Optional[str] = Field(None, description="Reason for replanning")
    used_evidence: Optional[Dict[str, Any]] = Field(None, description="Evidence items referenced")
```

### 5. VLM Client Reuse
**Decision**: Reuse VLM client abstraction from Phases 2 & 3.

**Rationale**:
- Consistent API across all three agents
- Configuration loading already implemented
- Supports multiple VLM backends (OpenAI, Anthropic, Google)
- Leverages retry logic and error handling

**Implementation**: Load summarizer VLM config from `configs/model_backends.yaml`:
```yaml
summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 512
```

### 6. Graph Conditional Edge Strategy
**Decision**: Implement conditional edge function for replanning decision.

**Logic Flow**:
```python
def decide_next_node(state: AgenticIQAState) -> Literal["planner", "__end__"]:
    """
    Conditional edge function after Summarizer.

    Returns:
        "planner" if replanning needed and iterations < max_iterations
        "__end__" otherwise
    """
    summarizer_result = state.get("summarizer_result")
    if not summarizer_result:
        return "__end__"

    # Check iteration count
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_replan_iterations", 2)

    if summarizer_result.need_replan and iteration < max_iterations:
        return "planner"

    return "__end__"
```

**Rationale**:
- LangGraph's conditional edges enable dynamic routing
- Iteration counter prevents infinite loops
- Flexible max_iterations allows configuration per task
- Clear termination conditions

**Trade-offs**:
- Fixed routing vs. dynamic conditional routing
- **Chosen**: Conditional routing for self-correction capability

### 7. Iteration Tracking
**Decision**: Add `iteration_count` and `max_replan_iterations` to state.

**Rationale**:
- Prevents infinite replanning loops
- Configurable per task (default: 2)
- Tracks pipeline progress for logging and debugging
- Enables performance analysis (replanning frequency)

**Implementation**:
```python
class AgenticIQAState(TypedDict):
    # Existing fields...
    summarizer_result: NotRequired[SummarizerOutput]
    iteration_count: NotRequired[int]  # NEW: Track replanning iterations
    max_replan_iterations: NotRequired[int]  # NEW: Max iterations (default 2)
    replan_history: NotRequired[List[str]]  # NEW: History of replan reasons
```

### 8. Evidence Formatting Strategy
**Decision**: Format Executor evidence as structured JSON in VLM prompts.

**Example**:
```python
evidence_json = {
    "distortion_analysis": {
        "Global": [
            {"type": "Blurs", "severity": "moderate", "explanation": "Edges appear soft."}
        ]
    },
    "quality_scores": {
        "Global": {
            "Blurs": ["TOPIQ_FR", 2.6]
        }
    }
}
```

**Rationale**:
- Structured format improves VLM parsing
- JSON aligns with model training data
- Easy to extend with additional evidence fields
- Supports automated validation

**Trade-offs**:
- JSON vs. natural language table
- **Chosen**: JSON for consistency and parseability

### 9. Error Handling Strategy
**Decision**: Three-tier approach consistent with Planner and Executor:
1. **Validation**: Pydantic schema validation for Summarizer outputs
2. **Retry**: Up to 3 attempts for VLM calls with stricter prompts
3. **Fallback**: If all retries fail, set need_replan=false and return generic answer

**Logging**: All errors logged with context (attempt, error type, evidence summary).

**State**: Errors recorded in `summarizer_result` with error flag.

### 10. Fusion Algorithm Fallback
**Decision**: Support both logit-based and classification-based VLM outputs.

**Rationale**:
- Not all VLM APIs expose logits (e.g., Claude, some OpenAI endpoints)
- Classification-only mode: VLM returns single letter, use uniform probabilities or skip fusion
- Graceful degradation maintains functionality across backends

**Implementation**:
```python
def extract_vlm_probabilities(vlm_output, mode="logits"):
    if mode == "logits":
        # Extract logits from API response
        return softmax(logits)
    elif mode == "classification":
        # VLM returns single letter, assign high probability
        return {predicted_level: 0.8, ...}  # Distribute remaining probability
    else:
        # Fallback: uniform distribution
        return {level: 0.2 for level in [1, 2, 3, 4, 5]}
```

**Trade-offs**:
- Logit-based fusion vs. classification-only
- **Chosen**: Support both with fallback for flexibility

## Implementation Sequence
1. **SummarizerOutput models** (state.py) - Type safety foundation
2. **Score fusion utility** (score_fusion.py) - Standalone fusion algorithm
3. **Summarizer node** (nodes/summarizer.py) - Two prompt modes, evidence integration
4. **Replanning logic** (nodes/summarizer.py) - Evidence sufficiency check
5. **Graph updates** (graph.py) - Add Summarizer node, conditional edges, iteration tracking
6. **Tests** (tests/test_summarizer.py, tests/test_score_fusion.py, tests/test_integration.py)

## Validation Approach
- Mock VLM outputs for unit tests (deterministic)
- Test fusion algorithm with known inputs and expected outputs
- Example from `docs/04_module_summarizer.md` as regression test
- Verify Planner→Executor→Summarizer→END and replanning loop correctness
- Test iteration limit enforcement
