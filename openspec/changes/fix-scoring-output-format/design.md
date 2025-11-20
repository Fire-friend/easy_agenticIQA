# Design: Fix Scoring Output Format

## Architecture Overview

### Current Flow (INCORRECT)
```
User: "Rate the perceptual quality of this image"
  ↓
Planner: query_type="IQA"
  ↓
Executor: quality_scores = {"Global": {"Blurs": ["TOPIQ_FR", 2.6]}}
  ↓
Summarizer:
  - Uses categorical prompt (A/B/C/D/E)
  - Only computes tool mean (not fusion)
  - Returns final_answer="C" (WRONG - should be 2.6)
  ↓
Output: "C" (categorical, not numerical)
```

### Proposed Flow (CORRECT)
```
User: "Rate the perceptual quality of this image"
  ↓
Planner: query_type="IQA"
  ↓
Executor: quality_scores = {"Global": {"Blurs": ["TOPIQ_FR", 2.6]}}
  ↓
Summarizer:
  - Detect query type: SCORING (no explicit options)
  - Use probability distribution prompt
  - VLM returns quality_probs: {1: -3.2, 2: -0.5, 3: -0.1, 4: -2.1, 5: -4.5}
  - ScoreFusion:
    1. α weights from tool scores: q̄=2.6 → α weights peak at level 3
    2. VLM probs: softmax(quality_probs) → {1:0.05, 2:0.15, 3:0.60, 4:0.15, 5:0.05}
    3. Fusion: q = Σ α_c · p_c · c = 2.73
  - Returns final_answer=2.73, quality_score=2.73
  ↓
Output: 2.73 (numerical score)
```

## Key Design Decisions

### Decision 1: Query Type Classification
**Problem**: How to distinguish scoring queries from MCQ queries?

**Options**:
1. **Pattern matching on query text** (CHOSEN)
   - Pros: Simple, no API changes
   - Cons: May miss edge cases
   - Patterns:
     - SCORING: "rate|score|assess|evaluate quality" + no options
     - MCQ: "A\)" or "choose from" or explicit options

2. **Add explicit query_subtype to Planner**
   - Pros: More reliable
   - Cons: Requires Planner changes, more complex

**Decision**: Use pattern matching (Option 1) for minimal changes. Add `detect_query_type()` utility.

### Decision 2: VLM Probability Extraction
**Problem**: How to get VLM probability distributions for quality levels?

**Options**:
1. **Let VLM directly output probabilities in JSON** (CHOSEN)
   - Pros: Simple, model-agnostic, matches paper's description
   - Cons: Relies on VLM's ability to estimate its own confidence
   - Implementation: Prompt asks VLM to return log-probabilities for each level

2. **Extract from API logprobs parameter**
   - Pros: True model probabilities
   - Cons: API-specific, not available for all models (Claude doesn't support)

3. **Use classification + smoothing fallback**
   - Pros: Works with any VLM
   - Cons: Less accurate than true probabilities

**Decision**: Let VLM output probabilities directly (Option 1). The paper states "the summarizer obtains log-probabilities log p̂_c" without specifying the mechanism, so we ask the VLM to provide them in its JSON output. Fallback to Option 3 if VLM fails to provide quality_probs.

### Decision 3: Output Format Compatibility
**Problem**: How to maintain backward compatibility with MCQ evaluation?

**Options**:
1. **Dual fields: final_answer (flexible) + quality_score (explicit)** (CHOSEN)
   - Pros: Clear separation, backward compatible
   - Cons: Slight redundancy

2. **Always return numerical score, map to letters externally**
   - Pros: Simpler model
   - Cons: Breaks MCQ evaluation, loses letter answer context

**Decision**: Use dual fields (Option 1):
- `final_answer`: Union[str, float] - letter for MCQ, float for scoring
- `quality_score`: Optional[float] - always populated for IQA queries

### Decision 4: Fusion Parameter Configuration
**Problem**: Should fusion parameters (η, quality levels) be configurable?

**Options**:
1. **Hardcode η=1.0, levels={1,2,3,4,5}** (CHOSEN)
   - Pros: Matches paper, simple
   - Cons: No flexibility

2. **Make configurable via YAML**
   - Pros: Research flexibility
   - Cons: More complexity, not in paper

**Decision**: Hardcode (Option 1) for initial implementation. Can add config later if needed.

## Component Design

### 1. ScoreFusion Class
```python
class ScoreFusion:
    def __init__(self, eta: float = 1.0, quality_levels: List[int] = [1,2,3,4,5]):
        self.eta = eta
        self.quality_levels = quality_levels

    def compute_alpha_weights(self, tool_scores: List[float]) -> Dict[int, float]:
        """Compute perceptual weights from tool scores using Gaussian."""
        if not tool_scores:
            return uniform_weights()

        q_bar = sum(tool_scores) / len(tool_scores)
        weights = {}

        for c in self.quality_levels:
            weights[c] = exp(-self.eta * (q_bar - c)**2)

        # Normalize
        total = sum(weights.values())
        return {c: w/total for c, w in weights.items()}

    def extract_vlm_probs(self, vlm_output: Dict) -> Dict[int, float]:
        """Extract VLM probability distribution from quality_probs."""
        if "quality_probs" in vlm_output:
            log_probs = vlm_output["quality_probs"]
            return softmax(log_probs)  # {1: p1, 2: p2, ...}

        # Fallback: classification
        if "quality_level" in vlm_output:
            level = vlm_output["quality_level"]
            return one_hot_with_smoothing(level, epsilon=0.15)

        # Fallback: uniform
        return uniform_distribution()

    def fuse(self, tool_scores: List[float], vlm_probs: Dict[int, float]) -> float:
        """Apply fusion formula: q = Σ α_c · p_c · c"""
        alpha = self.compute_alpha_weights(tool_scores)

        score = 0.0
        for c in self.quality_levels:
            score += alpha[c] * vlm_probs[c] * c

        # Validate and clip
        score = max(1.0, min(5.0, score))

        logger.info(f"Fusion: q̄={mean(tool_scores):.2f}, α={alpha}, p={vlm_probs}, q={score:.2f}")
        return score
```

### 2. Query Type Detection
```python
from enum import Enum

class QueryType(Enum):
    SCORING = "scoring"      # Pure quality scoring
    MCQ = "mcq"              # Multiple choice question
    EXPLANATION = "explanation"  # Descriptive explanation

def detect_query_type(query: str) -> QueryType:
    """Detect query type from text patterns."""
    query_lower = query.lower()

    # MCQ: explicit options
    if re.search(r'[A-E]\)', query) or "choose from" in query_lower:
        return QueryType.MCQ

    # SCORING: rate/score/assess keywords + no options
    if re.search(r'\b(rate|score|assess|evaluate)\b.*\bquality\b', query_lower):
        return QueryType.SCORING

    # EXPLANATION: why/explain/describe
    if re.search(r'\b(why|explain|describe|what)\b', query_lower):
        return QueryType.EXPLANATION

    # Default: treat as MCQ if planner says query_type="IQA"
    return QueryType.SCORING
```

### 3. Updated Summarizer Logic
```python
def summarizer_node(state, config, max_retries=3):
    plan = state["plan"]
    executor_output = state["executor_evidence"]
    query = state["query"]

    # Detect query type
    query_type = detect_query_type(query)
    logger.info(f"Detected query type: {query_type}")

    if plan.query_type == "IQA" and query_type == QueryType.SCORING:
        # SCORING MODE: request probability distributions
        prompt = SCORING_WITH_FUSION_PROMPT_TEMPLATE.format(
            query=query,
            distortion_analysis=format_distortion_analysis(executor_output),
            tool_scores=format_tool_scores(executor_output)
        )

        vlm_response = vlm_client.generate(prompt, images, ...)
        response_data = parse_json_response(vlm_response)

        # Extract VLM probabilities
        fusion = ScoreFusion(eta=1.0)
        vlm_probs = fusion.extract_vlm_probs(response_data)

        # Collect tool scores
        tool_scores = extract_all_tool_scores(executor_output)

        # Apply fusion
        if tool_scores:
            final_score = fusion.fuse(tool_scores, vlm_probs)
        else:
            # No tool scores: use VLM probs only
            final_score = sum(c * vlm_probs[c] for c in [1,2,3,4,5])

        return {
            "summarizer_result": SummarizerOutput(
                final_answer=final_score,
                quality_score=final_score,
                quality_reasoning=response_data["quality_reasoning"],
                need_replan=check_evidence_sufficiency(...)
            )
        }

    elif query_type == QueryType.MCQ:
        # MCQ MODE: categorical selection
        prompt = EXPLANATION_PROMPT_TEMPLATE.format(...)
        vlm_response = vlm_client.generate(prompt, images, ...)
        response_data = parse_json_response(vlm_response)

        return {
            "summarizer_result": SummarizerOutput(
                final_answer=response_data["final_answer"],  # Letter
                quality_score=None,
                quality_reasoning=response_data["quality_reasoning"],
                need_replan=check_evidence_sufficiency(...)
            )
        }

    else:
        # EXPLANATION MODE: descriptive answer
        # ... similar to MCQ mode
```

### 4. Updated Prompt Template
```python
SCORING_WITH_FUSION_PROMPT_TEMPLATE = """System:
You are a visual quality assessment assistant. Assess the image quality and provide your confidence for each quality level.

Quality levels:
- Level 5 (Excellent): no visible distortions
- Level 4 (Good): minor distortions, minimal impact
- Level 3 (Fair): moderate distortions, noticeable impact
- Level 2 (Poor): severe distortions, significant impact
- Level 1 (Bad): extreme distortions, unusable quality

Return valid JSON with log-probabilities for each level:
{{
  "quality_probs": {{
    "1": <log_prob_for_level_1>,
    "2": <log_prob_for_level_2>,
    "3": <log_prob_for_level_3>,
    "4": <log_prob_for_level_4>,
    "5": <log_prob_for_level_5>
  }},
  "quality_reasoning": "<concise justification referencing distortions and tool scores>"
}}

User:
Query: {query}
Tool scores (1-5 scale, higher is better): {tool_scores}
Tool score mean: {tool_mean:.2f}
Distortion analysis: {distortion_analysis}

The image: <image>"""
```

**Implementation Note**: The VLM is asked to directly output log-probabilities in the JSON response. While these are the model's self-assessed confidences rather than true logits, this approach is model-agnostic and aligns with the paper's description of "obtaining" log-probabilities.

## Error Handling

### Edge Cases
1. **Empty tool scores**: Use uniform α weights, rely on VLM probs
2. **Invalid quality_probs**: Retry with stricter prompt, fallback to classification
3. **VLM probs don't sum to 1**: Renormalize with warning
4. **Numerical instability**: Use log-sum-exp trick for softmax
5. **Score out of range**: Clip to [1, 5] with warning

### Fallback Strategy
```
Request quality_probs
  ↓ (fails)
Retry 1: Add "Return ONLY valid JSON"
  ↓ (fails)
Retry 2: Add explicit format example
  ↓ (fails)
Fallback: Extract quality level from reasoning text
  ↓
Apply fusion with one-hot probabilities (smoothed)
```

## Performance Considerations

### Latency
- New prompt requires VLM to generate probabilities (5 values) instead of 1 letter
- Expected increase: +5-10% tokens, minimal latency impact
- Fusion computation: O(n) where n=5, negligible

### Token Usage
- Prompt adds ~50 tokens (quality_probs format)
- Response adds ~30 tokens (5 log-probs)
- Total increase: ~80 tokens per query (~0.5% for typical prompt)

### Caching
- Cache fusion results by (tool_scores, vlm_probs) tuple
- Cache VLM responses by (query, image_hash) for repeated queries

## Testing Strategy

### Unit Tests
- ScoreFusion.compute_alpha_weights with various tool scores
- ScoreFusion.extract_vlm_probs with different JSON formats
- ScoreFusion.fuse with known test cases
- detect_query_type with edge cases

### Integration Tests
- End-to-end: scoring query → numerical output
- End-to-end: MCQ query → letter output
- Score fusion with real VLM responses
- Backward compatibility: MCQ evaluation unchanged

### Validation Tests
- Compare fusion output with manual calculation
- Verify α weights sum to 1.0
- Verify VLM probs sum to 1.0
- Verify final score ∈ [1, 5]

## Migration Plan

### Backward Compatibility
- Old code expects `final_answer` as str → still works for MCQ
- New code checks `quality_score` field → None for MCQ, float for scoring
- Evaluation scripts handle both formats

### Rollout
1. Deploy with feature flag: `use_numerical_scoring=True`
2. A/B test: compare categorical vs numerical outputs
3. Validate SRCC/PLCC improvements
4. Enable globally after validation

## Open Questions
1. Should we support 7-level scale (1-7) for some datasets? → Future work
2. How to handle partial tool scores (e.g., only 1 tool)? → Use single score as q̄
3. Should we expose fusion parameters in config? → Not initially
4. What if VLM refuses to provide log-probs? → Fallback to classification + smoothing
