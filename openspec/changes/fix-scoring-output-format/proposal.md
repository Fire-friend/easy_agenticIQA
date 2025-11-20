# Proposal: Fix Scoring Output Format to Return Numerical Scores

## Summary
The current implementation incorrectly returns categorical ratings (A/B/C/D/E) for IQA scoring queries instead of continuous numerical scores (1-5 scale) as specified in the paper. This proposal fixes the scoring behavior to align with the paper's "Tool-Augment Score Prediction" mechanism.

## Problem Statement

### Current Behavior
When a user asks "Rate the perceptual quality of this image", the system currently:
1. Uses scoring mode prompt that forces selection from "A. Excellent, B. Good, C. Fair, D. Poor, E. Bad"
2. Returns categorical output like `final_answer: "C"`
3. Only computes tool score mean without applying the full fusion formula
4. Never returns continuous numerical scores

### Expected Behavior (from Paper)
According to the paper (Section 3.3.2, lines 382-410):
1. For IQA scoring queries, the system should produce **continuous quality scores** (1-5 scale)
2. The score should be computed using the fusion formula: `q = Σ_c∈C α_c · p_c · c`
3. Where α_c are perceptual weights from tool scores: `α_c = exp(-η(q̄-c)²) / Σ exp(-η(q̄-j)²)`
4. And p_c are VLM probabilities for each quality level c ∈ {1,2,3,4,5}
5. The final output should be a numerical score (e.g., 2.6) with quality reasoning

### Root Causes
1. **Scoring prompt template** (summarizer.py:59-73) forces categorical selection instead of requesting probability distributions
2. **Score fusion is incomplete** (summarizer.py:361-367) - only computes tool mean, doesn't apply fusion formula
3. **Output format is hardcoded** as categorical in SummarizerOutput for IQA queries
4. **No distinction** between:
   - Pure scoring queries: "Rate/score the quality" → should return numerical score
   - MCQ with quality options: "Is quality: A) Excellent B) Good..." → should return letter

## Proposed Solution

### 1. Update Scoring Mode Prompt
Replace categorical selection prompt with probability distribution request, letting the VLM directly output probabilities:

```text
System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion
analysis), assess the image quality and provide your confidence for each quality level.

Quality levels:
- Level 5: Excellent quality (no visible distortions)
- Level 4: Good quality (minor distortions)
- Level 3: Fair quality (moderate distortions)
- Level 2: Poor quality (severe distortions)
- Level 1: Bad quality (extreme distortions)

Return valid JSON:
{
  "quality_probs": {
    "1": <log_probability_for_level_1>,
    "2": <log_probability_for_level_2>,
    "3": <log_probability_for_level_3>,
    "4": <log_probability_for_level_4>,
    "5": <log_probability_for_level_5>
  },
  "quality_reasoning": "<concise justification referencing distortions and tool scores>"
}

User:
Tool scores (1-5 scale): {tool_scores}
Distortion analysis: {distortion_analysis}
The image: <image>
```

**Note:** The VLM directly outputs log-probabilities in the JSON, which are then used in the fusion formula.

### 2. Fully Implement Score Fusion
Update summarizer.py to apply the complete fusion formula per paper equations (4), (5):
- Compute tool score mean: q̄ = (1/n) Σ q̂_i
- Compute perceptual weights: α_c = exp(-η(q̄-c)²) / Σ exp(-η(q̄-j)²)  where η=1
- Parse VLM's quality_probs and apply softmax: p_c = exp(log p̂_c) / Σ exp(log p̂_j)
- Compute final continuous score: q = Σ_{c∈{1,2,3,4,5}} α_c · p_c · c
- Return numerical score (e.g., 2.73) instead of categorical rating (e.g., "C")

### 3. Update SummarizerOutput Model
Modify `SummarizerOutput` to support numerical scores:
```python
class SummarizerOutput(BaseModel):
    final_answer: Union[str, float]  # Letter for MCQ, float for scoring
    quality_score: Optional[float] = None  # Continuous score (1-5) for IQA queries
    quality_reasoning: str
    need_replan: bool = False
    replan_reason: Optional[str] = None
```

### 4. Query Type Detection
Enhance query classification to distinguish:
- **Scoring queries**: "Rate/score/assess the quality" → return numerical score
- **MCQ queries**: Explicit options provided → return letter
- **Explanation queries**: "Why/explain/describe" → return descriptive text

## Affected Components

### Modified Specs
- **summarizer-core**: Update scoring mode prompt template and output format
- **score-fusion**: Ensure full fusion formula is applied, not just tool mean
- **summarizer-state-models**: Add `quality_score` field to SummarizerOutput

### Modified Files
- `src/agentic/nodes/summarizer.py` - Update prompt, fusion logic, output format
- `src/agentic/state.py` - Modify SummarizerOutput model
- `src/agentic/score_fusion.py` - Complete fusion implementation

### New Utilities
- VLM probability extraction logic (from quality_probs or fallback to classification)
- Query type classification (scoring vs MCQ vs explanation)

## Validation

### Test Cases
1. **Pure scoring query**: "Rate the perceptual quality of this image"
   - Expected: `final_answer: 2.6, quality_score: 2.6, quality_reasoning: "..."`

2. **MCQ with quality options**: "Is quality: A) Excellent B) Good C) Fair?"
   - Expected: `final_answer: "C", quality_score: null, quality_reasoning: "..."`

3. **Fusion with tool scores**: tool_scores=[2.5, 2.7], VLM probs favor level 3
   - Expected: Score between 2.5 and 3.0 reflecting weighted fusion

4. **Empty tool scores**: No tool evidence available
   - Expected: Uniform α weights, rely on VLM probs

### Acceptance Criteria
- [ ] Scoring queries return continuous numerical scores (1-5 scale)
- [ ] Score fusion formula is fully applied per paper specification
- [ ] MCQ queries still return letter answers
- [ ] Quality reasoning references both tool scores and VLM assessment
- [ ] Backward compatibility: existing MCQ evaluation still works
- [ ] SRCC/PLCC evaluation metrics improve with continuous scores

## Migration Impact

### Breaking Changes
- SummarizerOutput.final_answer changes from always str to Union[str, float]
- Downstream code expecting categorical output must handle numerical scores

### Compatibility Strategy
- Check query type: if MCQ with explicit options, return letter (backward compatible)
- If pure scoring query, return numerical score (new behavior, aligns with paper)
- Add quality_score field to carry numerical value explicitly

## Implementation Order

1. **Update SummarizerOutput model** - Add quality_score field, make final_answer flexible
2. **Implement full score fusion** - Complete ScoreFusion.fuse() method
3. **Update scoring prompt template** - Request probability distributions
4. **Implement VLM probability extraction** - Parse quality_probs or fallback
5. **Update summarizer_node logic** - Apply fusion, return numerical score
6. **Add query type detection** - Distinguish scoring vs MCQ
7. **Update tests** - Validate scoring output format
8. **Update evaluation scripts** - Handle numerical scores in SRCC/PLCC calculation

## References
- Paper Section 3.3.2: "Tool-Augment Score Prediction" (lines 382-410)
- Paper equations: α_c formula (line 391-394), fusion formula (line 408)
- docs/04_module_summarizer.md: Section 3 "分数融合算法"
- Current implementation: src/agentic/nodes/summarizer.py:347-373
