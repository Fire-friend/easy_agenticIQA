# Implementation Tasks: Fix Scoring Output Format

## Phase 1: Update State Models
- [x] Modify `SummarizerOutput` in `src/agentic/state.py`
  - Add `quality_score: Optional[float]` field for numerical scores
  - Change `final_answer: str` to `final_answer: Union[str, float]`
  - Update field validators and docstrings
  - Update examples in config_schema_examples()
- [ ] Run `pytest tests/test_state_models.py` to validate changes

## Phase 2: Complete Score Fusion Implementation
- [x] Implement `ScoreFusion.compute_alpha_weights(tool_scores)` in `src/agentic/score_fusion.py`
  - Calculate tool score mean: q̄ = (1/n) Σ q̂_i
  - Compute Gaussian weights: α_c = exp(-η(q̄-c)²) / Σ exp(-η(q̄-j)²)
  - Handle edge cases: empty scores, single score
  - Add numerical stability (subtract max before exp)
- [x] Implement `ScoreFusion.extract_vlm_probs(vlm_output)`
  - Parse quality_probs dict from VLM JSON
  - Apply softmax: p_c = exp(log p̂_c) / Σ exp(log p̂_j)
  - Fallback: classification → one-hot + smoothing
  - Fallback: no probs → uniform distribution
- [x] Implement `ScoreFusion.fuse(tool_scores, vlm_probs)`
  - Compute α weights from tool scores
  - Apply fusion: q = Σ_c α_c · p_c · c
  - Validate q ∈ [1, 5], clip if needed
  - Return continuous score and discrete level
- [x] Add comprehensive logging to all fusion methods
- [ ] Write unit tests for ScoreFusion class

## Phase 3: Update Scoring Prompt Template
- [x] Replace SCORING_PROMPT_TEMPLATE in `src/agentic/nodes/summarizer.py`
  - Request quality_probs dict with log-probabilities for levels 1-5
  - Remove categorical selection (A/B/C/D/E)
  - Emphasize numerical quality levels
- [x] Add SCORING_WITH_FUSION_PROMPT_TEMPLATE constant
- [x] Update format_evidence_for_scoring() if needed

## Phase 4: Implement Query Type Detection
- [x] Add `detect_query_type(query: str) -> QueryType` utility function
  - Enum: SCORING, MCQ, EXPLANATION
  - SCORING: "rate/score/assess quality", no explicit options
  - MCQ: "A) ... B) ..." or "choose from" patterns
  - EXPLANATION: "why/explain/describe"
- [x] **Upgrade to VLM-based intent detection**
  - Add `detect_query_type_with_vlm()` for intelligent intent recognition
  - Add `detect_query_type_rule_based()` as fallback
  - Automatic fallback when VLM fails
  - Fixes edge cases like "What is the major distortion?"
- [x] Test VLM detection with various query types

## Phase 5: Update Summarizer Node Logic
- [x] Update `summarizer_node()` in `src/agentic/nodes/summarizer.py`
  - Detect query type before selecting prompt mode
  - For SCORING queries:
    - Use SCORING_WITH_FUSION_PROMPT_TEMPLATE
    - Parse VLM response for quality_probs
    - Apply ScoreFusion.fuse() to get numerical score
    - Set final_answer=q (float), quality_score=q
  - For MCQ queries:
    - Use existing EXPLANATION_PROMPT_TEMPLATE
    - Parse final_answer as letter
    - Set quality_score=None
  - Update return statement to include quality_score
- [x] Update VLM response parsing to handle new JSON format
- [x] Add retry logic for invalid quality_probs

## Phase 6: Update Evidence Formatting
- [ ] Modify format_evidence_for_scoring() to include:
  - Tool scores with better formatting
  - Tool score mean (q̄) to guide VLM
  - Suggested quality level range based on tools
- [ ] Test evidence formatting with real Executor outputs

## Phase 7: Update Tests and Validation
- [ ] Update test cases in `tests/test_summarizer.py`
  - Test scoring query → numerical output
  - Test MCQ query → letter output
  - Test score fusion with various inputs
  - Test fallback when tool scores missing
- [ ] Add integration test with full pipeline
- [ ] Test backward compatibility with AgenticIQA-Eval MCQ dataset

## Phase 8: Update Evaluation Scripts
- [ ] Modify `scripts/eval_srocc_plcc.py`
  - Extract numerical scores from summarizer_result.quality_score
  - Handle legacy final_answer letters by mapping to numbers
- [ ] Update `scripts/eval_agenticqa_eval.py`
  - Ensure MCQ evaluation still works with letter answers
- [ ] Run evaluation on test set to verify improvements

## Phase 9: Documentation and Logging
- [ ] Update docstrings in summarizer.py and score_fusion.py
- [ ] Add detailed logging:
  - Log detected query type
  - Log fusion inputs (α weights, VLM probs)
  - Log final numerical score vs discrete level
- [ ] Update docs/04_module_summarizer.md with new behavior
- [ ] Add examples of scoring output format

## Phase 10: Final Validation
- [ ] Run demo.sh with scoring queries and verify numerical outputs
- [ ] Compare outputs with paper examples (if available)
- [ ] Verify SRCC/PLCC metrics on TID2013, BID datasets
- [ ] Check for regressions in MCQ accuracy on AgenticIQA-Eval
- [ ] Performance testing: ensure no significant latency increase

## Verification Checklist
- [x] Scoring queries return float final_answer (not letter)
- [x] quality_score field populated for IQA queries
- [x] MCQ queries still return letter final_answer
- [x] Score fusion formula matches paper exactly
- [x] α weights sum to 1.0
- [x] VLM probs sum to 1.0
- [x] Final score q ∈ [1, 5]
- [ ] All tests pass
- [ ] No breaking changes to MCQ evaluation
- [ ] Improved SRCC/PLCC correlation
