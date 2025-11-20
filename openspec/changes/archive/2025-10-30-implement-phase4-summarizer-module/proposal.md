# Proposal: Implement Phase 4 - Summarizer Module

## Summary
Implement the Summarizer agent module, the final component in the AgenticIQA three-agent pipeline (Planner→Executor→Summarizer). The Summarizer integrates evidence from the Executor with direct visual understanding to produce final answers and quality reasoning, with support for score fusion in scoring tasks and conditional replanning when evidence is insufficient.

## Why
Phase 3 implemented the Executor module which collects distortion evidence and tool scores. Phase 4 introduces the evidence synthesis and decision-making capability by implementing the Summarizer module as specified in `docs/04_module_summarizer.md`. The Summarizer is critical as it:
- Synthesizes Executor evidence (distortion analysis, tool scores) with VLM visual understanding
- Produces human-interpretable final answers and quality reasoning
- Implements weighted score fusion for scoring tasks (combining tool scores with VLM probabilities)
- Enables self-correction through conditional replanning when evidence is insufficient
- Completes the end-to-end AgenticIQA pipeline

## What Changes
This change introduces four new capabilities:

### 1. **summarizer-core** - Summarizer agent with evidence integration
- **Explanation/QA Mode**: Answers user questions using evidence or direct visual analysis
- **Scoring Mode**: Assesses image quality and maps to discrete levels (A-E: Excellent/Good/Fair/Poor/Bad)
- **Evidence Integration**: Formats distortion_analysis and quality_scores for VLM prompts
- **Replanning Logic**: Determines if evidence is sufficient or if replanning is needed

### 2. **summarizer-state-models** - State extensions for Summarizer
- `SummarizerOutput` model with final_answer, quality_reasoning, need_replan fields
- Extend `AgenticIQAState` to include summarizer_result field
- Field validators for answer format and reasoning content

### 3. **score-fusion** - Weighted fusion for scoring tasks
- Perceptual weight calculation using Gaussian: α_c = exp(-η(q̄ - c)²) / Σ_j exp(-η(q̄ - j)²)
- VLM probability extraction from logits or classification output
- Fusion formula: q = Σ_c α_c · p_c · c
- Mapping to discrete quality levels (1-5 or A-E)

### 4. **replanning-logic** - Graph updates for conditional replanning
- Add Summarizer node to StateGraph
- Conditional edge: if need_replan=true → Planner, else → END
- Iteration counter to prevent infinite loops (max 2 replanning iterations)
- State propagation: Summarizer feedback → Planner for context-aware replanning

## Context
- **Documentation**: `docs/04_module_summarizer.md` provides detailed specifications, prompt templates, fusion algorithm, and test examples
- **Dependencies**: Phase 2 (Planner) and Phase 3 (Executor) completed and archived
- **Architecture**: Follows LangGraph node pattern with conditional edges for replanning loop
- **VLM Integration**: Reuses VLM client abstraction from Phases 2 & 3

## Out of Scope
- AgenticIQA-Eval MCQ evaluation (Phase 5)
- SRCC/PLCC scoring evaluation on TID2013/BID/AGIQA-3K (Phase 5)
- Advanced fusion strategies beyond the paper's weighted combination
- Fine-tuning VLMs for better probability calibration
- Multi-turn conversation or clarification questions

## Success Criteria
1. Summarizer accepts Executor output and produces valid JSON with final_answer, quality_reasoning, need_replan
2. Two prompt modes (explanation/QA and scoring) produce appropriate outputs for different task types
3. Score fusion algorithm correctly combines tool scores with VLM probabilities for scoring tasks
4. Replanning mechanism triggers when evidence is insufficient and properly feeds back to Planner
5. Graph supports conditional flow: Executor → Summarizer → [Planner (if replan) or END]
6. Unit tests validate JSON structure, fusion algorithm, and replanning logic
7. Integration tests verify end-to-end Planner→Executor→Summarizer→(optional replan)→END flow
8. Max iteration limit prevents infinite replanning loops

## Dependencies
- Phase 2: Planner module (completed and archived)
- Phase 3: Executor module (completed and archived)
- Libraries: NumPy (for fusion algorithm), existing VLM client abstraction

## Risks & Mitigations
- **Risk**: VLM produces invalid JSON or inconsistent answers
  - **Mitigation**: Retry with stricter prompts, JSON schema validation with Pydantic, fallback to no-replan
- **Risk**: Score fusion requires logits but VLM doesn't expose them
  - **Mitigation**: Allow direct classification mode with post-processing, document limitations
- **Risk**: Replanning loops indefinitely
  - **Mitigation**: Hard limit on iterations (default 2), track iteration count in state
- **Risk**: Summarizer ignores Executor evidence
  - **Mitigation**: Explicitly format evidence in prompts, test coverage for evidence utilization
- **Risk**: Integration complexity across three agents
  - **Mitigation**: Comprehensive integration tests, state logging, visualization tools

## Implementation Strategy
1. Start with SummarizerOutput Pydantic models for type safety
2. Implement score fusion utility as standalone module (testable independently)
3. Build Summarizer node with two prompt template functions
4. Implement replanning decision logic within Summarizer
5. Update graph with conditional edges and iteration tracking
6. Add unit tests for fusion algorithm and Summarizer outputs
7. Add integration tests for full pipeline with and without replanning
8. Implement iteration counter and max limit enforcement

## Testing Approach
- Unit tests: JSON schema validation, fusion algorithm with known inputs, prompt rendering
- Integration tests: Planner→Executor→Summarizer flow with mocked VLM outputs
- Replanning tests: Trigger replanning with insufficient evidence, verify loop behavior
- Fusion tests: Compare fusion results with manual calculations
- Manual tests: Real VLM + end-to-end pipeline with examples from documentation
- Fixtures: Save example inputs/outputs for Phase 5 evaluation

## Related Work
- Builds on Planner (Phase 2) and Executor (Phase 3) modules
- Completes the three-agent architecture described in the paper
- Follows LangGraph conditional edge patterns for replanning
- Prepares state for Phase 5 evaluation
