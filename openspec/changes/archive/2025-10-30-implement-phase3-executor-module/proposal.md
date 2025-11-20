# Proposal: Implement Phase 3 - Executor Module

## Summary
Implement the Executor agent module, the second component in the AgenticIQA three-agent pipeline (Planner→Executor→Summarizer). The Executor collects distortion evidence and quantitative scores through four subtasks guided by the Planner's control flags, integrating VLM reasoning with traditional IQA tools.

## Why
Phase 2 implemented the Planner module which analyzes user queries and generates structured plans. Phase 3 introduces the evidence collection capability by implementing the Executor module as specified in `docs/03_module_executor.md`. The Executor is critical as it:
- Bridges VLM understanding with traditional IQA tool measurements
- Provides structured evidence (distortions, severity analysis, tool scores) for the Summarizer
- Enables query-aware quality assessment through conditional subtask execution
- Normalizes heterogeneous tool outputs to a unified 1-5 scale

## What Changes
This change introduces three new capabilities:

### 1. **executor-core** - Executor agent with four subtasks
- **Distortion Detection**: Identifies potential distortion types when `distortion_source="Inferred"`
- **Distortion Analysis**: Estimates distortion severity and visual impact
- **Tool Selection**: Assigns appropriate IQA tools based on distortion types and tool capabilities
- **Tool Execution**: Runs selected tools and normalizes scores to 1-5 scale using logistic mapping

### 2. **tool-registry** - IQA tool integration system
- Tool metadata management (type, strengths, parameters)
- Tool execution interface with caching support
- Logistic score normalization (five-parameter monotonic function)
- Support for IQA-PyTorch tools (TOPIQ_FR, QAlign, LPIPS, DISTS, BRISQUE, NIQE, etc.)

### 3. **executor-state-models** - State extensions for Executor
- `ExecutorOutput` model with distortion_set, distortion_analysis, selected_tools, quality_scores
- `DistortionAnalysis` model with type, severity, explanation fields
- `ToolExecutionLog` model for recording tool calls and scores
- Extend `AgenticIQAState` to include executor_evidence field

## Context
- **Documentation**: `docs/03_module_executor.md` provides detailed specifications, prompt templates, and test examples
- **Dependencies**: Phase 2 (Planner module) completed and archived
- **Architecture**: Follows LangGraph node pattern, conditionally executes subtasks based on Planner's control flags
- **Tool Integration**: Requires IQA-PyTorch installation and tool weight management

## Out of Scope
- Summarizer module (Phase 4)
- Full replanning loop (requires Summarizer)
- AgenticIQA-Eval MCQ evaluation (Phase 5)
- SRCC/PLCC scoring evaluation on TID2013/BID/AGIQA-3K (Phase 5)
- Advanced caching strategies (simple hash-based caching is sufficient)

## Success Criteria
1. Executor accepts Planner output and executes subtasks based on control flags
2. Four subtasks produce valid JSON outputs matching schemas in `docs/03_module_executor.md`
3. Tool registry successfully loads metadata and executes IQA-PyTorch tools
4. Score normalization maps tool outputs to 1-5 scale using logistic function
5. Unit tests validate JSON structure, control flag logic, and tool execution
6. Integration tests verify Planner→Executor data flow
7. Error handling covers tool failures, VLM timeouts, and invalid outputs

## Dependencies
- Phase 2: Planner module (completed and archived)
- External: IQA-PyTorch installed and weights downloaded
- Libraries: IQA-PyTorch, OpenCV, NumPy, SciPy

## Risks & Mitigations
- **Risk**: Tool outputs are inconsistent or fail
  - **Mitigation**: Implement fallback to generic NR tools (BRISQUE, NIQE), log failures
- **Risk**: VLM produces invalid JSON for subtasks
  - **Mitigation**: Retry with stricter prompts, JSON schema validation with Pydantic
- **Risk**: Score normalization parameters unavailable
  - **Mitigation**: Use empirical defaults from paper Appendix A.3, document alternatives
- **Risk**: Tool execution is slow
  - **Mitigation**: Implement image hash-based caching, support batching where possible

## Implementation Strategy
1. Start with ExecutorOutput Pydantic models for type safety
2. Implement tool registry with metadata loading and execution interface
3. Build four subtask functions with prompt templates from docs
4. Create Executor node that orchestrates subtasks based on control flags
5. Integrate Executor node into LangGraph StateGraph
6. Add unit tests with fixture examples from `docs/03_module_executor.md`
7. Implement caching and logging for tool execution

## Testing Approach
- Unit tests: JSON schema validation for each subtask output
- Integration tests: Planner→Executor flow with mocked tool outputs
- Tool tests: Verify IQA-PyTorch integration with sample images
- Manual tests: Real VLM + tool calls with examples from documentation
- Fixtures: Save example inputs/outputs for Phase 4 integration

## Related Work
- Builds on Planner module (Phase 2)
- Prepares state for Summarizer module (Phase 4)
- Follows architecture patterns in `openspec/project.md`
- Complements VLM integration from Phase 2
