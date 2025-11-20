# Proposal: Implement Phase 2 - Planner Module

## Summary
Implement the Planner agent module, the first component in the AgenticIQA three-agent pipeline (Planner→Executor→Summarizer). The Planner analyzes user queries, images, and optional reference images to generate structured JSON plans that guide downstream execution.

## Motivation
Phase 1 established the project environment and infrastructure. Phase 2 introduces the core agentic reasoning capability by implementing the Planner module as specified in `docs/02_module_planner.md`. The Planner is critical as it determines:
- Task type (IQA vs Other) and reference mode (Full-Reference vs No-Reference)
- Query scope (Global or specific objects)
- Distortion detection strategy (Explicit vs Inferred)
- Control flags for Executor subtasks (detection, analysis, tool selection, tool execution)

## Context
- **Documentation**: `docs/02_module_planner.md` provides detailed specifications
- **Dependencies**: Phase 1 (environment setup) completed
- **Architecture**: Follows LangGraph node pattern with Pydantic state models
- **VLM Integration**: Requires abstraction layer for OpenAI, Anthropic, Qwen2.5-VL, Google

## Scope
This change introduces four new capabilities:

### 1. **planner-core** - Planner agent implementation
- Pydantic models for plan input/output JSON schema
- Planner node implementation with prompt template from paper (Appendix A.2)
- JSON output parsing and validation
- Retry logic and error handling

### 2. **vlm-integration** - VLM client abstraction
- Unified VLM client interface supporting vision + text inputs
- Provider implementations: OpenAI (GPT-4o), Anthropic (Claude 3.5), Google (Gemini)
- Configuration loading from `configs/model_backends.yaml`
- Fallback model support

### 3. **state-models** - LangGraph state definitions
- Pydantic models for AgenticIQA state (plan, evidence, result)
- Type-safe state transitions
- State serialization/deserialization for checkpointing

### 4. **langgraph-setup** - Basic graph orchestration
- StateGraph initialization with Planner node
- Entry/exit point configuration
- State persistence and logging integration

## Out of Scope
- Executor and Summarizer modules (Phase 3 & 4)
- Full replanning loop (requires Summarizer)
- Tool registry and IQA tool integration (Phase 3)
- End-to-end evaluation pipelines (Phase 5)
- Qwen2.5-VL local inference (optional, can use API fallback)

## Success Criteria
1. Planner accepts user query, image, and optional reference image
2. Planner outputs valid JSON matching schema in `docs/02_module_planner.md`
3. Unit tests validate JSON structure and control flag logic
4. VLM client abstraction works with at least OpenAI and Anthropic
5. Example outputs saved to `artifacts/planner/` directory
6. Configuration loading from YAML works correctly
7. Error handling covers malformed JSON and API failures

## Dependencies
- Phase 1: Environment setup (completed)
- External: OpenAI/Anthropic API keys configured
- Libraries: LangGraph, Pydantic, OpenAI SDK, Anthropic SDK

## Risks & Mitigations
- **Risk**: VLM produces invalid JSON
  - **Mitigation**: Implement retry with stricter prompts, JSON schema validation
- **Risk**: API rate limits during development
  - **Mitigation**: Local caching, support for cheaper models (GPT-4o-mini)
- **Risk**: Qwen2.5-VL local inference complexity
  - **Mitigation**: Mark as optional, use API fallback initially

## Implementation Strategy
1. Start with Pydantic models for type safety
2. Implement VLM client abstraction (OpenAI first, then others)
3. Build Planner node with prompt template
4. Add LangGraph state and graph setup
5. Create unit tests with fixture examples
6. Document example usage in artifacts/

## Testing Approach
- Unit tests: JSON schema validation, control flag logic
- Integration tests: VLM client with mocked responses
- Manual tests: Real API calls with example images from `docs/02_module_planner.md`
- Fixtures: Save example inputs/outputs for Phase 3 integration

## Related Work
- Follows architecture in `openspec/project.md`
- Complements configuration system in Phase 1
- Prepares for Executor module in Phase 3
