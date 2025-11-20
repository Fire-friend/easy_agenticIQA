# Tasks: Implement Phase 2 - Planner Module

## Prerequisites
- [x] Phase 1 environment setup completed and validated
- [x] `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` configured in environment
- [x] `configs/model_backends.yaml` and `configs/pipeline.yaml` exist

## Task List

### 1. Implement Pydantic State Models ✅
**Capability**: state-models
**Estimated Time**: 2-3 hours
**Dependencies**: None (foundational)

- [x] Create `src/agentic/state.py` file
- [x] Define `PlanControlFlags` model with 4 boolean fields
- [x] Define `PlannerOutput` model with all required fields from spec
  - Add field validators for `query_type`, `query_scope`, `distortion_source`, `reference_mode`
  - Include JSON schema examples in `model_config`
- [x] Define `PlannerInput` model with file path validation
  - Add validator to check file existence for `image_path` and `reference_path`
- [x] Define `AgenticIQAState` TypedDict for LangGraph
  - Include Phase 2 fields: `query`, `image_path`, `reference_path`, `plan`, `error`
  - Add comments for future Phase 3/4 fields
- [x] Define `PlannerError` model for error tracking
- [x] Write unit tests: `tests/test_state_models.py`
  - Test valid and invalid JSON parsing
  - Test field validation rules
  - Test serialization round-trip
- [x] Verify: Run tests and ensure 100% pass rate (22/22 tests passed)

### 2. Implement VLM Client Abstraction
**Capability**: vlm-integration
**Estimated Time**: 4-5 hours
**Dependencies**: Task 1 (state models for type hints)

- [x] Create `src/agentic/vlm_client.py` file
- [x] Define abstract `VLMClient` base class
  - Abstract method: `generate(prompt: str, images: List[Image], **kwargs) -> str`
  - Properties: `backend_name`, `supports_vision`
- [x] Implement `OpenAIVLMClient` class
  - Initialize with API key and optional base URL
  - Implement image to base64 encoding
  - Implement multimodal message construction
  - Handle API errors (auth, rate limit, timeout)
- [x] Implement `AnthropicVLMClient` class
  - Initialize with API key and optional base URL
  - Implement image to base64 with media type detection
  - Implement content blocks construction
  - Handle API errors
- [x] Implement `GoogleVLMClient` class (optional for Phase 2)
  - Can be stubbed and marked as "TODO" if time-constrained
- [x] Implement `create_vlm_client(backend: str, config: dict)` factory function
  - Parse backend identifier (e.g., "openai.gpt-4o")
  - Instantiate appropriate client class
  - Handle unsupported backends with clear error
- [x] Implement image loading utility: `load_image(path: str) -> Image`
  - Validate file format
  - Convert to RGB if needed
  - Optionally resize large images
- [x] Write unit tests: `tests/test_vlm_client.py`
  - Mock API calls for testing
  - Test image encoding functions
  - Test error handling for each client
  - Test factory function with various backends
- [x] Verify: Run tests and ensure all VLM clients work with mocked APIs

### 3. Implement Planner Core Logic
**Capability**: planner-core
**Estimated Time**: 3-4 hours
**Dependencies**: Tasks 1, 2 (state models and VLM clients)

- [x] Create `src/agentic/nodes/planner.py` file
- [x] Define `PLANNER_PROMPT_TEMPLATE` constant from `docs/02_module_planner.md` (paper Appendix A.2)
  - Use the exact template text specified in the documentation
  - System message: "You are a planner in an image quality assessment (IQA) system..."
  - Include complete JSON schema specification with all fields
  - Include placeholders for user query and images: `{query}` and `<image>`
- [x] Implement `construct_planner_prompt(query: str, has_reference: bool) -> str`
  - Format template with query
  - Adjust prompt based on reference availability
- [x] Implement `parse_planner_output(json_str: str) -> PlannerOutput`
  - Try parsing as JSON
  - Validate with Pydantic `PlannerOutput` model
  - Raise descriptive errors on failure
- [x] Implement `planner_node(state: AgenticIQAState) -> Dict[str, Any]`
  - Load configuration from `configs/model_backends.yaml` (reuse Phase 1 utilities)
  - Create VLM client via factory
  - Load image(s) from paths in state
  - Construct prompt
  - Call VLM client with retry logic (up to 3 attempts)
  - Parse and validate output
  - Return state update with `plan` field
  - Handle errors and populate `error` field if needed
- [x] Write unit tests: `tests/test_planner.py`
  - Test prompt construction
  - Test JSON parsing (valid and invalid cases)
  - Test planner_node with mocked VLM client
  - Test retry logic and error handling
- [x] Verify: Run tests and ensure Planner logic works end-to-end with mocks

### 4. Implement LangGraph Setup
**Capability**: langgraph-setup
**Estimated Time**: 2-3 hours
**Dependencies**: Tasks 1, 3 (state models and planner node)

- [x] Create `src/agentic/graph.py` file
- [x] Implement `create_agentic_graph() -> StateGraph`
  - Initialize `StateGraph(AgenticIQAState)`
  - Add Planner node: `graph.add_node("planner", planner_node)`
  - Set entry point: `graph.set_entry_point("planner")`
  - Add END edge: `graph.add_edge("planner", END)`
  - Return graph instance
- [x] Implement `compile_graph(graph: StateGraph, config: dict) -> CompiledGraph`
  - Load graph settings from `configs/pipeline.yaml`
  - Optionally configure checkpointer (MemorySaver for Phase 2)
  - Compile graph
  - Return compiled graph
- [x] Implement `run_pipeline(query: str, image_path: str, reference_path: Optional[str] = None) -> AgenticIQAState`
  - Create initial state
  - Load and compile graph
  - Invoke graph with initial state
  - Return final state
- [x] Add graph visualization utility: `visualize_graph(graph: StateGraph, output_path: str)`
  - Export to Mermaid format
  - Save to file if path provided
- [x] Write integration tests: `tests/test_graph.py`
  - Test graph creation and compilation
  - Test pipeline execution with mocked Planner
  - Test state flow through graph
- [x] Verify: Run integration tests and ensure graph executes correctly

### 5. Create Example Artifacts and Manual Testing
**Capability**: planner-core (validation)
**Estimated Time**: 1-2 hours
**Dependencies**: Tasks 1-4 (complete implementation)

- [x] Create `artifacts/planner/` directory
- [x] Prepare test images: Copy or download sample images with distortions
  - Example: blurry car image, noisy portrait, etc.
- [x] Create manual test script: `scripts/test_planner_manual.py`
  - Load test images
  - Run Planner with real VLM API calls
  - Save outputs to `artifacts/planner/examples.jsonl`
- [x] Execute manual tests with real API:
  - Test case 1: "Is the vehicle blurry?" (explicit distortion)
  - Test case 2: "What's wrong with this image?" (inferred distortion)
  - Test case 3: Full-Reference query with reference image
  - Test case 4: Global quality query
- [x] Review outputs and verify:
  - JSON structure matches schema
  - Control flags are logical given query type
  - Distortion identification is reasonable
- [x] Save successful examples to artifacts/
- [x] Document any discrepancies or unexpected behaviors

### 6. Documentation and Configuration Updates
**Capability**: Cross-cutting
**Estimated Time**: 1-2 hours
**Dependencies**: Tasks 1-5 (implementation complete)

- [x] Update `configs/model_backends.yaml` if needed
  - Add fallback_backend option
  - Document configuration options in comments
- [x] Update `configs/pipeline.yaml` if needed
  - Add planner-specific settings (retry_attempts, timeout)
- [x] Create usage example: `examples/planner_usage.py`
  - Show how to initialize and use Planner standalone
  - Include error handling examples
- [x] Update project README or documentation (if exists)
  - Note: Only if documentation already exists; don't create new docs proactively
- [x] Verify: Configuration files are valid YAML and load correctly

### 7. Testing and Validation
**Capability**: Cross-cutting
**Estimated Time**: 1-2 hours
**Dependencies**: Tasks 1-6 (all implementation)

- [x] Run full test suite: `pytest tests/ -v`
  - Ensure all unit tests pass
  - Ensure all integration tests pass
  - Check test coverage: `pytest --cov=src/agentic`
  - Target: >80% coverage for Phase 2 code
- [x] Run environment validation: `python scripts/check_env.py`
  - Ensure all dependencies are installed
  - Verify API keys are configured
- [x] Run end-to-end manual test:
  - `python scripts/test_planner_manual.py`
  - Review outputs in `artifacts/planner/`
  - Verify outputs match expected schema
- [x] Test error scenarios:
  - Invalid image path
  - Missing API key
  - Malformed VLM response (simulate with mock)
- [x] Verify: All tests pass, artifacts are generated correctly

### 8. Code Review and Cleanup
**Capability**: Cross-cutting
**Estimated Time**: 1 hour
**Dependencies**: Tasks 1-7 (implementation and testing complete)

- [x] Review code for adherence to project conventions:
  - Minimize file creation (reuse existing structure)
  - Avoid unnecessary optionality
  - No fallback mechanisms unless required
  - Chinese comments allowed if helpful
- [x] Check for TODO comments and resolve or document
- [x] Remove any debug print statements
- [x] Ensure consistent code formatting (use black if configured)
- [x] Verify type hints are complete and accurate
- [x] Run linter if configured: `ruff check src/` or `flake8 src/`
- [x] Verify: Code is clean, well-documented, and follows conventions

## Completion Checklist

- [x] All 8 tasks completed
- [x] All unit and integration tests pass
- [x] Manual tests with real VLM API successful
- [x] Example outputs saved to `artifacts/planner/`
- [x] Configuration files validated
- [x] No critical TODOs remaining
- [x] Code adheres to project conventions

## Validation Commands

```bash
# 1. Run all tests
pytest tests/ -v --cov=src/agentic

# 2. Validate environment
python scripts/check_env.py

# 3. Manual Planner test
python scripts/test_planner_manual.py

# 4. Check configuration loading
python -c "from src.utils.config import load_model_backends; print(load_model_backends())"

# 5. Validate LangGraph setup
python -c "from src.agentic.graph import create_agentic_graph; graph = create_agentic_graph(); print('Graph created:', graph)"
```

## Notes

- **Parallel Work**: Tasks 1 and 2 can be implemented in parallel (state models and VLM clients are independent)
- **Incremental Testing**: Test each task immediately after implementation, don't wait until the end
- **API Costs**: Use mocked tests during development, only use real API for manual validation
- **Qwen2.5-VL**: Optional for Phase 2, can be implemented in a future phase if needed
- **Debugging**: Enable `debug_mode: true` in `configs/pipeline.yaml` for verbose logging during development
