# Design: Phase 2 Planner Module

## Architectural Overview

### Component Hierarchy
```
src/agentic/
├── nodes/
│   └── planner.py          # Planner LangGraph node
├── state.py                # Pydantic state models
├── graph.py                # LangGraph StateGraph (initial setup)
└── vlm_client.py           # VLM abstraction layer
```

## Design Decisions

### 1. VLM Client Abstraction Pattern

**Decision**: Create unified `VLMClient` abstract base class with provider-specific implementations.

**Rationale**:
- Multiple VLM providers (OpenAI, Anthropic, Google, Qwen2.5-VL) need consistent interface
- Enables easy provider switching via configuration
- Simplifies testing with mock clients
- Future-proof for additional providers

**Trade-offs**:
- **Pro**: Clean separation of concerns, testability
- **Con**: Slight overhead for abstraction layer
- **Chosen**: Abstraction benefits outweigh minimal overhead

**Implementation**:
```python
class VLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image: Image, **kwargs) -> str:
        """Generate text response from VLM."""
        pass

class OpenAIVLMClient(VLMClient):
    # Implementation using OpenAI SDK

class AnthropicVLMClient(VLMClient):
    # Implementation using Anthropic SDK
```

### 2. Prompt Template Management

**Decision**: Store prompt template as Python string constant in `planner.py`, not external file.

**Rationale**:
- Prompt is tightly coupled to Planner logic
- Reduces file I/O and complexity
- Version control tracks prompt changes alongside code
- Easier to maintain consistency

**Trade-offs**:
- **Pro**: Simplicity, no file loading errors, version tracking
- **Con**: Less flexible for runtime prompt editing
- **Chosen**: Follow paper specification exactly, no need for runtime editing

**Template Source**:
- Paper Appendix A.2 as documented in `docs/02_module_planner.md`
- Exact text: System message starting with "You are a planner in an image quality assessment (IQA) system..."
- Complete JSON schema specification with all required fields
- Parameters: `temperature=0.0`, `top_p=0.1`, `max_tokens=2048` (as per documentation)

### 3. State Model Structure

**Decision**: Use Pydantic V2 models for all state objects with explicit field validation.

**Rationale**:
- Type safety catches errors early
- Automatic JSON serialization/deserialization
- LangGraph StateGraph requires typed state
- Schema validation for VLM outputs

**Implementation**:
```python
class PlannerOutput(BaseModel):
    query_type: Literal["IQA", "Other"]
    query_scope: Union[List[str], Literal["Global"]]
    distortion_source: Literal["Explicit", "Inferred"]
    distortions: Optional[Dict[str, List[str]]]
    reference_mode: Literal["Full-Reference", "No-Reference"]
    required_tool: Optional[str]
    plan: PlanControlFlags

class PlanControlFlags(BaseModel):
    distortion_detection: bool
    distortion_analysis: bool
    tool_selection: bool
    tool_execution: bool

class AgenticIQAState(TypedDict):
    query: str
    image_path: str
    reference_path: Optional[str]
    plan: Optional[PlannerOutput]
    # ... (executor evidence, summarizer result added in Phase 3/4)
```

### 4. Error Handling Strategy

**Decision**: Three-tier error handling: validation → retry → fallback.

**Rationale**:
- VLM outputs can be non-deterministic or malformed
- Paper doesn't specify exact error handling, so use best practices
- Balance reliability with cost (retries consume API credits)

**Implementation**:
1. **Validation**: Parse JSON with Pydantic, catch schema violations
2. **Retry**: Up to 3 attempts with stricter prompts (add "return valid JSON only")
3. **Fallback**: Switch to alternative VLM provider or raise exception

**Trade-offs**:
- **Pro**: Robust against transient failures
- **Con**: Increased latency and cost
- **Chosen**: Max 3 retries to balance reliability and cost

### 5. Configuration Integration

**Decision**: Reuse existing `configs/model_backends.yaml` structure from Phase 1.

**Rationale**:
- Consistency with established configuration system
- Environment variable interpolation already supported
- Validated with Pydantic schemas

**Extension**:
```yaml
planner:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 2048
  top_p: 0.1
  retry_attempts: 3
  fallback_backend: anthropic.claude-3.5-sonnet
```

### 6. LangGraph Integration Approach

**Decision**: Minimal LangGraph setup in Phase 2, expand in Phase 3.

**Rationale**:
- Phase 2 only implements Planner node, full graph needs Executor/Summarizer
- Premature graph complexity would complicate testing
- Incremental approach reduces risk

**Phase 2 Scope**:
- Define `AgenticIQAState` TypedDict
- Create Planner node function
- Initialize StateGraph with single node (testing only)
- Graph expansion deferred to Phase 3

**Phase 3+ Scope**:
- Add Executor and Summarizer nodes
- Implement conditional edges for replanning
- Add state persistence and checkpointing

## Alternative Approaches Considered

### Alternative 1: External Prompt Files
**Rejected**: Adds complexity without clear benefit. Prompt is static from paper specification.

### Alternative 2: Single Unified VLM Client
**Rejected**: Provider APIs differ significantly (image encoding, message formats). Abstraction layer cleaner.

### Alternative 3: Custom JSON Parsing
**Rejected**: Pydantic provides superior validation, error messages, and type safety.

### Alternative 4: Full Graph Implementation in Phase 2
**Rejected**: Violates incremental development principle. Would require stubbing Executor/Summarizer.

## Performance Considerations

### Latency
- **VLM API calls**: 2-5 seconds typical (GPT-4o vision)
- **Retry overhead**: 3x worst case (rare)
- **JSON parsing**: Negligible (<1ms)

### Cost
- **GPT-4o**: ~$0.01-0.03 per plan (depends on image size)
- **Claude 3.5**: ~$0.02-0.04 per plan
- **Mitigation**: Cache results by (query, image_hash) for duplicate requests

### Memory
- **Image loading**: PIL.Image in memory (~10-50MB per image)
- **State object**: <1KB JSON per sample
- **VLM client**: Minimal overhead

## Security Considerations

1. **API Key Management**: Already handled by Phase 1 environment setup
2. **Image Path Validation**: Validate file paths to prevent directory traversal
3. **Prompt Injection**: User query is part of prompt; VLM providers handle safety
4. **Sensitive Data**: Never log API keys; sanitize logs

## Testing Strategy

### Unit Tests
- Pydantic model validation (valid/invalid JSON)
- Control flag logic (explicit vs inferred distortions)
- VLM client mocking

### Integration Tests
- Real VLM API calls with test images
- Configuration loading end-to-end
- LangGraph node execution

### Manual Tests
- Examples from `docs/02_module_planner.md`
- Edge cases: no reference image, global scope, multiple objects

## Open Questions
1. **Qwen2.5-VL local inference**: Defer to optional feature? → Yes, use API fallback initially
2. **Prompt variations**: Support fine-tuned prompts? → No, use paper template exactly
3. **Caching strategy**: Implement now or Phase 5? → Basic image hash caching in Phase 2
4. **Logging verbosity**: JSON lines or text? → JSON lines (already configured in Phase 1)

## Success Metrics
- Planner node produces valid JSON 100% of time (with retries)
- Unit test coverage >80%
- Example outputs match `docs/02_module_planner.md` expected format
- Configuration loading works with Phase 1 YAML
