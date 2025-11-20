# Design: Phase 5 Inference Pipeline

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     Batch Pipeline Runner                    │
│  (run_pipeline.py - CLI entry point)                         │
└──────────────┬──────────────────────────────────────────────┘
               │
               ├──> Configuration Manager
               │    ├─ Load pipeline.yaml
               │    ├─ Parse CLI overrides
               │    └─ Validate settings
               │
               ├──> Dataset Loader
               │    ├─ Read JSONL manifests
               │    ├─ Skip processed samples (resume)
               │    └─ Batch iterator
               │
               ├──> Sample Processor
               │    ├─ Call run_pipeline() from graph.py
               │    ├─ Wrap with error handling
               │    └─ Track timing/metrics
               │
               ├──> Result Writer
               │    ├─ Serialize to JSONL
               │    ├─ Write after each sample (checkpoint)
               │    └─ Save artifacts (optional)
               │
               └──> Logger & Metrics
                    ├─ Structured JSON Lines logs
                    ├─ Cost tracking (token usage)
                    └─ Progress reporting
```

### Data Flow

```
Input JSONL → Batch Loader → For each sample:
                              ├─> run_pipeline(sample)
                              │   └─> LangGraph: Planner→Executor→Summarizer
                              ├─> Extract results
                              ├─> Log metrics
                              └─> Write JSONL output
```

## Key Design Decisions

### 1. Resume Capability
**Decision**: Check existing output file and skip already-processed samples
**Rationale**: API calls are expensive and slow; interruptions are common
**Implementation**: 
- Read output JSONL at startup
- Build set of processed sample_ids
- Filter input samples by this set

### 2. Structured Logging
**Decision**: JSON Lines format for execution logs
**Rationale**: Machine-parseable, easy to analyze, supports streaming
**Schema**:
```json
{
  "timestamp": "2025-10-30T19:45:00Z",
  "sample_id": "sample_001",
  "query": "...",
  "execution_time_ms": 12450,
  "planner_backend": "openai.gpt-4o",
  "executor_backend": "qwen2.5-vl-local",
  "summarizer_backend": "openai.gpt-4o",
  "replan_count": 1,
  "tokens_used": {"planner": 245, "executor": 1024, "summarizer": 512},
  "cost_usd": 0.034,
  "status": "success|error",
  "error_details": null
}
```

### 3. Result Format
**Decision**: Single JSONL file with complete state per sample
**Rationale**: Self-contained, supports streaming writes, easy to inspect
**Schema**:
```json
{
  "sample_id": "sample_001",
  "query": "...",
  "image_path": "...",
  "reference_path": null,
  "plan": {...},
  "executor_evidence": {...},
  "summarizer_result": {...},
  "metadata": {
    "execution_time_ms": 12450,
    "replan_count": 1,
    "final_status": "success"
  }
}
```

### 4. Error Handling Strategy
**Approach**: Three-tier fallback
1. **Retry with same backend**: Up to 3 attempts with exponential backoff
2. **Fallback model**: Switch to cheaper/alternative model if available
3. **Graceful degradation**: Save partial results with error flag

**Errors to Handle**:
- API rate limits (429) → Wait and retry
- Authentication errors (401) → Fail fast
- Timeout errors → Retry with longer timeout
- Tool execution failures → Use fallback tool or mark as unavailable

### 5. Caching Strategy
**Tool Results**:
- Cache key: `{tool_name}_{image_hash}_{params_hash}`
- Location: `{cache_dir}/tool_results/{tool_name}/{image_hash}.json`
- TTL: Indefinite (tool outputs are deterministic)

**API Responses** (Optional):
- Cache key: `{backend}_{prompt_hash}_{image_hash}`
- Location: `{cache_dir}/api_cache/{backend}/{hash}.json`
- TTL: Configurable (default: disabled for evaluation reproducibility)

### 6. CLI Interface Design
**Command Structure**:
```bash
run_pipeline.py \
  --config CONFIG_PATH \
  --input INPUT_JSONL \
  --output OUTPUT_JSONL \
  [--resume] \
  [--max-samples N] \
  [--backend-override KEY=VALUE] \
  [--verbose]
```

**Configuration Override**:
- CLI args > Environment variables > Config file
- Supports dot notation: `--backend-override planner.backend=openai.gpt-4o-mini`

## Performance Considerations

### Concurrency
- **VLM calls**: Sequential (API rate limits)
- **Tool execution**: Parallel per distortion (thread pool)
- **File I/O**: Async writes for non-blocking

### Cost Optimization
- Cache tool results aggressively
- Use cheaper models for debugging (`--backend-override`)
- Support dry-run mode for prompt validation

### Memory Management
- Stream input JSONL (don't load entire dataset)
- Write outputs immediately after each sample
- Limit tool result cache size (LRU eviction)

## Testing Strategy

### Unit Tests
- JSONL parsing/writing
- Resume logic (skip processed samples)
- Error handling and fallback
- Cost calculation

### Integration Tests
- End-to-end with 5-sample mini-dataset
- Interruption and resume
- Error injection (mock API failures)
- Output format validation

### Evaluation Tests
- MCQ accuracy calculation
- SRCC/PLCC correlation metrics
- Report generation

## Alternative Designs Considered

### Alternative 1: Database Backend
**Rejected**: Too heavy for research reproduction use case
**Rationale**: JSONL is simpler, portable, and sufficient

### Alternative 2: Async/Concurrent Batch Processing
**Rejected**: API rate limits make concurrency difficult
**Rationale**: Sequential processing is safer and easier to debug

### Alternative 3: Separate CLI Tools per Phase
**Rejected**: Users want end-to-end pipeline
**Rationale**: Single tool is more user-friendly, supports replanning
