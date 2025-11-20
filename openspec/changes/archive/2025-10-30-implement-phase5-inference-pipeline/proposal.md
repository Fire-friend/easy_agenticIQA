# Proposal: Implement Phase 5 Inference Pipeline

## Summary
Implement the batch inference pipeline and system integration layer for AgenticIQA, enabling end-to-end evaluation on datasets with structured logging, caching, error handling, and evaluation scripts.

## Context
Phase 4 completed the core Planner→Executor→Summarizer workflow with replanning capability. The LangGraph orchestration (`src/agentic/graph.py`) successfully runs single-sample inference through `run_pipeline()`. However, the system lacks:

1. **Batch processing infrastructure** to process entire datasets (AgenticIQA-Eval, TID2013, BID, AGIQA-3K)
2. **Structured logging** to track execution metrics (timing, tokens, costs, replanning)
3. **Result persistence** in JSONL format with intermediate artifacts
4. **CLI interface** for configuration override and batch execution
5. **Evaluation scripts** to calculate MCQ accuracy, SRCC/PLCC metrics
6. **Error recovery** mechanisms for production-grade robustness

This phase transforms the working prototype into a complete evaluation system ready for paper reproduction.

## Why

### User Impact
Researchers and paper reviewers need to evaluate AgenticIQA on standard benchmarks (AgenticIQA-Eval, TID2013, BID, AGIQA-3K) to verify the paper's claims. Currently, `run_pipeline()` only processes single samples, requiring manual scripting for batch evaluation. This creates barriers to adoption and reproduction.

### Business Value
- **Paper Reproduction**: Enables independent verification of published results
- **Research Velocity**: Researchers can evaluate modifications quickly without custom scripting
- **Cost Transparency**: Detailed logging enables budget planning for API-based evaluations
- **Reliability**: Error recovery ensures completion of long-running evaluations

### Technical Motivation
The existing `graph.py` provides the core inference logic but lacks production-grade orchestration:
- No batch processing → Manual iteration required
- No checkpointing → Full restart on interruption
- No structured logging → Difficult to analyze performance/costs
- No evaluation tools → Manual metric calculation
- Minimal error handling → Single API failure aborts entire run

## Goals
1. **Batch Processing**: Process entire datasets with resume capability and progress tracking
2. **Production Logging**: Structured JSON Lines logs with execution metrics
3. **Result Management**: JSONL output format with all intermediate results (plan, evidence, final answer)
4. **CLI Interface**: User-friendly command-line interface with Typer
5. **Evaluation Tools**: Scripts to calculate MCQ accuracy and correlation metrics
6. **Error Resilience**: Retry logic, model fallback, graceful degradation

## Non-Goals
- REST API or web interface (future work)
- Real-time visualization dashboard (debugging only)
- Custom dataset preprocessing (assumes processed JSONL manifests exist)
- Fine-tuning or model training (inference-only reproduction)

## Stakeholders
- **End Users**: Researchers running AgenticIQA evaluation and reproduction
- **Developers**: Contributors extending the pipeline or integrating new tools
- **Paper Reviewers**: Verifying reproduction correctness against paper results

## Risks & Mitigation
- **Risk**: API rate limits causing batch failures
  - **Mitigation**: Configurable retry delays, exponential backoff, resume capability
- **Risk**: Incomplete results due to tool failures
  - **Mitigation**: Fallback tools, graceful degradation, detailed error logs
- **Risk**: Cost overruns on large datasets
  - **Mitigation**: Cost tracking, cache-first execution, configurable API usage limits

## Success Criteria
- [ ] `run_pipeline.py` successfully processes 100+ sample dataset
- [ ] JSONL output includes all intermediate results (plan, evidence, final answer)
- [ ] Resume capability works after interruption (checkpoint-based)
- [ ] Evaluation scripts correctly calculate MCQ accuracy and SRCC/PLCC
- [ ] Structured logs enable cost/performance analysis
- [ ] End-to-end pipeline test passes with all phases

## Related Changes
- Depends on: `implement-phase5-summarizer-module` (archived)
- Enables: Future evaluation protocol optimization (Phase 6)
