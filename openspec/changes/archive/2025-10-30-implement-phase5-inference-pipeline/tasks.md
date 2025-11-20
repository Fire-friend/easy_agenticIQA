# Tasks: Implement Phase 5 Inference Pipeline

## Overview
This change implements the batch inference pipeline for AgenticIQA, enabling dataset-scale evaluation with robust error handling, structured logging, and evaluation tools.

## Task List

### 1. Batch Processing Infrastructure (Core)
**Priority**: P0 - Critical path
**Estimated complexity**: Medium
**Dependencies**: None (uses existing graph.py)

#### 1.1 Create `run_pipeline.py` main script
- [x] Setup Typer CLI interface with argument parsing
- [x] Implement configuration loading (YAML + CLI overrides)
- [x] Add JSONL input reader with streaming
- [x] Create batch processing loop
- [x] **Validation**: Run with 5-sample test dataset

#### 1.2 Implement resume capability
- [x] Read existing output JSONL and extract processed sample_ids
- [x] Filter input samples to skip processed ones
- [x] Add `--resume` flag to enable/disable
- [x] **Validation**: Interrupt run, restart with --resume, verify no duplicates

#### 1.3 Add progress tracking
- [x] Integrate Rich progress bar
- [x] Display: current/total samples, ETA, success/error count
- [x] Log periodic progress summaries
- [x] **Validation**: Visual inspection of progress display

### 2. Result Management
**Priority**: P0 - Critical path
**Estimated complexity**: Low
**Dependencies**: Task 1.1

#### 2.1 Implement JSONL output writer
- [x] Define output schema (sample_id, query, plan, evidence, result, metadata)
- [x] Create ResultWriter class with atomic writes
- [x] Write after each sample (checkpoint strategy)
- [x] **Validation**: Validate output against schema, check atomicity

#### 2.2 Add artifact management (optional)
- [ ] Create artifacts directory structure
- [ ] Save intermediate visualizations (if enabled)
- [ ] Implement cleanup policy (max age/size)
- [ ] **Validation**: Check artifact files are created correctly

### 3. Structured Logging
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: Task 1.1

#### 3.1 Create execution logger
- [x] Define log schema (timestamp, sample_id, backends, timing, tokens, cost)
- [x] Implement JSON Lines log writer
- [x] Add log rotation (by size or date)
- [x] **Validation**: Parse logs and verify schema compliance

#### 3.2 Add metrics tracking
- [x] Track token usage per agent (planner, executor, summarizer)
- [x] Calculate cost estimates (configurable rate card)
- [x] Log replanning statistics
- [x] **Validation**: Compare tracked metrics against API usage

#### 3.3 Implement logging levels
- [x] INFO: Summary per sample
- [x] DEBUG: Include prompts and responses
- [x] TRACE: Full state dumps
- [x] **Validation**: Test each level, verify output volume

### 4. Error Handling & Recovery
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: Task 1.1

#### 4.1 Implement retry logic
- [ ] Add exponential backoff for rate limits (429 errors)
- [ ] Retry up to 3 times for transient errors
- [ ] Fast-fail for authentication errors (401, 403)
- [ ] **Validation**: Mock API errors, verify retry behavior

#### 4.2 Add model fallback
- [ ] Define fallback model chains (e.g., GPT-4o → GPT-4o-mini)
- [ ] Implement automatic fallback on failure
- [ ] Log fallback events
- [ ] **Validation**: Force primary model failure, verify fallback

#### 4.3 Graceful degradation
- [ ] Continue processing on tool execution failures
- [ ] Save partial results with error flags
- [ ] Generate summary of failed samples
- [ ] **Validation**: Inject tool failures, verify partial results saved

### 5. CLI Interface
**Priority**: P0 - Critical path
**Estimated complexity**: Low
**Dependencies**: Tasks 1-4

#### 5.1 Implement CLI arguments
- [x] `--config PATH`: Path to pipeline.yaml
- [x] `--input PATH`: Input JSONL manifest
- [x] `--output PATH`: Output JSONL results
- [x] `--resume`: Resume from interruption
- [x] `--max-samples N`: Limit processing
- [x] `--verbose`: Enable debug logging
- [x] **Validation**: Test each argument combination

#### 5.2 Add configuration override
- [x] `--backend-override KEY=VALUE`: Override config values
- [x] Support dot notation (e.g., `planner.backend=...`)
- [x] Merge with config file values
- [x] **Validation**: Override backend, verify correct model used

#### 5.3 Add dry-run mode
- [x] `--dry-run`: Validate config and data without execution
- [x] Print execution plan (sample count, estimated cost)
- [x] Validate file paths and permissions
- [x] **Validation**: Run dry-run, verify no execution

### 6. Evaluation Scripts
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: Task 2 (needs output format)

#### 6.1 Create `scripts/eval_mcq_accuracy.py`
- [ ] Load output JSONL with MCQ results
- [ ] Extract final_answer and ground truth
- [ ] Calculate accuracy per question type
- [ ] Generate accuracy report
- [ ] **Validation**: Test with known ground truth, verify 100% accuracy

#### 6.2 Create `scripts/eval_correlation.py`
- [ ] Load output JSONL with quality scores
- [ ] Extract predicted and ground truth MOS
- [ ] Calculate SRCC (Spearman rank correlation)
- [ ] Calculate PLCC (Pearson linear correlation)
- [ ] **Validation**: Test with synthetic data, verify correlation values

#### 6.3 Create `scripts/generate_report.py`
- [ ] Aggregate results from multiple runs
- [ ] Generate markdown report with tables and charts
- [ ] Include: accuracy, SRCC, PLCC, cost, timing
- [ ] **Validation**: Generate report from test results, verify format

### 7. Integration Tests
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: All previous tasks

#### 7.1 Create end-to-end test
- [ ] Prepare 10-sample test dataset (FR + NR + MCQ)
- [ ] Run full pipeline
- [ ] Validate output format and completeness
- [ ] **Validation**: pytest test with assertions

#### 7.2 Create resume test
- [ ] Process 5 samples
- [ ] Interrupt (simulate crash)
- [ ] Resume and process remaining 5
- [ ] Verify total 10 samples, no duplicates
- [ ] **Validation**: Automated test

#### 7.3 Create error handling test
- [ ] Mock API failures at various stages
- [ ] Verify retry and fallback behavior
- [ ] Check partial results saved correctly
- [ ] **Validation**: pytest with mocked APIs

### 8. Documentation & Examples
**Priority**: P2 - Nice to have
**Estimated complexity**: Low
**Dependencies**: All implementation tasks

#### 8.1 Update README with usage examples
- [ ] Add batch processing example
- [ ] Document CLI arguments
- [ ] Add troubleshooting section
- [ ] **Validation**: Follow examples, verify they work

#### 8.2 Create example datasets
- [ ] Prepare 5-sample AgenticIQA-Eval mini-dataset
- [ ] Prepare 10-sample TID2013 subset
- [ ] Include expected outputs for validation
- [ ] **Validation**: Run pipeline on examples, compare outputs

## Task Dependencies

```
1.1 (CLI + Batch Loop) ────┬──> 1.2 (Resume) ──┐
                            ├──> 1.3 (Progress) ┤
                            └──> 2.1 (Output) ───┼──> 5.1 (CLI Args) ──┐
                                                  │                      │
3.1 (Logging) ──> 3.2 (Metrics) ──> 3.3 (Levels)┤                      │
                                                  │                      ├──> 7.1 (E2E Test)
4.1 (Retry) ──> 4.2 (Fallback) ──> 4.3 (Degrade)┤                      │
                                                  │                      │
2.1 (Output) ──────────────────────────> 6.1 (MCQ) ──> 6.3 (Report) ──┘
                                          6.2 (Corr) ──┘
```

## Parallel Work Opportunities
- Tasks 3 (Logging) and 4 (Error Handling) can be developed in parallel
- Tasks 6 (Evaluation) can start once output format is defined (Task 2.1)
- Task 8 (Documentation) can be done throughout

## Estimated Timeline
- **Core batch processing (Tasks 1-2)**: 2-3 days
- **Logging & error handling (Tasks 3-4)**: 2-3 days
- **CLI & evaluation (Tasks 5-6)**: 2 days
- **Testing & documentation (Tasks 7-8)**: 1-2 days
- **Total**: ~1 week with focused effort

## Definition of Done
- [ ] All P0 and P1 tasks completed
- [ ] All tests passing (unit + integration)
- [ ] Can process 100+ sample dataset without manual intervention
- [ ] Evaluation scripts produce correct metrics
- [ ] Documentation updated with examples
- [ ] Code reviewed and merged
