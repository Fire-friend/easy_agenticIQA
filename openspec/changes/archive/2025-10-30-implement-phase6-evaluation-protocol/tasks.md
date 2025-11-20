# Tasks: Implement Phase 6 Evaluation Protocol

This document breaks down Phase 6 implementation into ordered, verifiable tasks across four major work streams.

## Work Streams

1. **Data Preparation** - Manifest schemas and generation scripts (P0)
2. **Statistical Evaluation** - Bootstrap CI, confusion matrices, breakdowns (P0)
3. **Report Generation** - Automated reproduction reports (P1)
4. **Qualitative Analysis** - Case selection and export (P1)

---

## 1. Data Preparation Pipeline
**Priority**: P0 - Critical path
**Estimated complexity**: Medium
**Dependencies**: None

### 1.1 Create manifest schemas
- [x] Define JSON Schema for TID2013 (sample_id, paths, MOS, distortion metadata)
- [x] Define JSON Schema for BID (sample_id, path, MOS, split)
- [x] Define JSON Schema for AGIQA-3K (sample_id, path, MOS, generation metadata)
- [x] Define JSON Schema for AgenticIQA-Eval (question, options, answer, task_type, reference_mode)
- [x] Create common fields schema (shared field definitions)
- [x] **Validation**: Validate example manifests against each schema

### 1.2 Implement manifest generators
- [x] Script: `scripts/data_prep/generate_tid2013_manifest.py`
  - Read TID2013 directory structure (distorted/reference images)
  - Parse mos.csv or annotations
  - Map to schema format
  - Write JSONL manifest
- [x] Script: `scripts/data_prep/generate_bid_manifest.py`
  - Parse BID annotations
  - Handle NR-IQA (no reference images)
  - Map to schema format
- [x] Script: `scripts/data_prep/generate_agiqa3k_manifest.py`
  - Parse AGIQA-3K metadata
  - Map generated images to MOS scores
  - Handle generation-specific metadata
- [x] Script: `scripts/data_prep/generate_agenticiqa_eval_manifest.py`
  - Parse MCQ questions from official format
  - Split by task type (planner/executor/summarizer)
  - Include reference_mode flag
- [x] **Validation**: Run each generator on sample data, verify output

### 1.3 Implement manifest validation
- [x] Script: `scripts/data_prep/validate_manifest.py`
  - Load JSON Schema
  - Validate each line in JSONL
  - Check file paths exist
  - Verify MOS ranges
  - Report errors with line numbers
- [x] Add `--strict` mode (fail on any error)
- [ ] Add `--fix` mode (attempt automatic corrections)
- [x] **Validation**: Validate generated manifests, ensure zero errors

---

## 2. Enhanced Statistical Evaluation
**Priority**: P0 - Critical path
**Estimated complexity**: Medium
**Dependencies**: Task 1 (for test data)

### 2.1 Implement bootstrap confidence intervals
- [x] Script: `scripts/eval_with_ci.py`
  - Function: `bootstrap_correlation(pred, mos, n_bootstrap=1000)`
  - Calculate SRCC with bootstrap CI
  - Calculate PLCC with bootstrap CI
  - Report p-values
  - Add multiprocessing for performance
- [x] CLI interface with arguments: `--input`, `--ground-truth`, `--n-bootstrap`, `--confidence`
- [x] JSON output format with CI bounds
- [x] **Validation**: Compare CI with scipy.stats parametric methods, verify coverage

### 2.2 Implement confusion matrix analysis
- [x] Enhance `scripts/eval_mcq_accuracy.py` or create `scripts/eval_mcq_with_confusion.py`
  - Generate confusion matrix (sklearn or manual)
  - Calculate per-option precision/recall
  - Identify most confused option pairs
  - Export matrix as JSON and ASCII table
- [ ] Add visualization (matplotlib/seaborn heatmap, optional)
- [x] **Validation**: Verify matrix sum equals total samples, diagonal = correct predictions

### 2.3 Implement per-category breakdown
- [x] Function: `eval_by_category(results_df, group_by='task_type')`
  - Group results by task_type, distortion_type, reference_mode, question_type
  - Calculate accuracy/SRCC/PLCC per group
  - Report sample counts per group
  - Export as JSON and Markdown table
- [ ] Add statistical significance tests between groups (chi-square for MCQ, t-test for correlations)
- [x] **Validation**: Verify group counts sum to total, no missing categories

### 2.4 Add paper result comparison
- [ ] Create `data/paper_results/baseline.json` with paper-reported metrics
  - AgenticIQA-Eval accuracies by task type
  - TID2013/BID/AGIQA-3K SRCC/PLCC
  - Reported model configurations
- [ ] Function: `compare_with_paper(our_results, paper_results)`
  - Calculate deltas (our - paper)
  - Identify statistically significant differences
  - Report which results match/deviate
- [ ] **Validation**: Load paper results, compute deltas, verify format

---

## 3. Automated Report Generation
**Priority**: P1 - High priority
**Estimated complexity**: High
**Dependencies**: Tasks 1, 2

### 3.1 Implement result aggregation
- [x] Script: `scripts/generate_report.py`
  - Function: `collect_results(output_dir)` - Load all outputs/*.jsonl
  - Function: `aggregate_metrics(results)` - Compute summary statistics
  - Handle missing files gracefully
  - Cache intermediate aggregations
- [x] **Validation**: Run on test outputs, verify all results loaded

### 3.2 Implement report formatting
- [x] Function: `format_environment_section()` - System info, dependencies, timestamps
- [ ] Function: `format_dataset_status()` - Dataset tables with sample counts
- [x] Function: `format_mcq_results()` - Accuracy tables with CI and deltas
- [x] Function: `format_correlation_results()` - SRCC/PLCC tables with CI and p-values
- [ ] Function: `format_confusion_matrix()` - ASCII or Markdown matrix
- [ ] Function: `format_qualitative_cases()` - Case summaries with links
- [ ] Function: `format_cost_analysis()` - Token usage and cost breakdown
- [x] **Validation**: Generate report on sample data, verify Markdown syntax

### 3.3 Implement report assembly
- [x] Function: `write_report(output_path, sections)`
  - Assemble all sections into final Markdown
  - Add table of contents
  - Include execution timestamp
  - Link to qualitative case directories
- [x] CLI interface: `--output`, `--template`, `--include-cases`
- [x] **Validation**: Generate full report, verify readability and formatting

### 3.4 Add alternative model comparison
- [ ] Function: `compare_models(results_by_model)` - Compare performance across backends
  - Load results from multiple runs with different backends
  - Generate comparison tables
  - Highlight best/worst performers
- [ ] Document model configurations (temperature, max_tokens, etc.)
- [ ] **Validation**: Run with 2+ model backends, verify comparison table

---

## 4. Qualitative Analysis Tools
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: Tasks 1, 2

### 4.1 Implement case selection
- [ ] Script: `scripts/qualitative/select_cases.py`
  - Function: `select_success_cases(results, n=5)` - High confidence + correct
  - Function: `select_failure_cases(results, n=5)` - High confidence + wrong
  - Function: `select_edge_cases(results, n=5)` - Low confidence or replanning
  - Ensure coverage of all task types and distortion types
- [ ] CLI interface: `--input`, `--output-dir`, `--n-cases`
- [ ] **Validation**: Select cases from test data, verify diversity

### 4.2 Implement state export
- [ ] Script: `scripts/qualitative/export_case.py`
  - Function: `export_case(sample_id, result, output_dir)`
    - Create case directory: `qualitative_cases/<sample_id>/`
    - Write metadata.json (sample info + ground truth)
    - Write plan.json (planner output)
    - Write evidence.json (executor output)
    - Write result.json (summarizer output)
    - Generate analysis.md (human-readable summary)
    - Symlink images (original + reference)
- [ ] **Validation**: Export sample case, verify all files created

### 4.3 Implement error taxonomy
- [ ] Define error categories (planning, tool selection, tool execution, reasoning, integration, reference mode)
- [ ] Function: `classify_error(case)` - Assign error category based on heuristics
  - Planning error: Wrong distortion in plan
  - Tool selection error: Inappropriate tools chosen
  - Tool execution error: Tool failed or invalid score
  - Reasoning error: Wrong conclusion from evidence
- [ ] Add manual override file for corrections
- [ ] **Validation**: Classify sample errors, verify category assignments

### 4.4 Implement case comparison
- [ ] Function: `compare_with_paper_example(our_case, paper_case)`
  - Match sample_id or image hash
  - Compare plans, evidence, final answers
  - Highlight differences
- [ ] Document paper examples in `data/paper_examples/`
- [ ] **Validation**: Compare with 3+ paper examples, verify diff output

---

## 5. Documentation & Integration
**Priority**: P2 - Nice to have
**Estimated complexity**: Low
**Dependencies**: All above tasks

### 5.1 Update README
- [ ] Add section: "Evaluation Protocol"
- [ ] Document manifest generation workflow
- [ ] Provide examples for all scripts
- [ ] Link to reproduction reports

### 5.2 Create evaluation guide
- [ ] Document: `docs/evaluation_guide.md`
  - Step-by-step reproduction procedure
  - Expected outputs at each stage
  - Troubleshooting common issues
  - Alternative model strategies

### 5.3 Add example datasets
- [ ] Create `data/examples/` with small test datasets
  - 10 samples from each dataset type
  - Include ground truth
  - Validated manifests
- [ ] Document how to obtain full datasets

---

## 6. Testing & Validation
**Priority**: P1 - High priority
**Estimated complexity**: Medium
**Dependencies**: All above tasks

### 6.1 Integration tests
- [ ] Test: End-to-end evaluation workflow
  - Generate manifests → Run pipeline → Compute metrics → Generate report
  - Verify all outputs created
  - Check report completeness
- [ ] Test: Bootstrap CI stability
  - Run multiple times, verify CI overlap
  - Check coverage probability (~95%)

### 6.2 Schema validation tests
- [ ] Test: Valid manifests pass validation
- [ ] Test: Invalid manifests fail with clear errors
- [ ] Test: Edge cases (missing fields, wrong types, invalid paths)

### 6.3 Report generation tests
- [ ] Test: Report generated from sample data
- [ ] Test: All sections present
- [ ] Test: Tables formatted correctly
- [ ] Test: Markdown lints successfully

---

## Task Summary

**Total tasks**: 48
- P0 (Critical): 16 tasks (Data prep, Statistical eval)
- P1 (High): 26 tasks (Report gen, Qualitative analysis, Testing)
- P2 (Nice): 6 tasks (Documentation)

**Estimated timeline**:
- Data Preparation: 2-3 days
- Statistical Evaluation: 2-3 days
- Report Generation: 3-4 days
- Qualitative Analysis: 2-3 days
- Documentation & Testing: 1-2 days
- **Total: 10-15 days**

**Parallelization opportunities**:
- Tasks 1.2 subtasks (manifest generators) can run in parallel
- Tasks 2.1-2.3 (statistical evaluation) can run in parallel
- Tasks 4.1-4.3 (qualitative analysis) can run in parallel
