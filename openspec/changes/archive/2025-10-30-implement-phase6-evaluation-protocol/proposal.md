# Proposal: Implement Phase 6 Evaluation Protocol

## Summary
Implement comprehensive evaluation protocol and result reproduction infrastructure for AgenticIQA, enabling systematic comparison with paper results through data preparation pipelines, enhanced evaluation scripts with statistical analysis, automated report generation, and qualitative case analysis.

## Context
Phase 5 completed the batch inference pipeline with basic evaluation scripts (`eval_mcq_accuracy.py`, `eval_correlation.py`). The system can process datasets and calculate MCQ accuracy and SRCC/PLCC correlations. However, to fully reproduce and validate the paper's results, we need:

1. **Standardized data preparation** - No manifest generation scripts or schema validation for TID2013, BID, AGIQA-3K, AgenticIQA-Eval
2. **Statistical rigor** - Missing bootstrap confidence intervals, confusion matrices, per-category breakdowns
3. **Reproducibility documentation** - No automated report generation comparing with paper results
4. **Qualitative analysis** - No tools for extracting and analyzing representative cases
5. **Alternative model tracking** - No systematic comparison of different model backends

This gap prevents researchers from:
- Independently reproducing paper results
- Understanding performance variance and confidence
- Identifying failure modes through qualitative analysis
- Documenting alternative model strategies

## Why

### User Impact
Researchers attempting to reproduce the paper face significant barriers:
- **Data preparation overhead** - Manual manifest creation for each dataset is error-prone and time-consuming
- **Statistical uncertainty** - Point estimates (SRCC/PLCC) without confidence intervals make it hard to assess significance
- **Opaque failures** - No confusion matrix or error analysis to understand where the system fails
- **Incomplete documentation** - No standardized report format for reproduction attempts

### Business Value
- **Scientific credibility** - Reproducible results with confidence intervals strengthen the paper's claims
- **Research velocity** - Automated evaluation pipeline reduces reproduction time from weeks to days
- **Community adoption** - Clear documentation and tools encourage independent verification and extension
- **Comparison framework** - Systematic alternative model tracking enables fair performance comparisons

### Technical Motivation
The current evaluation infrastructure lacks production-grade analysis capabilities:
- **eval_mcq_accuracy.py** - Only reports overall accuracy, missing per-category and confusion analysis
- **eval_correlation.py** - Point estimates only, no statistical significance testing
- **No manifest tooling** - Researchers must manually create JSONL manifests for each dataset
- **No report generation** - Results scattered across console output and JSON files
- **No qualitative tools** - Representative cases must be manually extracted and analyzed

## What Changes

This change implements four major capabilities:

### 1. Data Preparation Pipeline
- **Manifest schemas** - JSON schemas for each dataset (TID2013, BID, AGIQA-3K, AgenticIQA-Eval)
- **Generation scripts** - Convert raw datasets to standardized JSONL manifests
- **Validation tools** - Schema validation and integrity checking

### 2. Enhanced Statistical Evaluation
- **Bootstrap confidence intervals** - 95% CI for SRCC/PLCC using resampling
- **Confusion matrix** - Per-option error analysis for MCQ tasks
- **Per-category breakdowns** - Accuracy by task type, distortion type, reference mode
- **Significance testing** - P-values and effect sizes for correlations

### 3. Automated Report Generation
- **Result aggregation** - Collect metrics across all datasets
- **Paper comparison** - Side-by-side tables with paper results
- **Cost analysis** - Token usage and API cost breakdown
- **Qualitative cases** - Automatic extraction of representative examples
- **Markdown reports** - Structured report with tables, figures, and analysis

### 4. Qualitative Analysis Tools
- **Case selection** - Identify high-confidence successes and failures
- **State extraction** - Export plan/evidence/result for inspection
- **Comparison utilities** - Match system output with paper examples
- **Error categorization** - Classify failure modes (planning, tool selection, reasoning)

## Goals
1. **Data Preparation**: Generate valid JSONL manifests for all benchmark datasets
2. **Statistical Rigor**: Provide confidence intervals and significance tests for all metrics
3. **Comprehensive Reports**: Auto-generate reproduction reports comparing with paper results
4. **Qualitative Analysis**: Extract and categorize representative cases (success/failure)
5. **Alternative Models**: Document performance across different VLM backends
6. **Reproducibility**: Enable independent researchers to reproduce paper results systematically

## Non-Goals
- Training or fine-tuning models (inference-only reproduction)
- Web-based visualization dashboard (command-line tools only)
- Real-time monitoring or alerting (batch analysis only)
- Custom metric development (use standard SRCC/PLCC/accuracy)
- Dataset distribution or hosting (assume researchers obtain datasets independently)

## Stakeholders
- **Paper Authors**: Need reproducible results to support publication claims
- **Reviewers**: Require confidence intervals and statistical validation
- **Researchers**: Want to compare their modifications against baseline
- **Practitioners**: Need cost and performance data for deployment decisions

## Risks & Mitigation
- **Risk**: Dataset access restrictions (some datasets require registration)
  - **Mitigation**: Provide manifest schemas and example generation scripts; document access procedures
- **Risk**: Statistical variance makes exact reproduction difficult
  - **Mitigation**: Report confidence intervals; document all hyperparameters and random seeds
- **Risk**: Alternative models produce significantly different results
  - **Mitigation**: Track multiple backends; clearly label deviations from paper setup
- **Risk**: Report generation becomes too rigid/opinionated
  - **Mitigation**: Provide both automated reports and raw data export for custom analysis

## Success Criteria
- [ ] Manifest generation scripts successfully process TID2013, BID, AGIQA-3K datasets
- [ ] Bootstrap confidence intervals calculated for all SRCC/PLCC metrics
- [ ] Confusion matrix shows per-option error breakdown for AgenticIQA-Eval MCQ
- [ ] Automated report generation produces complete reproduction report
- [ ] Qualitative analysis extracts at least 10 representative cases per dataset
- [ ] Alternative model comparison table documents performance across 3+ VLM backends
- [ ] Documentation enables independent researcher to reproduce results in <1 week

## Related Changes
- Depends on: `implement-phase5-inference-pipeline` (archived)
- Enables: Future work on prompt optimization, tool selection refinement
