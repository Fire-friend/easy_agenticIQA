## ADDED Requirements

### Requirement: Result Aggregation
**Priority**: P0
**Capability**: report-generation

The system SHALL collect and aggregate pipeline outputs, evaluation metrics, and execution statistics from multiple JSONL files into a unified data structure for report generation.

#### Scenario: Collect Pipeline Results
**Given** pipeline outputs in `outputs/` directory: agenticiqa_eval_results.jsonl, tid2013_scores.jsonl, bid_scores.jsonl
**When** `generate_report.py` runs
**Then** all JSONL files are loaded and parsed
**And** results are organized by dataset
**And** missing files are logged as warnings but don't block report generation

#### Scenario: Aggregate Summary Statistics
**Given** collected results from multiple datasets
**When** aggregation is performed
**Then** overall statistics are calculated: total_samples, success_count, error_count
**And** dataset-specific metrics are computed: per-dataset SRCC, PLCC, accuracy
**And** execution statistics are aggregated: total_tokens, total_cost, total_time

---

### Requirement: Markdown Report Structure
**Priority**: P0
**Capability**: report-generation

The system SHALL generate structured Markdown reports following a standard template with sections for environment, dataset status, metrics, comparison, qualitative cases, cost analysis, and discussion.

#### Scenario: Generate Standard Report Structure
**Given** aggregated results and metrics
**When** report is generated
**Then** Markdown file includes sections in order:
  1. Title and metadata (date, AgenticIQA version)
  2. Environment & Configuration (Python, models, tools)
  3. Dataset Status (sample counts, missing data)
  4. AgenticIQA-Eval Results (accuracy tables)
  5. SRCC/PLCC Correlation Results (tables with CI)
  6. Confusion Matrix (if MCQ evaluation)
  7. Qualitative Cases (links and summaries)
  8. Cost & Performance (tokens, cost, time)
  9. Analysis & Discussion
  10. Recommendations

#### Scenario: Include Table of Contents
**Given** report with multiple sections
**When** report is assembled
**Then** table of contents is auto-generated with links
**And** TOC includes all section headers (## and ###)
**And** links navigate to correct sections in Markdown viewers

---

### Requirement: Metric Comparison Tables
**Priority**: P0
**Capability**: report-generation

The system SHALL format metric comparison tables showing paper baselines, our results, deltas, confidence intervals, and statistical significance indicators.

#### Scenario: Format MCQ Accuracy Comparison Table
**Given** our MCQ accuracies and paper baselines
**When** table is formatted
**Then** Markdown table includes columns: Task Type, Paper Acc, Our Acc, Delta, CI (95%), Significant
**And** Delta column shows signed difference (e.g., -0.02)
**And** Significant column shows ✓ or ✗ based on CI overlap with paper value
**And** Table is properly aligned and formatted

#### Scenario: Format Correlation Comparison Table
**Given** SRCC/PLCC for multiple datasets with paper baselines
**When** table is formatted
**Then** Markdown table includes: Dataset, Metric, Paper, Ours, Delta, CI (95%), P-value
**And** Best results are highlighted (bold or *)
**And** Significant deviations are marked

---

### Requirement: Qualitative Case Integration
**Priority**: P1
**Capability**: report-generation

The system SHALL extract and format representative qualitative cases in the report with links to detailed case directories and inline summaries.

#### Scenario: Include Success Case Summary
**Given** selected success case: sample_id=tid2013_0245
**And** case directory at `qualitative_cases/tid2013_0245/`
**When** report includes qualitative section
**Then** case appears with summary:
  - Sample ID and link to case directory
  - Ground truth and prediction
  - Brief explanation (from result.json)
  - Key insight (why it succeeded)

#### Scenario: Include Failure Case Summary
**Given** selected failure case with error classification
**When** report includes failure case
**Then** summary shows:
  - Error category (e.g., "Tool Selection Error")
  - What went wrong (brief description)
  - Expected vs actual behavior
  - Root cause analysis

---

### Requirement: Cost Analysis Section
**Priority**: P1
**Capability**: report-generation

The system SHALL generate cost analysis section showing token usage breakdown by module, API costs per dataset, and performance metrics (tokens/sample, time/sample).

#### Scenario: Format Cost Breakdown
**Given** execution logs with token usage per module (planner/executor/summarizer)
**When** cost section is formatted
**Then** table shows: Module, Total Tokens, Avg Tokens/Sample, Cost (USD)
**And** Total row sums across modules
**And** Cost is calculated using model-specific rate card

#### Scenario: Cost Comparison Across Datasets
**Given** cost data for multiple datasets
**When** cost comparison is generated
**Then** table shows: Dataset, Samples, Total Cost, Cost/Sample, Avg Time/Sample
**And** Most expensive dataset is highlighted
**And** Cost efficiency insights are provided (e.g., "BID most cost-effective due to NR mode")

---

### Requirement: Alternative Model Comparison
**Priority**: P1
**Capability**: report-generation

The system SHALL generate comparison tables for results obtained with different VLM backends, highlighting performance and cost trade-offs.

#### Scenario: Compare Model Performance
**Given** results from runs with GPT-4o, Claude 3.5, and Qwen2.5-VL
**When** model comparison section is generated
**Then** table shows: Model, Accuracy (MCQ), SRCC (TID2013), PLCC (TID2013), Cost/Sample, Time/Sample
**And** Best performer in each metric is highlighted
**And** Cost-performance trade-offs are discussed

#### Scenario: Document Model Configurations
**Given** multiple model runs with different configurations
**When** report is generated
**Then** each model's configuration is documented: temperature, max_tokens, backend URL
**And** Any deviations from paper setup are noted

---

### Requirement: Report Customization
**Priority**: P2
**Capability**: report-generation

The system SHALL support report customization through template selection, section filtering, and output format options.

#### Scenario: Generate Abbreviated Report
**Given** full results and metrics
**When** report is generated with `--template minimal`
**Then** only key sections are included: metrics tables and comparison
**And** Detailed sections (qualitative cases, cost breakdown) are omitted

#### Scenario: Filter Report Sections
**Given** user wants only MCQ evaluation results
**When** report is generated with `--include mcq_eval`
**Then** only AgenticIQA-Eval section is included
**And** Correlation results are omitted

---

### Requirement: Report Validation
**Priority**: P1
**Capability**: report-generation

The system SHALL validate generated reports for Markdown syntax correctness, broken links, missing sections, and table formatting issues.

#### Scenario: Validate Markdown Syntax
**Given** generated report Markdown file
**When** validation runs
**Then** Markdown is parsed and checked for syntax errors
**And** Invalid table formats are detected
**And** Broken heading hierarchy is flagged

#### Scenario: Check Link Integrity
**Given** report with links to case directories and images
**When** validation runs with `--check-links`
**Then** all internal links are verified
**And** Missing case directories are reported
**And** Broken image symlinks are detected

---

### Requirement: Timestamped Reports
**Priority**: P0
**Capability**: report-generation

The system SHALL include execution timestamps, report generation date, and version information in each report for traceability.

#### Scenario: Include Metadata
**Given** report generation time: 2025-10-30 21:30:00 UTC
**When** report is assembled
**Then** header includes:
  - Report generation date and time
  - AgenticIQA version/commit hash
  - Pipeline execution date range
  - Python and key dependency versions
