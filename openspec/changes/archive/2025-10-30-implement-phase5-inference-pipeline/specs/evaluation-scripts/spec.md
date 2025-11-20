## ADDED Requirements

### Requirement: MCQ Accuracy Calculation
**Priority**: P1
**Capability**: evaluation-scripts

The system SHALL provide a script to calculate multiple-choice question (MCQ) accuracy from pipeline output by comparing final_answer against ground truth.

#### Scenario: Calculate Overall Accuracy
**Given** an output JSONL with 100 MCQ results
**And** a ground truth file with correct answers
**When** the evaluation script runs
**Then** accuracy is calculated as (correct / total) * 100
**And** results show "Accuracy: 87.5% (87/100)"

#### Scenario: Calculate Per-Category Accuracy
**Given** MCQ results with categories (Planner, Executor, Summarizer)
**When** the evaluation script runs
**Then** accuracy is calculated per category
**And** results show breakdown: "Planner: 90%, Executor: 85%, Summarizer: 88%"

---

### Requirement: Correlation Metrics (SRCC/PLCC)
**Priority**: P1
**Capability**: evaluation-scripts

The system SHALL provide a script to calculate Spearman Rank Correlation (SRCC) and Pearson Linear Correlation (PLCC) for quality scoring tasks.

#### Scenario: Calculate SRCC for TID2013
**Given** pipeline output with predicted quality scores
**And** ground truth MOS (Mean Opinion Score) values
**When** the correlation script runs
**Then** SRCC is calculated using scipy.stats.spearmanr
**And** PLCC is calculated using scipy.stats.pearsonr
**And** results show "SRCC: 0.892, PLCC: 0.901"

#### Scenario: Handle Missing Scores
**Given** some samples failed and have no predicted scores
**When** correlation is calculated
**Then** failed samples are excluded from computation
**And** a warning is logged: "Excluded 5 samples with missing scores"

---

### Requirement: Comprehensive Report Generation
**Priority**: P2
**Capability**: evaluation-scripts

The system SHALL provide a script to generate a comprehensive markdown report aggregating results across multiple runs.

#### Scenario: Generate Evaluation Report
**Given** results from AgenticIQA-Eval and TID2013 evaluations
**When** the report generator runs
**Then** a markdown report is created with:
- Summary table (dataset, samples, accuracy/SRCC/PLCC)
- Execution metrics (avg time, total cost)
- Error summary (failed samples, error types)
**And** report is saved to specified path

---

### Requirement: Cost Analysis
**Priority**: P2
**Capability**: evaluation-scripts

The system SHALL provide utilities to analyze execution logs and compute total API costs, token usage, and timing statistics.

#### Scenario: Calculate Total Cost
**Given** execution logs with token usage per sample
**And** rate card (e.g., GPT-4o: $2.50/1M input, $10/1M output)
**When** cost analysis runs
**Then** total cost is computed across all samples
**And** breakdown by model is provided: "GPT-4o: $12.50, GPT-4o-mini: $0.80"

---

### Requirement: Performance Benchmarking
**Priority**: P2
**Capability**: evaluation-scripts

The system SHALL provide scripts to benchmark execution performance including average time per sample, throughput, and bottleneck identification.

#### Scenario: Analyze Execution Time
**Given** execution logs with timing per sample
**When** performance analysis runs
**Then** statistics are computed: mean, median, p95, p99 execution time
**And** slowest samples are identified
**And** time distribution by agent (planner/executor/summarizer) is shown
