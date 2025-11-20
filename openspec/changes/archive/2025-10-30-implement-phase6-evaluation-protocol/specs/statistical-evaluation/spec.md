## ADDED Requirements

### Requirement: Bootstrap Confidence Intervals
**Priority**: P0
**Capability**: statistical-evaluation

The system SHALL calculate bootstrap confidence intervals for SRCC and PLCC metrics using resampling with replacement (default n=1000 iterations) to provide statistical rigor for correlation results.

#### Scenario: Calculate SRCC with Bootstrap CI
**Given** predictions and ground truth MOS for 3000 samples
**When** `eval_with_ci.py --metric srcc --n-bootstrap 1000` is executed
**Then** SRCC is calculated on original data
**And** 1000 bootstrap samples are generated with replacement
**And** SRCC is calculated for each bootstrap sample
**And** 95% confidence interval is computed as [2.5th percentile, 97.5th percentile]
**And** result includes: {'srcc': 0.874, 'srcc_ci': [0.865, 0.883], 'srcc_pvalue': <0.001}

#### Scenario: Calculate PLCC with Bootstrap CI
**Given** predictions and MOS for a dataset
**When** bootstrap CI is calculated for PLCC
**Then** Pearson correlation is computed on original data
**And** bootstrap resampling provides CI
**And** result includes PLCC value, CI bounds, and p-value

#### Scenario: Parallel Bootstrap for Performance
**Given** large dataset requiring bootstrap CI
**When** evaluation runs with `--parallel` flag
**Then** bootstrap iterations are distributed across CPU cores
**And** computation time is reduced proportionally
**And** results are identical to sequential execution

---

### Requirement: Confusion Matrix Analysis
**Priority**: P0
**Capability**: statistical-evaluation

The system SHALL generate confusion matrices for MCQ evaluation showing per-option error patterns, precision/recall, and most commonly confused option pairs.

#### Scenario: Generate MCQ Confusion Matrix
**Given** MCQ predictions and ground truth answers for 1000 questions with options A/B/C/D
**When** `eval_mcq_with_confusion.py` is executed
**Then** confusion matrix is generated with shape (4, 4)
**And** rows represent ground truth, columns represent predictions
**And** diagonal elements show correct predictions per option
**And** off-diagonal elements show confusion patterns (e.g., A predicted as B)

#### Scenario: Calculate Per-Option Precision and Recall
**Given** confusion matrix for MCQ evaluation
**When** per-option metrics are computed
**Then** precision for each option = diagonal / column sum
**And** recall for each option = diagonal / row sum
**And** results show which options are over/under-predicted

#### Scenario: Identify Most Confused Pairs
**Given** confusion matrix with non-zero off-diagonal elements
**When** confusion analysis runs
**Then** top 5 most confused (ground_truth, predicted) pairs are identified
**And** results show: [('A', 'B', 45 errors), ('C', 'D', 32 errors), ...]
**And** confusion pairs are sorted by error count descending

---

### Requirement: Per-Category Performance Breakdown
**Priority**: P0
**Capability**: statistical-evaluation

The system SHALL compute accuracy and correlation metrics broken down by multiple categorical dimensions including task_type, distortion_type, reference_mode, and question_type.

#### Scenario: Breakdown by Task Type
**Given** AgenticIQA-Eval results with task_type field (planner/executor_distortion/executor_tool/summarizer)
**When** `eval_by_category.py --group-by task_type` is executed
**Then** accuracy is calculated separately for each task type
**And** sample count per task type is reported
**And** results table shows: task_type, accuracy, sample_count, ci_lower, ci_upper

#### Scenario: Breakdown by Distortion Type
**Given** TID2013 results with distortion_type metadata (JPEG/Gaussian blur/noise/...)
**When** evaluation groups by distortion_type
**Then** SRCC and PLCC are calculated per distortion type
**And** results identify which distortions have highest/lowest correlation

#### Scenario: Breakdown by Reference Mode
**Given** AgenticIQA-Eval results with reference_mode field (FR/NR)
**When** evaluation groups by reference_mode
**Then** accuracy is calculated separately for FR and NR questions
**And** statistical test compares FR vs NR performance (chi-square)
**And** p-value indicates if difference is significant

#### Scenario: Multi-Level Breakdown
**Given** results with both task_type and reference_mode
**When** evaluation runs with `--group-by task_type,reference_mode`
**Then** metrics are calculated for each combination (e.g., planner+FR, planner+NR, ...)
**And** results table has hierarchical grouping

---

### Requirement: Statistical Significance Testing
**Priority**: P1
**Capability**: statistical-evaluation

The system SHALL perform statistical significance tests including p-values for correlations, chi-square tests for categorical comparisons, and t-tests for mean differences.

#### Scenario: Test Correlation Significance
**Given** SRCC or PLCC correlation coefficient
**When** significance test is performed
**Then** p-value is calculated using appropriate distribution
**And** null hypothesis (no correlation) is tested
**And** result indicates if correlation is statistically significant (p < 0.05)

#### Scenario: Compare Accuracies Across Categories
**Given** accuracy for two task types (planner: 0.85, executor: 0.76)
**When** chi-square test is performed
**Then** test determines if accuracy difference is statistically significant
**And** p-value is reported
**And** effect size (Cohen's h) is calculated

---

### Requirement: Paper Result Comparison
**Priority**: P1
**Capability**: statistical-evaluation

The system SHALL compare computed metrics with paper-reported baselines, calculate deltas, and identify statistically significant deviations.

#### Scenario: Load Paper Baseline Results
**Given** paper results file at `data/paper_results/baseline.json`
**When** comparison is initialized
**Then** paper-reported SRCC, PLCC, and accuracies are loaded
**And** model configurations are recorded (which VLM, which tools)

#### Scenario: Calculate Metric Deltas
**Given** our results: TID2013 SRCC = 0.874, paper: SRCC = 0.892
**When** comparison is performed
**Then** delta is calculated: 0.874 - 0.892 = -0.018
**And** relative difference is calculated: -0.018 / 0.892 = -2.0%
**And** result indicates performance below paper baseline

#### Scenario: Identify Significant Deviations
**Given** our accuracy with 95% CI: [0.80, 0.86], paper accuracy: 0.89
**When** comparison checks for overlap
**Then** paper value falls outside our CI
**And** deviation is marked as statistically significant
**And** report flags this for investigation

---

### Requirement: Correlation Visualization
**Priority**: P2
**Capability**: statistical-evaluation

The system SHALL generate scatter plots showing predicted vs ground truth scores with regression lines for visual assessment of correlation quality.

#### Scenario: Generate Scatter Plot
**Given** predictions and MOS for TID2013
**When** visualization is enabled with `--plot`
**Then** scatter plot is generated with predictions on x-axis, MOS on y-axis
**And** regression line is overlaid
**And** SRCC and PLCC values are annotated on plot
**And** plot is saved as PNG

---

### Requirement: JSON Output Format
**Priority**: P0
**Capability**: statistical-evaluation

The system SHALL export all statistical results in structured JSON format for programmatic consumption and report generation.

#### Scenario: Export Evaluation Results as JSON
**Given** completed evaluation with SRCC, PLCC, CI, p-values
**When** results are saved with `--output results.json`
**Then** JSON file includes all computed metrics
**And** structure is: {'srcc': float, 'srcc_ci': [float, float], 'srcc_pvalue': float, ...}
**And** JSON is valid and parseable
