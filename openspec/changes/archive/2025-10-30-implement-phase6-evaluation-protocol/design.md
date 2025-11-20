# Design: Phase 6 Evaluation Protocol

## Overview

Phase 6 implements a comprehensive evaluation protocol infrastructure that transforms raw datasets into standardized manifests, performs statistical analysis with confidence intervals, generates automated reproduction reports, and extracts qualitative cases for analysis.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 6 Evaluation Protocol                 │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐    ┌────────────────┐    ┌──────────────────┐
│ Data Prep     │    │ Statistical    │    │ Report           │
│ Pipeline      │    │ Evaluation     │    │ Generation       │
├───────────────┤    ├────────────────┤    ├──────────────────┤
│ • Schemas     │    │ • Bootstrap CI │    │ • Aggregation    │
│ • Generators  │    │ • Confusion    │    │ • Comparison     │
│ • Validators  │    │ • Per-category │    │ • Markdown       │
└───────────────┘    └────────────────┘    └──────────────────┘
        │                     │                     │
        └─────────────────────┴─────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ Qualitative       │
                    │ Analysis          │
                    ├───────────────────┤
                    │ • Case selection  │
                    │ • State export    │
                    │ • Error taxonomy  │
                    └───────────────────┘
```

## Component Design

### 1. Data Preparation Pipeline

**Purpose**: Transform raw datasets into standardized JSONL manifests with validated schemas.

**Components**:

#### 1.1 Manifest Schemas (`data/schemas/`)
```
schemas/
  agenticiqa_eval_schema.json    # MCQ questions with task types
  tid2013_schema.json             # FR-IQA with MOS scores
  bid_schema.json                 # NR-IQA with MOS scores
  agiqa3k_schema.json            # NR-IQA generated images
  common_fields.json              # Shared field definitions
```

**Schema Structure** (example for TID2013):
```json
{
  "type": "object",
  "required": ["sample_id", "dataset", "distorted_path", "reference_path", "mos"],
  "properties": {
    "sample_id": {"type": "string", "pattern": "^tid2013_\\d{4}$"},
    "dataset": {"const": "tid2013"},
    "distorted_path": {"type": "string", "format": "file-path"},
    "reference_path": {"type": "string", "format": "file-path"},
    "mos": {"type": "number", "minimum": 0, "maximum": 9},
    "split": {"enum": ["train", "val", "test"]},
    "metadata": {
      "type": "object",
      "properties": {
        "distortion_type": {"type": "string"},
        "level": {"type": "integer", "minimum": 1, "maximum": 5"}
      }
    }
  }
}
```

#### 1.2 Manifest Generators (`scripts/data_prep/`)
```
scripts/data_prep/
  generate_tid2013_manifest.py
  generate_bid_manifest.py
  generate_agiqa3k_manifest.py
  generate_agenticiqa_eval_manifest.py
  validate_manifest.py
```

**Generator Workflow**:
1. Read raw dataset directory structure
2. Parse metadata files (CSV, XML, annotations)
3. Map to standardized schema
4. Validate all file paths exist
5. Write JSONL manifest with checksums

#### 1.3 Validation Tools
- **Schema validation**: JSON Schema validator
- **File path validation**: Check all images/references exist
- **Data integrity**: Verify MOS ranges, no duplicates
- **Completeness**: Ensure all required fields present

### 2. Enhanced Statistical Evaluation

**Purpose**: Provide statistically rigorous metrics with confidence intervals and error analysis.

**Components**:

#### 2.1 Bootstrap Confidence Intervals
```python
# scripts/eval_with_ci.py
def bootstrap_correlation(predictions, ground_truth, n_bootstrap=1000, confidence=0.95):
    """
    Calculate SRCC/PLCC with bootstrap confidence intervals.

    Returns:
        {
            'srcc': float,
            'srcc_ci': (lower, upper),
            'srcc_pvalue': float,
            'plcc': float,
            'plcc_ci': (lower, upper),
            'plcc_pvalue': float
        }
    """
```

**Algorithm**:
1. For B=1000 iterations:
   - Resample (predictions, MOS) pairs with replacement
   - Calculate SRCC/PLCC for resampled data
2. Compute percentiles: [2.5%, 97.5%] for 95% CI
3. Report mean, CI, and original metric

#### 2.2 Confusion Matrix Analysis
```python
# scripts/eval_mcq_with_confusion.py
def analyze_mcq_confusion(predictions, ground_truth, options=['A', 'B', 'C', 'D']):
    """
    Generate confusion matrix and per-option error analysis.

    Returns:
        {
            'confusion_matrix': np.ndarray,  # shape (n_options, n_options)
            'per_option_precision': dict,
            'per_option_recall': dict,
            'most_confused_pairs': list
        }
    """
```

**Output**:
- Confusion matrix heatmap (matplotlib/text)
- Precision/recall per option
- Most common error pairs (e.g., A→B confusion)

#### 2.3 Per-Category Breakdown
```python
def eval_by_category(results_df, group_by='task_type'):
    """
    Calculate metrics broken down by category.

    Categories:
    - task_type: planner / executor_distortion / executor_tool / summarizer
    - distortion_type: JPEG / Gaussian blur / ...
    - reference_mode: FR / NR
    - question_type: What / How / YesNo
    """
```

### 3. Automated Report Generation

**Purpose**: Generate comprehensive Markdown reports comparing results with paper baselines.

**Architecture**:

```
scripts/generate_report.py
  ├─ collect_results()       # Load all outputs/*.jsonl
  ├─ aggregate_metrics()     # Calculate summary statistics
  ├─ compare_with_paper()    # Load paper results, compute deltas
  ├─ format_tables()         # Generate Markdown tables
  ├─ extract_cases()         # Select qualitative examples
  ├─ calculate_costs()       # Aggregate token usage and costs
  └─ write_report()          # Assemble final report
```

**Report Structure** (`reports/reproduction_YYYYMMDD.md`):
```markdown
# AgenticIQA Reproduction Report

## 1. Environment & Configuration
- Python version, dependencies
- Model backends (planner, executor, summarizer)
- IQA tools used
- Execution timestamp

## 2. Dataset Status
| Dataset | Samples | Missing | MOS Range | Split |
|---------|---------|---------|-----------|-------|
| TID2013 | 3000    | 0       | 0-9       | test  |

## 3. AgenticIQA-Eval Results
| Task Type           | Paper Acc | Our Acc | Delta | CI (95%)      |
|---------------------|-----------|---------|-------|---------------|
| Planner             | 0.85      | 0.83    | -0.02 | [0.80, 0.86]  |
| Executor Distortion | 0.78      | 0.76    | -0.02 | [0.72, 0.80]  |

## 4. SRCC/PLCC Correlation Results
| Dataset  | Metric | Paper  | Ours   | Delta | CI (95%)       | P-value |
|----------|--------|--------|--------|-------|----------------|---------|
| TID2013  | SRCC   | 0.892  | 0.874  | -0.018| [0.865, 0.883] | <0.001  |

## 5. Confusion Matrix (AgenticIQA-Eval)
[Matrix visualization or ASCII table]

## 6. Qualitative Cases
### Success Case: Sample tid2013_0245
- Plan: {...}
- Evidence: {...}
- Explanation: System correctly identified JPEG artifacts

### Failure Case: Sample bid_1234
- Error: Misclassified as blur instead of noise
- Root cause: Tool selection error

## 7. Cost & Performance
- Total tokens: 1,234,567
- Total cost: $45.67
- Avg time per sample: 12.3s

## 8. Analysis & Discussion
- Performance gap analysis
- Alternative model comparison
- Known limitations

## 9. Recommendations
- Suggested improvements
- Future work
```

### 4. Qualitative Analysis Tools

**Purpose**: Extract and analyze representative cases for understanding system behavior.

**Components**:

#### 4.1 Case Selection Strategy
```python
# scripts/qualitative/select_cases.py
def select_representative_cases(results_df, n_cases=10):
    """
    Select diverse cases covering:
    - High-confidence correct predictions
    - High-confidence incorrect predictions (errors)
    - Low-confidence/borderline cases
    - Different distortion types
    - Different task types
    """
```

**Selection Criteria**:
- **Success cases**: High confidence + correct answer + diverse distortion types
- **Failure cases**: High confidence + wrong answer + clear error patterns
- **Edge cases**: Low confidence or replanning triggered
- **Coverage**: Ensure all task types and distortion types represented

#### 4.2 State Export
```python
# scripts/qualitative/export_case.py
def export_case(sample_id, result, output_dir):
    """
    Export complete case analysis:
    - Original question/image
    - Planner output (plan JSON)
    - Executor output (evidence JSON)
    - Summarizer output (final answer + reasoning)
    - Ground truth
    - Comparison with paper example (if available)
    """
```

**Output Format** (`qualitative_cases/<sample_id>/`):
```
sample_id/
  metadata.json          # Sample info and ground truth
  plan.json              # Planner output
  evidence.json          # Executor output
  result.json            # Summarizer output
  analysis.md            # Human-readable analysis
  images/                # Original and reference images (symlinks)
```

#### 4.3 Error Taxonomy
```python
# Classify errors by root cause
ERROR_CATEGORIES = {
    'planning_error': 'Planner chose wrong distortion type',
    'tool_selection_error': 'Executor chose inappropriate tool',
    'tool_execution_error': 'Tool failed or returned invalid score',
    'reasoning_error': 'Summarizer drew wrong conclusion from evidence',
    'integration_error': 'Inconsistency between modules',
    'reference_mode_error': 'FR/NR confusion'
}
```

## Key Design Decisions

### D1: Schema-First Data Preparation
**Decision**: Use JSON Schema for validation rather than ad-hoc checks

**Rationale**:
- **Pros**: Declarative, portable, tool ecosystem support, versioning
- **Cons**: Learning curve for JSON Schema syntax
- **Alternative considered**: Python Pydantic models (chose schemas for language-agnostic validation)

### D2: Bootstrap for Confidence Intervals
**Decision**: Use bootstrap resampling (n=1000) instead of parametric CI

**Rationale**:
- **Pros**: No distribution assumptions, works for SRCC/PLCC, widely accepted
- **Cons**: Computationally expensive (1000 iterations)
- **Alternative considered**: Fisher z-transform for PLCC (doesn't work well for SRCC)

### D3: Markdown Reports
**Decision**: Generate Markdown reports instead of HTML/PDF

**Rationale**:
- **Pros**: Version control friendly, human-readable, easy to edit, GitHub rendering
- **Cons**: Limited formatting, no interactive elements
- **Alternative considered**: Jupyter notebooks (chose Markdown for simplicity and automation)

### D4: Qualitative Case Selection
**Decision**: Automated selection with diversity heuristics

**Rationale**:
- **Pros**: Reproducible, covers edge cases, scalable
- **Cons**: May miss interesting cases, requires manual review
- **Alternative considered**: Full manual selection (too time-consuming for large datasets)

### D5: Per-Category Analysis
**Decision**: Multiple grouping dimensions (task type, distortion, reference mode)

**Rationale**:
- **Pros**: Reveals nuanced performance patterns, identifies systematic biases
- **Cons**: Proliferation of tables/metrics
- **Alternative considered**: Overall metrics only (loses important insights)

## Data Flow

```
Raw Datasets (TID2013, BID, AGIQA-3K, AgenticIQA-Eval)
  │
  ▼ [Manifest Generators]
JSONL Manifests (validated against schemas)
  │
  ▼ [run_pipeline.py]
Pipeline Results (outputs/*.jsonl)
  │
  ├──▶ [eval_with_ci.py] ──▶ SRCC/PLCC with CI
  ├──▶ [eval_mcq_with_confusion.py] ──▶ Confusion matrices
  ├──▶ [select_cases.py] ──▶ Qualitative cases
  │
  ▼ [generate_report.py]
Final Reproduction Report (reports/*.md)
```

## Performance Considerations

### P1: Bootstrap Performance
- **Challenge**: 1000 iterations per dataset can take minutes
- **Mitigation**: Parallelize with multiprocessing, use NumPy vectorization, cache results

### P2: Manifest Generation
- **Challenge**: Large datasets (10K+ images) take time to validate
- **Mitigation**: Incremental validation, parallel path checking, progress bars

### P3: Report Generation
- **Challenge**: Loading all results into memory
- **Mitigation**: Stream JSONL files, aggregate incrementally, use generators

## Alternative Designs Considered

### A1: SQL Database for Results
**Rejected**: Adds complexity, JSONL sufficient for batch analysis

### A2: Interactive Dashboard
**Rejected**: Out of scope, command-line tools prioritized

### A3: Per-Sample Cost Tracking
**Deferred**: Execution logger captures metrics, detailed cost analysis can be added later

## Success Metrics

- Manifest generation completes for all datasets in <5 minutes
- Bootstrap CI calculation overhead <2x baseline evaluation time
- Report generation completes in <1 minute
- Qualitative case export covers 10+ cases per dataset
- Documentation enables reproduction in <1 week (validated by external user)
