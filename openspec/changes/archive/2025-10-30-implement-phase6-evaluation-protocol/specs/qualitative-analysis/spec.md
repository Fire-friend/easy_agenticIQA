## ADDED Requirements

### Requirement: Representative Case Selection
**Priority**: P0
**Capability**: qualitative-analysis

The system SHALL automatically select representative cases covering success scenarios, failure scenarios, edge cases, and diverse coverage across task types and distortion types.

#### Scenario: Select High-Confidence Success Cases
**Given** pipeline results with confidence scores and correctness labels
**When** case selection runs for success cases
**Then** top 5 cases are selected with: correct prediction + high confidence (>0.8) + diverse distortion types
**And** Cases cover different task types (planner, executor, summarizer)
**And** No two cases have the same distortion type

#### Scenario: Select High-Confidence Failure Cases
**Given** pipeline results with errors
**When** failure case selection runs
**Then** top 5 cases are selected with: wrong prediction + high confidence + diverse error types
**And** Each failure represents different error category (planning, tool selection, reasoning)

#### Scenario: Select Edge Cases
**Given** pipeline results with confidence scores and replanning flags
**When** edge case selection runs
**Then** cases are selected with: low confidence (<0.5) OR replanning triggered
**And** Cases represent borderline decisions or ambiguous inputs

#### Scenario: Ensure Coverage Diversity
**Given** selected cases across success/failure/edge categories
**When** final case set is assembled
**Then** all task types are represented (at least 1 per type)
**And** all major distortion categories are covered
**And** Both FR and NR reference modes are included

---

### Requirement: Case State Export
**Priority**: P0
**Capability**: qualitative-analysis

The system SHALL export complete case state including planner output, executor evidence, summarizer result, ground truth, and metadata into structured directories for human inspection.

#### Scenario: Export Case Directory Structure
**Given** selected case with sample_id="tid2013_0245"
**When** case export runs
**Then** directory is created at `qualitative_cases/tid2013_0245/`
**And** directory contains files:
  - metadata.json (sample info, ground truth, dataset)
  - plan.json (planner output with query_type, distortions, tools)
  - evidence.json (executor output with distortion_set, tool_scores)
  - result.json (summarizer final_answer, quality_reasoning)
  - analysis.md (human-readable summary)
  - images/ (symlinks to original and reference images)

#### Scenario: Generate Analysis Markdown
**Given** case state exported to directory
**When** analysis.md is generated
**Then** file includes:
  - Sample overview (ID, dataset, distortion type, ground truth)
  - Planner analysis (query interpretation, plan correctness)
  - Executor analysis (detected distortions, tool selection, scores)
  - Summarizer analysis (reasoning quality, final answer)
  - Overall assessment (success/failure reason)
  - Comparison with paper example (if available)

#### Scenario: Symlink Images
**Given** case with image paths in metadata
**When** images/ directory is created
**Then** symbolic links are created to original distorted image
**And** symbolic link to reference image (if FR mode)
**And** links are relative paths for portability

---

### Requirement: Error Taxonomy Classification
**Priority**: P1
**Capability**: qualitative-analysis

The system SHALL classify error cases into predefined categories (planning error, tool selection error, tool execution error, reasoning error, integration error, reference mode error) based on heuristic analysis of intermediate states.

#### Scenario: Classify Planning Error
**Given** failure case where planner identified wrong distortion type
**And** ground truth distortion is "JPEG compression"
**And** planner identified "Gaussian blur"
**When** error classification runs
**Then** case is classified as "planning_error"
**And** explanation notes: "Planner misidentified distortion type"

#### Scenario: Classify Tool Selection Error
**Given** failure case where planner correctly identified distortion
**And** executor selected inappropriate tool (e.g., LPIPS for color distortion)
**When** error classification runs
**Then** case is classified as "tool_selection_error"
**And** explanation notes: "Executor chose tool unsuitable for distortion type"

#### Scenario: Classify Reasoning Error
**Given** failure case where plan and evidence are correct
**And** summarizer drew wrong conclusion
**When** error classification runs
**Then** case is classified as "reasoning_error"
**And** explanation notes: "Summarizer failed to correctly interpret tool scores"

#### Scenario: Classify Ambiguous Error
**Given** failure case with errors in multiple modules
**When** error classification runs
**Then** all applicable error categories are listed
**And** primary error (root cause) is highlighted
**And** explanation describes error cascade

---

### Requirement: Paper Example Comparison
**Priority**: P2
**Capability**: qualitative-analysis

The system SHALL compare selected cases with paper-provided examples when available, highlighting similarities and differences in plans, evidence, and reasoning.

#### Scenario: Match Case with Paper Example
**Given** case with sample_id="tid2013_I01_01_1"
**And** paper example for same sample in `data/paper_examples/`
**When** comparison is performed
**Then** paper example is loaded
**And** plans are compared (our distortions vs paper distortions)
**And** tools are compared (our tool selection vs paper tools)
**And** reasoning is compared (our explanation vs paper explanation)
**And** differences are highlighted in comparison.md

#### Scenario: Explain Deviations from Paper
**Given** case where our result differs from paper example
**When** comparison analysis runs
**Then** report identifies root cause of deviation:
  - Different model backend (GPT-4o vs Qwen2.5-VL)
  - Different tool library (QAlign vs TOPIQ)
  - Different prompt engineering
**And** impact of deviation is assessed (acceptable vs concerning)

---

### Requirement: Case Visualization Support
**Priority**: P2
**Capability**: qualitative-analysis

The system SHALL support optional visualization of cases including distortion heatmaps, tool score distributions, and confidence plots.

#### Scenario: Generate Distortion Heatmap
**Given** case with spatial distortion localization (if supported by tools)
**When** visualization is enabled
**Then** heatmap image is generated overlaying distortion intensity on original image
**And** heatmap is saved in case images/ directory

#### Scenario: Plot Tool Score Distribution
**Given** case with multiple tool scores
**When** visualization runs
**Then** bar chart is generated showing tool scores side-by-side
**And** Ground truth MOS is overlaid for comparison
**And** Chart is saved as PNG in case directory

---

### Requirement: Batch Case Export
**Priority**: P0
**Capability**: qualitative-analysis

The system SHALL support exporting multiple selected cases in batch mode with parallel processing for efficiency.

#### Scenario: Export 20 Cases in Parallel
**Given** 20 selected cases from select_cases.py
**When** batch export runs with `--parallel`
**Then** cases are exported concurrently across CPU cores
**And** all 20 case directories are created
**And** export completes faster than sequential processing

#### Scenario: Handle Export Failures Gracefully
**Given** batch export with 1 case having missing image files
**When** export runs
**Then** error is logged for failed case
**And** export continues with remaining cases
**And** summary reports: 19 success, 1 failure

---

### Requirement: Case Anonymization
**Priority**: P2
**Capability**: qualitative-analysis

The system SHALL support anonymizing exported cases by removing or obfuscating sensitive metadata when sharing for external review.

#### Scenario: Anonymize Case Metadata
**Given** case with sample_id containing dataset-specific identifier
**When** export runs with `--anonymize` flag
**Then** sample_id is replaced with generic ID (e.g., "case_001")
**And** Image paths are redacted
**And** Dataset name is replaced with generic "Dataset A"
**And** analysis.md uses anonymized identifiers

---

### Requirement: Interactive Case Browser
**Priority**: P3
**Capability**: qualitative-analysis

The system SHALL provide a simple command-line interactive browser for navigating exported cases, filtering by category, and viewing summaries.

#### Scenario: Browse Cases Interactively
**Given** exported cases in qualitative_cases/ directory
**When** `browse_cases.py` is run
**Then** CLI shows list of cases with sample_id, category, success/failure
**And** User can navigate with arrow keys
**And** Selecting a case displays analysis.md content
**And** User can filter by error category or distortion type
