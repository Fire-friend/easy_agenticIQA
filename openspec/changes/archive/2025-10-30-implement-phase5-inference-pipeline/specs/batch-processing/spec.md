## ADDED Requirements

### Requirement: JSONL Dataset Loading
**Priority**: P0
**Capability**: batch-processing

The system SHALL load input datasets from JSONL (JSON Lines) format where each line represents a single sample with required fields: `sample_id`, `query`, `image_path`, and optional `reference_path`.

#### Scenario: Load AgenticIQA-Eval Dataset
**Given** a JSONL file at `data/processed/agenticiqa_eval/planner.jsonl`
**When** the batch processor reads the file
**Then** each line is parsed as a JSON object with sample metadata
**And** missing required fields cause validation errors

#### Scenario: Handle Malformed JSONL
**Given** a JSONL file with invalid JSON on line 42
**When** the parser encounters the malformed line
**Then** the error is logged with line number
**And** processing continues with the next line (or fails based on config)

---

### Requirement: Streaming Input Processing
**Priority**: P0
**Capability**: batch-processing

The system SHALL process input JSONL in streaming mode to avoid loading the entire dataset into memory.

#### Scenario: Process Large Dataset
**Given** a JSONL file with 10,000 samples
**When** batch processing starts
**Then** samples are read one at a time
**And** memory usage remains constant regardless of dataset size

---

### Requirement: Resume from Interruption
**Priority**: P0
**Capability**: batch-processing

The system SHALL support resuming batch processing after interruption by reading the existing output JSONL and skipping already-processed samples.

#### Scenario: Resume After Crash
**Given** a batch run that processed 50 out of 100 samples before crashing
**When** the pipeline restarts with `--resume` flag
**Then** the system reads the output JSONL
**And** extracts the 50 processed sample_ids
**And** skips these samples in the input
**And** processes only the remaining 50 samples

#### Scenario: Detect Duplicate Processing
**Given** a sample_id appears in both input and output
**When** resume mode is enabled
**Then** the sample is skipped
**And** a log entry indicates "Sample already processed"

---

### Requirement: Progress Tracking
**Priority**: P1
**Capability**: batch-processing

The system SHALL display real-time progress during batch processing including current sample number, total samples, success count, error count, and estimated time remaining.

#### Scenario: Display Progress Bar
**Given** a batch run processing 100 samples
**When** sample 25 completes successfully
**Then** progress shows "25/100 (25%)"
**And** ETA is calculated based on average sample time
**And** success count shows "24 successful, 1 error"

---

### Requirement: Sample Filtering
**Priority**: P1
**Capability**: batch-processing

The system SHALL support limiting batch processing to a specific number of samples or sample ID range for testing and debugging.

#### Scenario: Process First N Samples
**Given** a dataset with 1000 samples
**When** the user runs with `--max-samples 10`
**Then** only the first 10 samples are processed
**And** the output contains exactly 10 results

#### Scenario: Process Specific Sample Range
**Given** a dataset with sample_ids "sample_001" through "sample_100"
**When** the user specifies `--sample-range 20-30`
**Then** only samples 20-30 (inclusive) are processed
**And** other samples are skipped

---

### Requirement: Batch Processing Loop
**Priority**: P0
**Capability**: batch-processing

The system SHALL iterate through input samples, invoke the LangGraph pipeline for each sample, collect results, and write outputs immediately after each sample completes.

#### Scenario: Process Batch Successfully
**Given** a batch of 5 samples
**When** batch processing runs
**Then** each sample invokes `run_pipeline()` from graph.py
**And** results are written to output JSONL after each sample
**And** execution continues even if individual samples fail

---

### Requirement: Execution Timeout
**Priority**: P1
**Capability**: batch-processing

The system SHALL enforce a configurable per-sample timeout to prevent hanging on slow or stuck samples.

#### Scenario: Timeout Long-Running Sample
**Given** a sample takes longer than the configured timeout (e.g., 5 minutes)
**When** the timeout expires
**Then** the sample execution is terminated
**And** an error is logged with "Execution timeout"
**And** processing continues with the next sample
