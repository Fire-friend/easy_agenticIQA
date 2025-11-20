## ADDED Requirements

### Requirement: JSONL Output Format
**Priority**: P0
**Capability**: result-management

The system SHALL write results in JSONL format where each line contains complete execution state for one sample including sample metadata, plan, executor evidence, summarizer result, and execution metadata.

#### Scenario: Write Complete Result
**Given** a sample completes successfully
**When** the result is written
**Then** the JSONL line contains: sample_id, query, image_path, plan, executor_evidence, summarizer_result, metadata
**And** the line is valid JSON

#### Scenario: Validate Result Schema
**Given** an output JSONL file
**When** validation runs
**Then** each line conforms to the expected schema
**And** required fields are present

---

### Requirement: Atomic Writes
**Priority**: P0
**Capability**: result-management

The system SHALL write each result atomically to prevent partial/corrupted entries in the output file.

#### Scenario: Ensure Atomic Write
**Given** a result is being written
**When** the process is interrupted mid-write
**Then** the output file contains either the complete entry or no entry
**And** no partial JSON lines exist

---

### Requirement: Immediate Persistence
**Priority**: P0
**Capability**: result-management

The system SHALL write results immediately after each sample completes to enable resume capability and minimize data loss.

#### Scenario: Checkpoint After Each Sample
**Given** a batch of 100 samples
**When** sample 50 completes
**Then** the result is written to disk before processing sample 51
**And** file handle is flushed to ensure persistence

---

### Requirement: Result Metadata
**Priority**: P1
**Capability**: result-management

The system SHALL include execution metadata with each result including execution time, replan count, final status, and error details (if any).

#### Scenario: Include Execution Metadata
**Given** a sample with 1 replan that took 12.45 seconds
**When** the result is written
**Then** metadata contains: execution_time_ms=12450, replan_count=1, final_status="success"

---

### Requirement: Partial Result Handling
**Priority**: P1
**Capability**: result-management

The system SHALL save partial results when execution fails mid-pipeline, marking the status as "partial" and including error information.

#### Scenario: Save Partial Results on Executor Failure
**Given** planner succeeds but executor fails
**When** the error is caught
**Then** the result includes plan (complete) and executor_evidence=null
**And** metadata.final_status="partial"
**And** metadata.error_details contains the error message
