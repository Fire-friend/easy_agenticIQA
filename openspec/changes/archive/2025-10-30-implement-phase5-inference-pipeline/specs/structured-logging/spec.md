## ADDED Requirements

### Requirement: JSON Lines Log Format
**Priority**: P0
**Capability**: structured-logging

The system SHALL write execution logs in JSON Lines format where each line is a valid JSON object containing execution metadata for one sample.

#### Scenario: Log Sample Execution
**Given** a sample completes successfully
**When** the log entry is written
**Then** the entry contains: timestamp, sample_id, execution_time_ms, backends, replan_count, status
**And** the log file can be parsed line-by-line as JSON

#### Scenario: Parse Execution Logs
**Given** a log file with 100 entries
**When** a monitoring script parses the file
**Then** each line is valid JSON
**And** aggregated metrics can be computed (avg time, error rate, etc.)

---

### Requirement: Execution Metrics Tracking
**Priority**: P1
**Capability**: structured-logging

The system SHALL track and log execution metrics per sample including execution time, token usage per agent, estimated cost, and replanning count.

#### Scenario: Track Token Usage
**Given** a sample with planner using 245 tokens, executor 1024 tokens, summarizer 512 tokens
**When** execution completes
**Then** the log entry contains `"tokens_used": {"planner": 245, "executor": 1024, "summarizer": 512}`
**And** total tokens are summed: 1781

#### Scenario: Estimate API Costs
**Given** token usage of 1781 tokens with GPT-4o (input: $2.50/1M, output: $10/1M)
**When** cost is calculated
**Then** estimated cost is logged (assuming 50/50 input/output split)
**And** `cost_usd` field contains the estimate

---

### Requirement: Logging Levels
**Priority**: P1
**Capability**: structured-logging

The system SHALL support three logging levels: INFO (summary only), DEBUG (include prompts/responses), and TRACE (full state dumps).

#### Scenario: INFO Level Logging
**Given** logging level set to INFO
**When** a sample is processed
**Then** only summary metrics are logged (no prompts or responses)
**And** log size is minimal (~200 bytes per sample)

#### Scenario: DEBUG Level Logging
**Given** logging level set to DEBUG
**When** a sample is processed
**Then** full prompts and VLM responses are included
**And** log size is moderate (~5-10 KB per sample)

#### Scenario: TRACE Level Logging
**Given** logging level set to TRACE
**When** a sample is processed
**Then** complete state dumps are included (plan, evidence, intermediate states)
**And** log size is large (~20-50 KB per sample)

---

### Requirement: Log Rotation
**Priority**: P2
**Capability**: structured-logging

The system SHALL rotate log files based on size or date to prevent unbounded growth.

#### Scenario: Rotate by Size
**Given** log rotation configured for 100 MB max size
**When** the current log file exceeds 100 MB
**Then** the file is renamed to `pipeline.log.1`
**And** a new `pipeline.log` is created
**And** old rotated files are deleted if count exceeds limit

---

### Requirement: Error and Warning Logging
**Priority**: P0
**Capability**: structured-logging

The system SHALL log errors and warnings with full stack traces and context to aid debugging.

#### Scenario: Log API Error
**Given** an API call fails with rate limit (429)
**When** the error is logged
**Then** the log entry contains: error_type, error_message, stack_trace, retry_count
**And** `status` field is set to "error"

#### Scenario: Log Validation Warning
**Given** a plan JSON is missing a recommended field
**When** validation runs
**Then** a warning is logged with field name and sample_id
**And** `status` remains "success" (warning, not error)
