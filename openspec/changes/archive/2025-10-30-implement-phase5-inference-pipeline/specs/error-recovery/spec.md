## ADDED Requirements

### Requirement: Retry with Exponential Backoff
**Priority**: P0
**Capability**: error-recovery

The system SHALL retry failed API calls up to 3 times with exponential backoff for transient errors (rate limits, timeouts, server errors).

#### Scenario: Retry on Rate Limit
**Given** an API call returns 429 (rate limit)
**When** the first retry is attempted
**Then** the system waits 2 seconds before retrying
**When** the second retry fails with 429
**Then** the system waits 4 seconds before retrying
**When** the third retry fails
**Then** the sample is marked as failed with "Max retries exceeded"

#### Scenario: Fast Fail on Auth Error
**Given** an API call returns 401 (unauthorized)
**When** the error is detected
**Then** no retries are attempted
**And** the pipeline fails immediately with authentication error

---

### Requirement: Model Fallback
**Priority**: P1
**Capability**: error-recovery

The system SHALL support automatic fallback to alternative models when the primary model fails after retries.

#### Scenario: Fallback to Cheaper Model
**Given** primary model is GPT-4o and fallback is GPT-4o-mini
**When** GPT-4o fails after 3 retries
**Then** the system attempts with GPT-4o-mini
**And** logs "Model fallback: GPT-4o → GPT-4o-mini"

---

### Requirement: Tool Execution Fallback
**Priority**: P1
**Capability**: error-recovery

The system SHALL use fallback IQA tools when the primary tool fails, defaulting to BRISQUE for NR tasks and LPIPS for FR tasks.

#### Scenario: Fallback to BRISQUE
**Given** primary tool QAlign fails with CUDA out of memory
**When** the error is caught
**Then** BRISQUE is executed as fallback
**And** result includes flag: fallback=true, fallback_tool="BRISQUE"

---

### Requirement: Graceful Degradation
**Priority**: P1
**Capability**: error-recovery

The system SHALL continue batch processing when individual samples fail, logging errors and saving partial results without aborting the entire batch.

#### Scenario: Continue After Sample Failure
**Given** a batch of 100 samples where sample 42 fails
**When** the error is caught and logged
**Then** sample 42 is marked as failed in output
**And** processing continues with sample 43
**And** final summary shows "98 success, 2 failed"

---

### Requirement: Error Aggregation
**Priority**: P2
**Capability**: error-recovery

The system SHALL aggregate and summarize errors at the end of batch processing to identify common failure patterns.

#### Scenario: Summarize Error Types
**Given** a batch with 5 rate limit errors and 3 timeout errors
**When** processing completes
**Then** a summary is logged: "Errors: 5x RateLimitError, 3x TimeoutError"
**And** failed sample_ids are listed for each error type
