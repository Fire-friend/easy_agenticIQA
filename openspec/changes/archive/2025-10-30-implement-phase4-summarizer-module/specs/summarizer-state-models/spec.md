# Spec: summarizer-state-models

## Purpose
Define Pydantic models and TypedDict extensions for Summarizer outputs and state management.

## ADDED Requirements

### Requirement: The system SHALL define a `SummarizerOutput` Pydantic model...
The system SHALL define a `SummarizerOutput` Pydantic model containing final_answer, quality_reasoning, and need_replan fields

#### Scenario: Create SummarizerOutput model with required fields
- **WHEN** Summarizer produces output
- **THEN** it SHALL use a Pydantic model with fields:
  - `final_answer: str` - The final answer (MCQ letter like "A" or quality level like "Excellent")
  - `quality_reasoning: str` - Evidence-based explanation (min_length=1)
  - `need_replan: bool` - Whether replanning is needed (default=False)
  - `replan_reason: Optional[str]` - Reason for replanning if need_replan=True
  - `used_evidence: Optional[Dict[str, Any]]` - Optional tracking of evidence items used

#### Scenario: Validate quality_reasoning is non-empty
- **WHEN** SummarizerOutput is created
- **THEN** quality_reasoning SHALL be validated to ensure it's not empty or whitespace-only
- **AND** whitespace SHALL be trimmed

#### Scenario: Serialize and deserialize SummarizerOutput
- **WHEN** SummarizerOutput needs JSON serialization
- **THEN** it SHALL support `model_dump_json()` for serialization
- **AND** `model_validate_json()` for deserialization
- **AND** round-trip serialization SHALL preserve all fields

### Requirement: extend AgenticIQAState TypedDict
The system SHALL extend AgenticIQAState TypedDict with summarizer_result and iteration tracking fields

#### Scenario: Add summarizer_result field to state
- **WHEN** AgenticIQAState is defined
- **THEN** it SHALL include `summarizer_result: NotRequired[SummarizerOutput]` field
- **AND** the field SHALL be optional (NotRequired) to support execution without Summarizer

#### Scenario: Add iteration tracking fields
- **WHEN** AgenticIQAState is defined for replanning support
- **THEN** it SHALL include:
  - `iteration_count: NotRequired[int]` - Current replanning iteration (starts at 0)
  - `max_replan_iterations: NotRequired[int]` - Maximum allowed iterations (default 2)
  - `replan_history: NotRequired[List[str]]` - History of replan reasons for debugging

#### Scenario: State merging with summarizer_result
- **WHEN** Summarizer node returns state update
- **THEN** LangGraph SHALL merge summarizer_result into state
- **AND** existing plan and executor_evidence SHALL be preserved
- **AND** iteration_count SHALL be incremented if replanning occurs

### Requirement: provide field validators
The system SHALL provide field validators for SummarizerOutput fields

#### Scenario: Validate final_answer format
- **WHEN** SummarizerOutput is created
- **THEN** final_answer SHALL be validated to ensure it's not empty
- **AND** whitespace SHALL be stripped

#### Scenario: Validate replan_reason presence
- **WHEN** need_replan=True in SummarizerOutput
- **THEN** replan_reason SHOULD be provided (logged warning if missing)
- **AND** if replan_reason is None, it SHALL be auto-set to "No reason provided"

#### Scenario: Validate quality_reasoning content
- **WHEN** quality_reasoning is set
- **THEN** it SHALL contain at least one non-whitespace character
- **AND** it SHALL raise ValidationError if empty after trimming

### Requirement: support backward compatibility
The system SHALL support backward compatibility with Phase 2 and Phase 3 states

#### Scenario: Execute pipeline without Summarizer
- **WHEN** A state has plan and executor_evidence but no summarizer_result
- **THEN** the state SHALL be valid
- **AND** downstream nodes SHALL handle missing summarizer_result gracefully

#### Scenario: Execute pipeline without Executor
- **WHEN** A state has plan but no executor_evidence (Phase 2 only)
- **THEN** the state SHALL be valid
- **AND** Summarizer SHALL not be called

### Requirement: provide JSON schema examples
The system SHALL provide JSON schema examples for SummarizerOutput

#### Scenario: Provide example for explanation/QA mode
- **WHEN** SummarizerOutput schema is documented
- **THEN** it SHALL include an example:
```json
{
  "final_answer": "B",
  "quality_reasoning": "The image shows moderate blur affecting sharpness, consistent with tool score of 2.6.",
  "need_replan": false
}
```

#### Scenario: Provide example for replanning scenario
- **WHEN** SummarizerOutput triggers replanning
- **THEN** it SHALL include an example:
```json
{
  "final_answer": "Unable to determine",
  "quality_reasoning": "Insufficient evidence for vehicle region.",
  "need_replan": true,
  "replan_reason": "Missing tool scores for vehicle region"
}
```

### Requirement: track replanning history
The system SHALL track replanning history for debugging and analysis

#### Scenario: Append replan reasons to history
- **WHEN** Summarizer sets need_replan=True
- **THEN** replan_reason SHALL be appended to state["replan_history"]
- **AND** history SHALL include iteration number and reason

#### Scenario: Limit history size
- **WHEN** replan_history exceeds 10 entries
- **THEN** oldest entries SHALL be removed (FIFO)
- **AND** warning SHALL be logged about excessive replanning
