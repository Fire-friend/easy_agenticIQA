# executor-state-models Specification

## Purpose
TBD - created by archiving change implement-phase3-executor-module. Update Purpose after archive.
## Requirements
### Requirement: DistortionAnalysis Model
The system SHALL define a Pydantic model for distortion analysis results.

#### Scenario: DistortionAnalysis structure
- **WHEN** DistortionAnalysis model is instantiated
- **THEN** it SHALL have field `type: str` for distortion type
- **AND** it SHALL have field `severity: Literal["none", "slight", "moderate", "severe", "extreme"]`
- **AND** it SHALL have field `explanation: str` for brief visual explanation
- **AND** all fields SHALL be required

#### Scenario: Validate severity levels
- **WHEN** DistortionAnalysis is created with severity value
- **THEN** severity SHALL be one of: none, slight, moderate, severe, extreme
- **AND** raise ValidationError if severity is invalid

#### Scenario: Validate explanation is non-empty
- **WHEN** DistortionAnalysis is created with explanation
- **THEN** explanation SHALL not be empty string
- **AND** explanation SHALL not be only whitespace
- **AND** raise ValidationError if explanation is empty

### Requirement: ToolExecutionLog Model
The system SHALL define a Pydantic model for recording tool execution details.

#### Scenario: ToolExecutionLog structure
- **WHEN** ToolExecutionLog model is instantiated
- **THEN** it SHALL have field `tool_name: str` for IQA tool identifier
- **AND** it SHALL have field `object_name: str` for query scope object (or "Global")
- **AND** it SHALL have field `distortion: str` for distortion type
- **AND** it SHALL have field `raw_score: float` for unnormalized tool output
- **AND** it SHALL have field `normalized_score: float` for [1, 5] normalized score
- **AND** it SHALL have field `execution_time: float` for tool runtime in seconds
- **AND** it SHALL have field `fallback: bool = False` for whether tool was a fallback
- **AND** it SHALL have field `error: Optional[str] = None` for error messages
- **AND** it SHALL have field `timestamp: datetime` for execution timestamp

#### Scenario: Validate score ranges
- **WHEN** ToolExecutionLog is created with scores
- **THEN** normalized_score SHALL be within [1, 5] range
- **AND** execution_time SHALL be non-negative
- **AND** raise ValidationError if constraints violated

#### Scenario: Record failed executions
- **WHEN** Tool execution fails
- **THEN** ToolExecutionLog SHALL have error field populated
- **AND** raw_score and normalized_score MAY be NaN or special values
- **AND** fallback flag SHALL indicate if fallback tool was used

### Requirement: ExecutorOutput Model
The system SHALL define a Pydantic model for the complete Executor output structure.

#### Scenario: ExecutorOutput structure
- **WHEN** ExecutorOutput model is instantiated
- **THEN** it SHALL have field `distortion_set: Optional[Dict[str, List[str]]]` for detected distortions
- **AND** it SHALL have field `distortion_analysis: Optional[Dict[str, List[DistortionAnalysis]]]` for severity analysis
- **AND** it SHALL have field `selected_tools: Optional[Dict[str, Dict[str, str]]]` for tool assignments
- **AND** it SHALL have field `quality_scores: Optional[Dict[str, Dict[str, Tuple[str, float]]]]` for (tool_name, normalized_score) pairs
- **AND** it SHALL have field `tool_logs: List[ToolExecutionLog]` with default empty list
- **AND** all Optional fields SHALL default to None

#### Scenario: Validate distortion_set structure
- **WHEN** ExecutorOutput is created with distortion_set
- **THEN** keys SHALL be object names or "Global"
- **AND** values SHALL be lists of distortion type strings
- **AND** distortion types SHALL be from valid categories list

#### Scenario: Validate distortion_analysis structure
- **WHEN** ExecutorOutput is created with distortion_analysis
- **THEN** keys SHALL be object names or "Global"
- **AND** values SHALL be lists of DistortionAnalysis instances
- **AND** each DistortionAnalysis SHALL have valid type and severity

#### Scenario: Validate selected_tools structure
- **WHEN** ExecutorOutput is created with selected_tools
- **THEN** outer keys SHALL be object names or "Global"
- **AND** inner keys SHALL be distortion types
- **AND** values SHALL be tool names (strings)
- **AND** tool names SHALL exist in ToolRegistry metadata

#### Scenario: Validate quality_scores structure
- **WHEN** ExecutorOutput is created with quality_scores
- **THEN** outer keys SHALL be object names or "Global"
- **AND** inner keys SHALL be distortion types
- **AND** values SHALL be tuples of (tool_name: str, score: float)
- **AND** scores SHALL be in [1, 5] range

### Requirement: AgenticIQAState Extension
The system SHALL extend AgenticIQAState TypedDict to include Executor output.

#### Scenario: Add executor_evidence field
- **WHEN** AgenticIQAState is defined
- **THEN** it SHALL include field `executor_evidence: NotRequired[ExecutorOutput]`
- **AND** field SHALL be optional (NotRequired) for backward compatibility with Phase 2
- **AND** field SHALL contain ExecutorOutput when Executor node completes

#### Scenario: State merging with executor evidence
- **WHEN** Executor node returns state update with executor_evidence
- **THEN** LangGraph SHALL merge it into AgenticIQAState
- **AND** existing fields (query, image_path, plan) SHALL remain unchanged
- **AND** executor_evidence SHALL be accessible to downstream nodes (Summarizer)

### Requirement: State Serialization
The system SHALL ensure all Executor state models support JSON serialization for LangGraph checkpointing.

#### Scenario: Serialize ExecutorOutput to JSON
- **WHEN** ExecutorOutput.model_dump_json() is called
- **THEN** it SHALL produce valid JSON string
- **AND** nested DistortionAnalysis SHALL be serialized
- **AND** ToolExecutionLog timestamps SHALL be ISO 8601 strings
- **AND** Optional None values SHALL be represented as null

#### Scenario: Deserialize ExecutorOutput from JSON
- **WHEN** ExecutorOutput.model_validate_json(json_string) is called
- **THEN** it SHALL reconstruct ExecutorOutput instance
- **AND** nested models SHALL be properly instantiated
- **AND** datetime fields SHALL be parsed from ISO strings
- **AND** raise ValidationError if JSON structure is invalid

#### Scenario: Round-trip serialization
- **WHEN** ExecutorOutput is serialized and deserialized
- **THEN** reconstructed instance SHALL equal original
- **AND** all field values SHALL be preserved exactly
- **AND** nested models SHALL maintain structure

### Requirement: State Validation
The system SHALL validate Executor state models to catch errors early.

#### Scenario: Validate field types
- **WHEN** Any Executor model is instantiated
- **THEN** Pydantic SHALL validate field types match annotations
- **AND** raise ValidationError for type mismatches
- **AND** provide clear error message indicating field and expected type

#### Scenario: Validate nested structures
- **WHEN** ExecutorOutput contains nested dictionaries
- **THEN** Pydantic SHALL validate nested structure recursively
- **AND** check DistortionAnalysis instances are valid
- **AND** check ToolExecutionLog instances are valid

#### Scenario: Custom validators for domain constraints
- **WHEN** Executor models define custom validators
- **THEN** validators SHALL check domain-specific constraints
- **AND** distortion types SHALL be from valid categories
- **AND** severity levels SHALL be from valid levels
- **AND** scores SHALL be in valid ranges
- **AND** raise ValidationError with descriptive messages

