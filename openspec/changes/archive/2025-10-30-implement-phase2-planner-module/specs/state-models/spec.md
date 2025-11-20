# Capability: state-models

Pydantic models for LangGraph state management, ensuring type-safe data flow through the Planner→Executor→Summarizer pipeline.

## ADDED Requirements

### Requirement: Planner Output Model
The system SHALL define a Pydantic model for the Planner's structured JSON output.

#### Scenario: PlannerOutput model structure
- **Given** the need to validate Planner JSON output
- **When** defining the `PlannerOutput` Pydantic model
- **Then** it includes fields:
  - `query_type: Literal["IQA", "Other"]`
  - `query_scope: Union[List[str], Literal["Global"]]`
  - `distortion_source: Literal["Explicit", "Inferred"]`
  - `distortions: Optional[Dict[str, List[str]]]`
  - `reference_mode: Literal["Full-Reference", "No-Reference"]`
  - `required_tool: Optional[str]`
  - `plan: PlanControlFlags`
- **And** all fields have descriptive docstrings

#### Scenario: PlanControlFlags nested model
- **Given** the plan control flags structure
- **When** defining the `PlanControlFlags` model
- **Then** it includes boolean fields:
  - `distortion_detection: bool`
  - `distortion_analysis: bool`
  - `tool_selection: bool`
  - `tool_execution: bool`
- **And** each field has a default value or is required

#### Scenario: PlannerOutput validation
- **Given** raw JSON from VLM with all required fields
- **When** parsing with `PlannerOutput.model_validate_json(json_str)`
- **Then** it successfully creates a model instance
- **And** all field types are validated
- **Given** JSON with invalid `query_type` value (e.g., "INVALID")
- **Then** it raises a Pydantic ValidationError with specific field error

#### Scenario: PlannerOutput serialization
- **Given** a validated `PlannerOutput` instance
- **When** calling `.model_dump()` or `.model_dump_json()`
- **Then** it serializes to a Python dict or JSON string
- **And** preserves all field values and types
- **And** can be deserialized back to identical instance

### Requirement: Planner Input Model
The system SHALL define a Pydantic model for the Planner's input data.

#### Scenario: PlannerInput model structure
- **Given** the need to validate Planner inputs
- **When** defining the `PlannerInput` model
- **Then** it includes fields:
  - `query: str` (required, min_length=1)
  - `image_path: str` (required, valid file path)
  - `reference_path: Optional[str]` (optional, valid file path if provided)
  - `prior_context: Optional[Dict[str, Any]]` (for multi-turn conversations)
- **And** field validators check file existence

#### Scenario: File path validation
- **Given** an `image_path` that doesn't exist
- **When** validating a `PlannerInput` instance
- **Then** a validator raises a `ValueError` with message "Image file not found: {path}"
- **Given** a `reference_path` with invalid extension (e.g., ".txt")
- **Then** the validator raises a `ValueError` with "Invalid image format"

#### Scenario: Query validation
- **Given** an empty or whitespace-only query string
- **When** validating `PlannerInput`
- **Then** Pydantic raises a ValidationError for min_length constraint

### Requirement: AgenticIQA State TypedDict
The system SHALL define the LangGraph state structure as a TypedDict for type checking.

#### Scenario: AgenticIQAState structure (Phase 2 subset)
- **Given** the need for LangGraph state definition
- **When** defining `AgenticIQAState` TypedDict
- **Then** it includes fields for Phase 2:
  - `query: str`
  - `image_path: str`
  - `reference_path: NotRequired[str]`
  - `plan: NotRequired[PlannerOutput]`
  - `error: NotRequired[str]`
- **And** uses `NotRequired` for optional fields (Python 3.11+)
- **And** includes docstring describing state evolution through pipeline

#### Scenario: State evolution placeholder
- **Given** future Executor and Summarizer modules
- **When** documenting the state structure
- **Then** include comments indicating Phase 3+ fields:
  ```python
  # Phase 3: Executor outputs
  # executor_evidence: NotRequired[ExecutorOutput]
  # Phase 4: Summarizer outputs
  # final_answer: NotRequired[str]
  # quality_reasoning: NotRequired[str]
  ```

### Requirement: Error Handling Models
The system SHALL define models for error states and retry logic.

#### Scenario: PlannerError model
- **Given** the need to track Planner execution errors
- **When** defining the `PlannerError` model
- **Then** it includes fields:
  - `error_type: str` (e.g., "validation_error", "api_error")
  - `message: str` (human-readable error description)
  - `details: Optional[Dict[str, Any]]` (additional context)
  - `retry_count: int` (number of retries attempted)
  - `timestamp: datetime`
- **And** provides a factory method `from_exception(exc: Exception)`

#### Scenario: Retry metadata tracking
- **Given** a failed Planner attempt that will be retried
- **When** storing retry metadata
- **Then** the error model captures:
  - Original exception type and message
  - Attempt number (1-indexed)
  - Timestamp of failure
  - VLM backend used for the attempt
- **And** this metadata is logged for debugging

### Requirement: Model Export and Utilities
The system SHALL provide utility functions for model operations and JSON schema export.

#### Scenario: JSON schema generation
- **Given** the `PlannerOutput` model
- **When** calling `PlannerOutput.model_json_schema()`
- **Then** it returns a JSON schema dict conforming to JSON Schema Draft 2020-12
- **And** the schema can be used to validate external JSON
- **And** includes field descriptions and constraints

#### Scenario: Model examples for documentation
- **Given** the need for documentation and testing
- **When** accessing `PlannerOutput.model_config`
- **Then** it includes a `json_schema_extra` with example instances
- **Example**:
  ```python
  class PlannerOutput(BaseModel):
      model_config = {
          "json_schema_extra": {
              "examples": [{
                  "query_type": "IQA",
                  "query_scope": ["vehicle"],
                  ...
              }]
          }
      }
  ```

#### Scenario: Model comparison and equality
- **Given** two `PlannerOutput` instances with identical field values
- **When** comparing with `==` operator
- **Then** they are considered equal
- **And** hash values are identical if all fields are hashable

### Requirement: State Reducer Functions
The system SHALL provide reducer functions for merging state updates in LangGraph.

#### Scenario: Plan state update reducer
- **Given** a LangGraph node returns a partial state update with new `plan`
- **When** LangGraph merges the update into existing state
- **Then** the `plan` field is completely replaced (not merged)
- **And** other fields remain unchanged

#### Scenario: Error accumulation reducer
- **Given** multiple nodes may produce errors
- **When** defining the `error` field reducer
- **Then** it appends new errors to a list (if accumulating)
- **Or** replaces with the latest error (if not accumulating)
- **Decision**: Replace strategy for Phase 2 simplicity
