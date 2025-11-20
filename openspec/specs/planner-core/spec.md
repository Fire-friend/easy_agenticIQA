# planner-core Specification

## Purpose
TBD - created by archiving change implement-phase2-planner-module. Update Purpose after archive.
## Requirements
### Requirement: Planner Output Schema
The Planner SHALL output a structured JSON plan conforming to the schema specified in the AgenticIQA paper.

#### Scenario: Valid plan JSON structure
- **Given** a user query "Is the vehicle blurry?" and an image
- **When** the Planner analyzes the query and image
- **Then** it returns a JSON object with required fields: `query_type`, `query_scope`, `distortion_source`, `distortions`, `reference_mode`, `required_tool`, `plan`
- **And** the `plan` object contains boolean flags: `distortion_detection`, `distortion_analysis`, `tool_selection`, `tool_execution`
- **And** all field types match the schema (e.g., `query_type` is "IQA" or "Other")
- **Example** output:
  ```json
  {
    "query_type": "IQA",
    "query_scope": ["vehicle"],
    "distortion_source": "Explicit",
    "distortions": {"vehicle": ["Blurs"]},
    "reference_mode": "No-Reference",
    "required_tool": null,
    "plan": {
      "distortion_detection": false,
      "distortion_analysis": true,
      "tool_selection": false,
      "tool_execution": false
    }
  }
  ```

#### Scenario: Explicit distortion handling
- **Given** a query explicitly mentioning a distortion type (e.g., "Is this image noisy?")
- **When** the Planner processes the query
- **Then** `distortion_source` is "Explicit"
- **And** `distortions` dict contains the mentioned distortion type
- **And** `plan.distortion_detection` is `false` (no need to detect, already known)

#### Scenario: Inferred distortion handling
- **Given** an open-ended query (e.g., "What's wrong with this image?")
- **When** the Planner processes the query
- **Then** `distortion_source` is "Inferred"
- **And** `distortions` may be `null` or contain initial hypotheses
- **And** `plan.distortion_detection` is `true` (need to detect distortions)

#### Scenario: Reference mode determination
- **Given** a reference image is provided alongside the distorted image
- **When** the Planner analyzes the inputs
- **Then** `reference_mode` is "Full-Reference"
- **Given** no reference image is provided
- **Then** `reference_mode` is "No-Reference"

#### Scenario: Query scope identification
- **Given** a query mentioning specific objects (e.g., "Is the car blurry?")
- **When** the Planner parses the query
- **Then** `query_scope` is a list containing ["car"]
- **Given** a global quality query (e.g., "How is the image quality?")
- **Then** `query_scope` is the string "Global"

### Requirement: Planner Prompt Template
The Planner SHALL use the prompt template from AgenticIQA paper Appendix A.2 to generate consistent plans.

#### Scenario: Prompt template structure
- **Given** the Planner needs to generate a plan
- **When** it constructs the VLM prompt
- **Then** the prompt uses the following template from `docs/02_module_planner.md` (paper Appendix A.2):
  ```text
  System:
  You are a planner in an image quality assessment (IQA) system. Your task is to analyze the user's query and
  generate a structured plan for downstream assessment.
  Return a valid JSON object in the following format:
  {
    "query_type": "IQA" or "Other",
    "query_scope": ["<object1>", "<object2>", ...] or "Global",
    "distortion_source": "Explicit" or "Inferred",
    "distortions": dict or null,
    "reference_mode": "Full-Reference" or "No-reference",
    "required_tool": "<tool_name>" or null,
    "plan": {
      "distortion_detection": true or false,
      "distortion_analysis": true or false,
      "tool_selection": true or false,
      "tool_execution": true or false
    }
  }

  User:
  User's query: {query}
  The image: <image>
  ```
- **And** the temperature is set to 0.0 for deterministic output
- **And** top_p is set to 0.1 for focused sampling
- **And** max_tokens is sufficient to cover the complete JSON output (recommended: 2048)

#### Scenario: Prompt with reference image
- **Given** both distorted and reference images are provided
- **When** constructing the prompt
- **Then** both images are included in the prompt
- **And** the prompt clarifies which is distorted and which is reference

### Requirement: JSON Output Parsing and Validation
The Planner SHALL robustly parse VLM JSON output and validate against the schema.

#### Scenario: Successful JSON parsing
- **Given** the VLM returns valid JSON matching the schema
- **When** the Planner parses the output
- **Then** it creates a `PlannerOutput` Pydantic model instance
- **And** all fields are validated according to their types
- **And** the parsed plan is returned without errors

#### Scenario: Malformed JSON handling
- **Given** the VLM returns malformed JSON (e.g., missing closing brace)
- **When** the Planner attempts to parse the output
- **Then** it catches the JSON parse error
- **And** it retries with a stricter prompt (up to 3 attempts)
- **And** if all retries fail, it raises a descriptive error

#### Scenario: Schema validation failure
- **Given** the VLM returns valid JSON but with incorrect field types
- **When** Pydantic validates the output
- **Then** it raises a validation error with specific field details
- **And** the Planner logs the error and retries with corrected prompt

### Requirement: Planner Node Function
The Planner SHALL be implemented as a LangGraph node function that accepts state and returns updated state.

#### Scenario: Planner node execution
- **Given** an `AgenticIQAState` with `query`, `image_path`, and optional `reference_path`
- **When** the Planner node function is invoked
- **Then** it loads the image(s) from disk
- **And** constructs the VLM prompt with query and image(s)
- **And** calls the VLM client to generate a plan
- **And** parses and validates the JSON output
- **And** returns the state with `plan` field populated

#### Scenario: Planner node error handling
- **Given** the image file does not exist
- **When** the Planner node loads the image
- **Then** it raises a FileNotFoundError with clear message
- **Given** the VLM API call fails (network error, rate limit)
- **When** the Planner attempts the API call
- **Then** it retries with exponential backoff (up to 3 attempts)
- **And** logs the error details
- **And** if all retries fail, it raises an exception

### Requirement: Planner Configuration Integration
The Planner SHALL load its VLM backend and parameters from `configs/model_backends.yaml`.

#### Scenario: Configuration loading
- **Given** `configs/model_backends.yaml` specifies `planner.backend: openai.gpt-4o`
- **When** the Planner initializes
- **Then** it loads the configuration using the Phase 1 config utilities
- **And** creates the appropriate VLM client (OpenAI)
- **And** applies the temperature, max_tokens, and top_p settings

#### Scenario: Fallback model configuration
- **Given** the primary VLM backend fails repeatedly
- **When** the Planner exhausts retries
- **Then** it checks for a `fallback_backend` in configuration
- **And** switches to the fallback backend if configured
- **And** logs the fallback event

