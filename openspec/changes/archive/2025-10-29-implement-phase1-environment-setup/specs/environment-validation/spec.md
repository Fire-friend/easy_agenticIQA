# Environment Validation Specification

## ADDED Requirements

### Requirement: Python Environment Validation
The system SHALL validate the Python environment meets AgenticIQA requirements before any module execution.

#### Scenario: Python version check
- **WHEN** the validation script is executed
- **THEN** it SHALL verify Python version is 3.10.x
- **AND** display the exact Python version
- **AND** fail with clear error message if version is incorrect

#### Scenario: Virtual environment detection
- **WHEN** the validation script is executed
- **THEN** it SHALL detect if running in a virtual environment (conda/venv)
- **AND** display the environment name if detected
- **AND** warn if not in a virtual environment

### Requirement: GPU and CUDA Validation
The system SHALL validate GPU availability and CUDA compatibility for local model inference.

#### Scenario: GPU availability check
- **WHEN** the validation script is executed
- **THEN** it SHALL check if CUDA is available via torch.cuda.is_available()
- **AND** display GPU name and compute capability if available
- **AND** display CUDA version if available
- **AND** continue without error if GPU is not available (API-only mode)

#### Scenario: GPU memory check
- **WHEN** GPU is available
- **THEN** the validation script SHALL display total GPU memory
- **AND** warn if memory is less than 24GB for local VLM inference

### Requirement: Package Dependency Validation
The system SHALL verify all required Python packages are installed with correct versions.

#### Scenario: Core package version check
- **WHEN** the validation script is executed
- **THEN** it SHALL check for these packages with versions:
  - torch >= 2.3.0
  - transformers >= 4.42.0
  - langgraph (any version)
  - langchain-core (any version)
  - pydantic >= 2.7.0
- **AND** display installed version for each package
- **AND** fail with clear error message if any required package is missing

#### Scenario: IQA tools validation
- **WHEN** the validation script is executed
- **THEN** it SHALL check if iqa-pytorch (pyiqa) package is installed
- **AND** display installed version
- **AND** fail if iqa-pytorch is not available

#### Scenario: Optional package detection
- **WHEN** the validation script is executed
- **THEN** it SHALL check for optional packages:
  - openai (for GPT-4o API)
  - anthropic (for Claude API)
  - google-generativeai (for Gemini API)
  - qwen-vl-utils (for local Qwen2.5-VL)
- **AND** display which packages are available
- **AND** warn but not fail if optional packages are missing

### Requirement: Environment Variable Validation
The system SHALL validate required environment variables are configured correctly.

#### Scenario: Required environment variables check
- **WHEN** the validation script is executed
- **THEN** it SHALL check these environment variables exist:
  - AGENTIC_ROOT
  - AGENTIC_DATA_ROOT
  - AGENTIC_TOOL_HOME
  - AGENTIC_LOG_ROOT
- **AND** verify each path exists and is writable
- **AND** display the configured path for each variable
- **AND** fail with clear error if any required variable is missing

#### Scenario: API key validation
- **WHEN** the validation script is executed with --check-api flag
- **THEN** it SHALL check for API key environment variables:
  - OPENAI_API_KEY
  - ANTHROPIC_API_KEY
  - GOOGLE_API_KEY
- **AND** indicate which API keys are configured (without displaying values)
- **AND** optionally test API connectivity if requested
- **AND** warn but not fail if no API keys are configured

#### Scenario: Custom endpoint validation
- **WHEN** the validation script is executed
- **THEN** it SHALL check for optional custom endpoint variables:
  - OPENAI_BASE_URL
  - ANTHROPIC_BASE_URL
  - GOOGLE_API_BASE_URL
- **AND** display configured custom endpoints if any
- **AND** warn if custom endpoint is unreachable (when --check-api is used)

### Requirement: Validation Report Generation
The system SHALL generate a comprehensive validation report with actionable guidance.

#### Scenario: Successful validation report
- **WHEN** all validation checks pass
- **THEN** the script SHALL display a success summary
- **AND** list all validated components with their versions
- **AND** exit with status code 0

#### Scenario: Failed validation report
- **WHEN** any validation check fails
- **THEN** the script SHALL display a failure summary
- **AND** list all failures with specific error messages
- **AND** provide actionable remediation steps for each failure
- **AND** exit with non-zero status code

#### Scenario: Warning-only report
- **WHEN** validation completes with warnings but no errors
- **THEN** the script SHALL display a warning summary
- **AND** list all warnings with recommendations
- **AND** exit with status code 0 (success with warnings)
