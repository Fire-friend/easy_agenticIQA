# User Documentation Specification

## ADDED Requirements

### Requirement: Project Overview Documentation
The README.md file SHALL provide a comprehensive project overview that includes the system's purpose, architecture, and key features.

#### Scenario: First-time user reads README
- **WHEN** a new user opens the README.md file
- **THEN** they shall understand that AgenticIQA is an agentic framework for Image Quality Assessment
- **AND** they shall see the Planner-Executor-Summarizer architecture explained
- **AND** they shall understand the key capabilities (scoring, explanation, MCQ tasks)

### Requirement: File and Directory Reference
The README.md file SHALL document the purpose and contents of all major files and directories in the project.

#### Scenario: Developer locates relevant files
- **WHEN** a developer needs to find where planner logic is implemented
- **THEN** the README shall indicate that `src/agentic/nodes/planner.py` contains the Planner agent node
- **AND** the README shall explain that `configs/` contains YAML configuration files
- **AND** the README shall describe the purpose of `iqa_tools/`, `data/`, `scripts/`, and other key directories

#### Scenario: User understands configuration files
- **WHEN** a user wants to know what configuration files exist
- **THEN** the README shall list `model_backends.yaml`, `pipeline.yaml`, and `graph_settings.yaml`
- **AND** for each configuration file, the README shall explain its purpose

### Requirement: Environment Setup Instructions
The README.md file SHALL provide complete instructions for setting up the development environment.

#### Scenario: New user sets up environment
- **WHEN** a user wants to set up the project for the first time
- **THEN** the README shall provide step-by-step conda environment creation commands
- **AND** the README shall list all required dependencies with exact versions
- **AND** the README shall explain how to install IQA-PyTorch
- **AND** the README shall document required environment variables (AGENTIC_ROOT, AGENTIC_DATA_ROOT, AGENTIC_TOOL_HOME, AGENTIC_LOG_ROOT)
- **AND** the README shall explain how to configure API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY)

#### Scenario: User configures custom API endpoints
- **WHEN** a user wants to use custom API endpoints
- **THEN** the README shall document the optional environment variables (OPENAI_BASE_URL, ANTHROPIC_BASE_URL, GOOGLE_API_BASE_URL)

### Requirement: Configuration Instructions
The README.md file SHALL explain how to configure the system for different use cases.

#### Scenario: User configures VLM backends
- **WHEN** a user wants to change which VLM models are used
- **THEN** the README shall explain how to edit `configs/model_backends.yaml`
- **AND** the README shall provide examples of configuring different backends (GPT-4o, Claude 3.5, Qwen2.5-VL)
- **AND** the README shall explain the temperature settings

#### Scenario: User configures pipeline settings
- **WHEN** a user wants to adjust pipeline behavior
- **THEN** the README shall explain the `configs/pipeline.yaml` settings
- **AND** the README shall document the max_replan parameter and its purpose

### Requirement: Usage Instructions
The README.md file SHALL provide clear instructions for using the system to perform IQA tasks.

#### Scenario: User runs pipeline on AgenticIQA-Eval dataset
- **WHEN** a user wants to evaluate the system on AgenticIQA-Eval
- **THEN** the README shall provide the exact command to run the pipeline
- **AND** the README shall explain the expected input format (JSONL manifest)
- **AND** the README shall describe the output format

#### Scenario: User evaluates on scoring datasets
- **WHEN** a user wants to calculate SRCC/PLCC on TID2013, BID, or AGIQA-3K
- **THEN** the README shall provide example commands for running scoring evaluation
- **AND** the README shall explain how to interpret the results

#### Scenario: User runs evaluation scripts
- **WHEN** a user wants to calculate evaluation metrics
- **THEN** the README shall document the `scripts/eval_agenticqa_eval.py` command for MCQ accuracy
- **AND** the README shall document the `scripts/eval_srocc_plcc.py` command for correlation metrics
- **AND** the README shall explain the purpose of `scripts/check_env.py` for environment validation

### Requirement: Development Workflow Documentation
The README.md file SHALL document common development workflows and commands.

#### Scenario: Developer runs environment validation
- **WHEN** a developer wants to verify their environment is correctly set up
- **THEN** the README shall document the `python scripts/check_env.py` command
- **AND** the README shall explain what this command checks

#### Scenario: Developer debugs with cheaper models
- **WHEN** a developer wants to reduce API costs during debugging
- **THEN** the README shall suggest using GPT-4o-mini or other cheaper alternatives
- **AND** the README shall explain how to configure these in model_backends.yaml

### Requirement: Troubleshooting and FAQ
The README.md file SHALL include a troubleshooting section addressing common issues.

#### Scenario: User encounters missing dependencies
- **WHEN** a user encounters import errors or missing tools
- **THEN** the README shall provide guidance on verifying dependencies
- **AND** the README shall reference the environment validation script

#### Scenario: User has API rate limiting issues
- **WHEN** a user encounters rate limit errors
- **THEN** the README shall explain caching strategies
- **AND** the README shall suggest batch processing with checkpointing

### Requirement: Documentation Cross-References
The README.md file SHALL provide links to detailed documentation for users who need more information.

#### Scenario: User needs detailed implementation information
- **WHEN** a user wants to understand implementation details
- **THEN** the README shall reference the `docs/` directory
- **AND** the README shall list the available detailed documentation files (00_overview.md through 06_evaluation_protocol.md)
- **AND** the README shall reference CLAUDE.md for AI assistant guidance
- **AND** the README shall reference the paper.pdf for academic details

### Requirement: Quick Start Guide
The README.md file SHALL provide a quick start guide that allows users to get up and running with minimal steps.

#### Scenario: User wants to run a simple example
- **WHEN** a user wants to quickly test the system
- **THEN** the README shall provide a minimal quick start sequence
- **AND** the quick start shall include environment setup, configuration, and a simple pipeline run
- **AND** the quick start shall take less than 10 minutes to complete
