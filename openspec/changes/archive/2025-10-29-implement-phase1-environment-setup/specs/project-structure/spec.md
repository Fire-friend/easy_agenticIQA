# Project Structure Specification

## ADDED Requirements

### Requirement: Standard Directory Layout
The system SHALL maintain a standard directory structure for all AgenticIQA components.

#### Scenario: Core directory creation
- **WHEN** the project is initialized
- **THEN** these directories SHALL exist:
  - `scripts/` for utility and validation scripts
  - `src/agentic/` for core LangGraph pipeline code
  - `src/agentic/nodes/` for agent node implementations
  - `src/utils/` for shared utility functions
  - `configs/` for YAML configuration files
- **AND** all directories SHALL be writable by the current user

#### Scenario: IQA tools directory structure
- **WHEN** the project is initialized
- **THEN** these IQA-related directories SHALL exist:
  - `iqa_tools/weights/` for model checkpoints
  - `iqa_tools/metadata/` for tool metadata JSON files
- **AND** directories SHALL align with AGENTIC_TOOL_HOME environment variable
- **AND** weights directory SHALL have sufficient disk space (>10GB recommended)

#### Scenario: Data directory structure
- **WHEN** the project is initialized
- **THEN** these data directories SHALL exist:
  - `data/raw/` for original datasets
  - `data/raw/agenticiqa_eval/` for MCQ evaluation data
  - `data/raw/tid2013/` for TID2013 dataset
  - `data/raw/bid/` for BID dataset
  - `data/raw/agiqa-3k/` for AGIQA-3K dataset
  - `data/processed/` for processed manifests
  - `data/cache/` for intermediate results
- **AND** directories SHALL align with AGENTIC_DATA_ROOT environment variable

#### Scenario: Logs directory structure
- **WHEN** the project is initialized
- **THEN** the `logs/` directory SHALL exist
- **AND** directory SHALL align with AGENTIC_LOG_ROOT environment variable
- **AND** directory SHALL be writable for log file creation

### Requirement: Directory Initialization Script
The system SHALL provide an automated script to create the directory structure.

#### Scenario: Automated directory creation
- **WHEN** the initialization script is executed
- **THEN** it SHALL create all missing directories
- **AND** preserve existing directories and their contents
- **AND** set appropriate permissions (755 for directories)
- **AND** verify each directory is writable after creation
- **AND** report which directories were created vs already existed

#### Scenario: Environment-aware initialization
- **WHEN** the initialization script is executed with environment variables set
- **THEN** it SHALL use environment variable paths for directory creation:
  - Use AGENTIC_ROOT as base path
  - Use AGENTIC_DATA_ROOT for data directories
  - Use AGENTIC_TOOL_HOME for IQA tools
  - Use AGENTIC_LOG_ROOT for logs
- **AND** create directories at the specified locations

#### Scenario: Initialization failure handling
- **WHEN** the initialization script encounters permission errors
- **THEN** it SHALL report which directories failed to create
- **AND** provide guidance on permission issues
- **AND** continue creating other directories
- **AND** exit with non-zero status if any directories failed

### Requirement: .gitignore Configuration
The system SHALL maintain appropriate .gitignore rules for generated and sensitive content.

#### Scenario: Ignore patterns for generated content
- **WHEN** git operations are performed
- **THEN** these patterns SHALL be ignored:
  - `logs/` (execution logs)
  - `data/cache/` (intermediate results)
  - `data/raw/*/` (large dataset files)
  - `iqa_tools/weights/` (model checkpoints)
  - `__pycache__/` and `*.pyc` (Python cache)
  - `.pytest_cache/` (test cache)
- **AND** git SHALL NOT track these files

#### Scenario: Ignore patterns for sensitive data
- **WHEN** git operations are performed
- **THEN** these patterns SHALL be ignored:
  - `.env` (environment variables)
  - `*.key` (API keys)
  - `*credentials*.json` (credentials files)
- **AND** prevent accidental commits of secrets

#### Scenario: Include patterns for configuration templates
- **WHEN** git operations are performed
- **THEN** these files SHALL be tracked:
  - `configs/*.yaml` (configuration templates)
  - `data/processed/*.jsonl` (processed manifests)
  - `.gitignore` itself
- **AND** allow version control of configuration examples
