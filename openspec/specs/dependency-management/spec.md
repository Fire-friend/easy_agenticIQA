# dependency-management Specification

## Purpose
TBD - created by archiving change implement-phase1-environment-setup. Update Purpose after archive.
## Requirements
### Requirement: Python Package Requirements
The system SHALL specify all Python dependencies with version constraints in requirements.txt.

#### Scenario: API client dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - openai>=2.0.0,<3.0.0 (for GPT-4o API, upgraded from 1.35.7 for httpx 0.28+ compatibility)
  - anthropic>=0.72.0,<1.0.0 (for Claude API, upgraded from 0.30.0)
  - google-genai (Google's library for Gemini API, requires httpx>=0.28.1)
  - qwen-vl-utils==0.0.8 (for local Qwen2.5-VL)
  - httpx>=0.28.1,<1.0.0 (required by google-genai and compatible with openai 2.x)

#### Scenario: LangGraph framework dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - langgraph (latest stable)
  - langchain-core>=1.0.0,<2.0.0 (upgraded for compatibility)
  - langchain-openai>=1.0.0,<2.0.0 (for OpenAI integration, upgraded for openai 2.x support)
  - langchain-anthropic>=1.0.0,<2.0.0 (for Anthropic integration, upgraded for anthropic 0.72+ support)

### Requirement: Conda Environment Specification
The system SHALL provide conda environment.yml for reproducible conda-based setup.

#### Scenario: Conda environment YAML structure
- **WHEN** environment.yml is read
- **THEN** it SHALL specify:
  - name: agenticIQA
  - python: 3.10
  - channels: conda-forge, pytorch, nvidia
  - dependencies from conda (python, pip)
  - pip dependencies section matching requirements.txt

#### Scenario: CUDA-aware conda configuration
- **WHEN** environment.yml is read
- **THEN** it SHALL include:
  - pytorch-cuda=12.1 dependency for GPU support
  - cudatoolkit specification for CUDA libraries
- **AND** document CPU-only alternative in comments

### Requirement: IQA-PyTorch Installation
The system SHALL provide clear instructions for IQA-PyTorch setup and integration.

#### Scenario: IQA-PyTorch installation from source
- **WHEN** IQA-PyTorch is installed
- **THEN** documentation SHALL provide:
  - Git clone command with repository URL
  - pip install -e . command for editable installation
  - Expected installation path
  - Verification command to check installation

#### Scenario: IQA tool weight download
- **WHEN** IQA tools are first used
- **THEN** documentation SHALL explain:
  - Auto-download behavior on first use
  - Manual download procedure for offline setup
  - Expected weight file locations in iqa_tools/weights/
  - Disk space requirements (>10GB)

### Requirement: Dependency Version Compatibility
The system SHALL ensure all dependencies are mutually compatible.

#### Scenario: API client version compatibility
- **WHEN** API clients are installed
- **THEN** versions SHALL be compatible with:
  - openai 2.x API (breaking changes from 1.x handled by VLM abstraction layer)
  - anthropic 0.72.x API (updated from 0.30.0)
  - google-genai latest (requires httpx>=0.28.1)
  - httpx 0.28.x (proxies parameter removed, handled by openai 2.x)
- **AND** pin versions to avoid breaking changes

#### Scenario: HTTP client dependency resolution
- **WHEN** httpx is installed
- **THEN** version SHALL satisfy all dependent packages:
  - google-genai requires httpx>=0.28.1
  - openai 2.x is compatible with httpx 0.28.x (removed proxies parameter usage)
  - anthropic 0.72.x is compatible with httpx 0.28.x
  - langchain packages use httpx through API client libraries
- **AND** no conflicting proxy configuration attempts

#### Scenario: LangChain integration compatibility
- **WHEN** langchain packages are installed
- **THEN** versions SHALL be compatible with:
  - langchain-core 1.0.x (upgraded from 0.2.x)
  - langchain-openai 1.0.x with openai 2.x support
  - langchain-anthropic 1.0.x with anthropic 0.72.x support
- **AND** no version conflicts between langchain-* packages

### Requirement: Dependency Installation Validation
The system SHALL validate successful dependency installation.

#### Scenario: Post-installation verification
- **WHEN** dependencies are installed
- **THEN** verification script SHALL:
  - Import each critical package
  - Check __version__ attribute matches expected version
  - Test basic functionality (e.g., torch.cuda.is_available())
  - Report any import or version mismatches

#### Scenario: Dependency conflict detection
- **WHEN** pip install is executed
- **THEN** installation SHALL:
  - Detect conflicting version requirements
  - Report which packages conflict and why
  - Suggest resolution steps
  - Fail clearly rather than install incompatible versions

### Requirement: Optional Dependency Handling
The system SHALL gracefully handle missing optional dependencies.

#### Scenario: API client optional dependencies
- **WHEN** only subset of API clients are installed
- **THEN** the system SHALL:
  - Detect which clients are available
  - Only load backends for available clients
  - Provide clear error if unavailable backend is requested
  - Suggest installation command for missing client

#### Scenario: Local model optional dependencies
- **WHEN** qwen-vl-utils or other local model dependencies are missing
- **THEN** the system SHALL:
  - Detect missing dependencies
  - Disable local model backends
  - Provide clear error if local backend is requested
  - Suggest installation command for missing dependencies

