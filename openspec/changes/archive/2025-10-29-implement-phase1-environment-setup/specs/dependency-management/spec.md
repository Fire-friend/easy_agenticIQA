# Dependency Management Specification

## ADDED Requirements

### Requirement: Python Package Requirements
The system SHALL specify all Python dependencies with version constraints in requirements.txt.

#### Scenario: Core dependencies specification
- **WHEN** requirements.txt is read
- **THEN** it SHALL include these core packages with exact or minimum versions:
  - torch==2.3.0 (with CUDA 12.1 index URL)
  - torchvision==0.18.0
  - transformers==4.42.0
  - accelerate==0.31.0
  - bitsandbytes==0.43.1
  - pydantic==2.7.1
- **AND** include installation index URLs for PyTorch

#### Scenario: LangGraph framework dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - langgraph (latest stable)
  - langchain-core (latest stable)
  - langchain-openai (for OpenAI integration)
  - langchain-anthropic (for Anthropic integration)

#### Scenario: IQA tools dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - pyiqa (IQA-PyTorch package)
  - opencv-python
  - pillow
  - numpy
  - scipy
  - scikit-image
  - einops

#### Scenario: API client dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - openai==1.35.7 (for GPT-4o API)
  - anthropic==0.30.0 (for Claude API)
  - google-generativeai==0.6.0 (for Gemini API)
  - qwen-vl-utils==0.0.8 (for local Qwen2.5-VL)

#### Scenario: Utility dependencies
- **WHEN** requirements.txt is read
- **THEN** it SHALL include:
  - typer (CLI framework)
  - rich (terminal formatting)
  - loguru (logging)
  - pyyaml (configuration parsing)
  - tqdm (progress bars)

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

#### Scenario: PyTorch and CUDA compatibility
- **WHEN** PyTorch is installed
- **THEN** torch version SHALL match CUDA version:
  - torch 2.3.0 requires CUDA 12.1
  - torchvision 0.18.0 requires torch 2.3.0
- **AND** installation SHALL fail clearly if CUDA driver is incompatible

#### Scenario: Transformers and model compatibility
- **WHEN** transformers is installed
- **THEN** version SHALL support:
  - Qwen2.5-VL model loading
  - Compatible with torch 2.3.0
  - Compatible with accelerate 0.31.0

#### Scenario: API client version compatibility
- **WHEN** API clients are installed
- **THEN** versions SHALL be compatible with:
  - Current OpenAI API specification
  - Current Anthropic API specification
  - Current Google Gemini API specification
- **AND** pin versions to avoid breaking changes

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
