# Phase 1: Environment Setup Implementation

## Why

AgenticIQA requires a properly configured development environment with specific dependencies, directory structures, and validation tools before any module development can begin. Phase 1 establishes the foundation by implementing:

1. **Environment validation**: Automated checks for Python version, GPU availability, required packages, and API connectivity
2. **Project structure**: Standard directory layout for tools, data, logs, and code modules
3. **Configuration management**: YAML-based configuration for model backends and pipeline settings
4. **Dependency management**: Comprehensive requirements specification and installation procedures

Without this foundation, subsequent phases (data preparation, module implementation, pipeline integration) cannot proceed reliably.

## What Changes

This proposal implements the complete Phase 1 environment setup as documented in `docs/01_environment_setup.md`:

- **Environment validation script** (`scripts/check_env.py`) to verify:
  - Python 3.10 environment
  - CUDA/GPU availability and driver compatibility
  - Required package versions (torch, transformers, langgraph, iqa-pytorch, etc.)
  - Environment variable configuration
  - Optional API connectivity tests for OpenAI/Anthropic/Google

- **Project directory structure**:
  - `scripts/` - Validation and utility scripts
  - `src/agentic/` - Core LangGraph pipeline (placeholder for Phase 3)
  - `iqa_tools/weights/` - IQA model checkpoints storage
  - `iqa_tools/metadata/` - Tool metadata for Executor module
  - `data/raw/` - Original evaluation datasets
  - `data/processed/` - Processed manifests
  - `data/cache/` - Intermediate results and prompt cache
  - `logs/` - Execution logs and traces

- **Configuration files**:
  - `configs/model_backends.yaml` (already exists, will be validated)
  - `configs/pipeline.yaml` (already exists, will be validated)
  - Optional: `configs/graph_settings.yaml` for LangGraph parameters

- **Dependency management**:
  - `requirements.txt` with pinned versions
  - Optional: `environment.yml` for conda users
  - Installation documentation and troubleshooting guidance

## Impact

### Affected Capabilities
- **NEW**: `environment-validation` - Automated environment checking
- **NEW**: `project-structure` - Standard directory layout
- **NEW**: `configuration-management` - YAML-based settings
- **NEW**: `dependency-management` - Package requirements specification

### Affected Code
- `scripts/check_env.py` (new) - Environment validation script
- `requirements.txt` (new) - Python dependencies
- `environment.yml` (new, optional) - Conda environment spec
- Directory structure creation (automated by scripts or manual)

### Dependencies
- Prerequisite for Phase 2 (Data Preparation)
- Prerequisite for Phase 3 (Module Implementation)
- Prerequisite for Phase 4 (Pipeline Integration)
- Prerequisite for Phase 5 (Evaluation & Testing)

### Success Criteria
1. `python scripts/check_env.py` passes all checks on target system
2. All required directories exist and are writable
3. Configuration files are valid YAML and pass schema validation
4. Dependencies install successfully without conflicts
5. GPU detection works correctly (if available)
6. Environment variables are documented and validated
