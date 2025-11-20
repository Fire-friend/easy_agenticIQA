# Phase 1 Implementation Tasks

## 1. Dependency Management
- [x] 1.1 Create `requirements.txt` with all core dependencies (torch, transformers, langgraph)
- [x] 1.2 Add IQA tools dependencies (pyiqa, opencv-python, pillow, numpy, scipy)
- [x] 1.3 Add API client dependencies (openai, anthropic, google-generativeai, qwen-vl-utils)
- [x] 1.4 Add utility dependencies (typer, rich, loguru, pyyaml, tqdm)
- [x] 1.5 Create optional `environment.yml` for conda users
- [x] 1.6 Document installation instructions with CUDA/CPU variants

## 2. Project Structure
- [x] 2.1 Create `scripts/` directory
- [x] 2.2 Create `src/agentic/` and `src/agentic/nodes/` directories
- [x] 2.3 Create `src/utils/` directory
- [x] 2.4 Create `iqa_tools/weights/` and `iqa_tools/metadata/` directories
- [x] 2.5 Create `data/raw/`, `data/processed/`, `data/cache/` directories
- [x] 2.6 Create dataset subdirectories (agenticiqa_eval, tid2013, bid, agiqa-3k)
- [x] 2.7 Create `logs/` directory
- [x] 2.8 Add placeholder `__init__.py` files for Python packages
- [x] 2.9 Create `.gitignore` with appropriate patterns

## 3. Environment Validation Script
- [x] 3.1 Create `scripts/check_env.py` skeleton with main function
- [x] 3.2 Implement Python version check (3.10.x)
- [x] 3.3 Implement virtual environment detection (conda/venv)
- [x] 3.4 Implement GPU/CUDA availability check
- [x] 3.5 Implement GPU memory check with warning for <24GB
- [x] 3.6 Implement core package version validation (torch, transformers, pydantic)
- [x] 3.7 Implement IQA tools package check (pyiqa)
- [x] 3.8 Implement optional package detection (API clients)
- [x] 3.9 Implement environment variable validation (AGENTIC_ROOT, etc.)
- [x] 3.10 Implement API key detection (without displaying values)
- [x] 3.11 Add --check-api flag for optional API connectivity tests
- [x] 3.12 Implement validation report generation (success/failure/warnings)
- [x] 3.13 Add colored output using rich library
- [x] 3.14 Add exit codes (0 for success, non-zero for failure)

## 4. Configuration Management
- [x] 4.1 Validate existing `configs/model_backends.yaml` structure
- [x] 4.2 Validate existing `configs/pipeline.yaml` structure
- [x] 4.3 Create configuration loading utility in `src/utils/config.py`
- [x] 4.4 Implement YAML parsing with error handling
- [x] 4.5 Implement environment variable interpolation (${VAR_NAME})
- [x] 4.6 Implement configuration schema validation using Pydantic
- [x] 4.7 Implement configuration merging with precedence rules
- [x] 4.8 Add configuration validation to check_env.py
- [x] 4.9 Document configuration file format and options

## 5. Directory Initialization
- [x] 5.1 Create `scripts/init_project.py` for automated directory creation
- [x] 5.2 Implement directory creation logic with permission checks
- [x] 5.3 Implement environment variable path resolution
- [x] 5.4 Add directory creation reporting (created vs existing)
- [x] 5.5 Add error handling for permission issues
- [x] 5.6 Make script idempotent (safe to run multiple times)

## 6. Documentation
- [x] 6.1 Create installation guide in comments or separate doc
- [x] 6.2 Document environment variable requirements
- [x] 6.3 Document API key setup procedures
- [x] 6.4 Add troubleshooting section for common issues
- [x] 6.5 Document alternative dependency versions for fallback
- [x] 6.6 Add IQA-PyTorch installation instructions
- [x] 6.7 Document GPU requirements and CPU-only alternatives

## 7. Testing and Validation
- [x] 7.1 Test `scripts/check_env.py` on clean environment
- [x] 7.2 Test with GPU available
- [x] 7.3 Test with CPU-only (no GPU)
- [x] 7.4 Test with partial dependencies (missing optional packages)
- [x] 7.5 Test with incorrect Python version
- [x] 7.6 Test with missing environment variables
- [x] 7.7 Test configuration loading and validation
- [x] 7.8 Test directory initialization script
- [x] 7.9 Verify all directories are created correctly
- [x] 7.10 Verify .gitignore patterns work correctly

## 8. Integration
- [x] 8.1 Run complete environment setup from scratch
- [x] 8.2 Verify all dependencies install without conflicts
- [x] 8.3 Verify check_env.py passes all checks
- [x] 8.4 Verify configurations load successfully
- [x] 8.5 Document Phase 1 completion checklist
- [x] 8.6 Prepare handoff notes for Phase 2 (Data Preparation)

## Dependencies
- All tasks in sections 1-2 should be completed before section 3
- Section 3 depends on section 1 (dependency management)
- Section 4 depends on section 2 (directory structure)
- Section 5 can be done in parallel with sections 3-4
- Section 6 should document as sections 1-5 are completed
- Section 7 depends on sections 1-5 being complete
- Section 8 is the final integration and validation phase

## Success Criteria
- [x] `python scripts/check_env.py` passes on target system
- [x] All required directories exist and are writable
- [x] Configuration files load and validate successfully
- [x] Dependencies install without version conflicts
- [x] GPU detection works correctly (if GPU available)
- [x] Documentation is complete and accurate
- [x] Ready for Phase 2 (Data Preparation) to begin
