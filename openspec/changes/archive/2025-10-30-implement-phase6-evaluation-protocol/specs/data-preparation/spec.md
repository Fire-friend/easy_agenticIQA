## ADDED Requirements

### Requirement: Manifest Schema Definitions
**Priority**: P0
**Capability**: data-preparation

The system SHALL define JSON Schema files for each supported dataset type (TID2013, BID, AGIQA-3K, AgenticIQA-Eval) that specify required fields, data types, validation rules, and value constraints.

#### Scenario: Validate TID2013 Manifest
**Given** a TID2013 manifest with sample_id, distorted_path, reference_path, and mos fields
**When** the manifest is validated against tid2013_schema.json
**Then** validation passes if all required fields are present with correct types
**And** mos value is between 0 and 9
**And** file paths follow the expected pattern

#### Scenario: Reject Invalid Manifest
**Given** a manifest missing required field "reference_path"
**When** validation is performed
**Then** validation fails with clear error message indicating missing field
**And** line number is reported for JSONL format

---

### Requirement: TID2013 Manifest Generation
**Priority**: P0
**Capability**: data-preparation

The system SHALL provide a script to generate JSONL manifests for TID2013 dataset by reading the directory structure, parsing MOS annotations, and mapping to the standardized schema format.

#### Scenario: Generate TID2013 Manifest from Raw Data
**Given** TID2013 raw dataset at `data/raw/tid2013/` with distorted/reference subdirectories
**And** a mos.csv file with sample IDs and MOS scores
**When** `generate_tid2013_manifest.py` is executed
**Then** a JSONL manifest is created at `data/processed/tid2013/manifest.jsonl`
**And** each line contains sample_id, distorted_path, reference_path, mos, and metadata
**And** all file paths are validated to exist
**And** manifest contains 3000 samples (25 references × 24 distortions × 5 levels)

#### Scenario: Handle Missing Reference Image
**Given** TID2013 dataset with missing reference image "I05.bmp"
**When** manifest generation runs
**Then** an error is logged for the missing reference
**And** affected distorted images are skipped
**And** generation continues with remaining samples

---

### Requirement: BID Manifest Generation
**Priority**: P0
**Capability**: data-preparation

The system SHALL provide a script to generate JSONL manifests for BID dataset handling No-Reference (NR) scenarios where no reference images exist.

#### Scenario: Generate BID Manifest
**Given** BID raw dataset at `data/raw/bid/` with distorted images only
**And** annotations file with MOS scores
**When** `generate_bid_manifest.py` is executed
**Then** a JSONL manifest is created with sample_id, distorted_path, mos (no reference_path)
**And** manifest contains 586 samples
**And** split field indicates train/val/test division

---

### Requirement: AGIQA-3K Manifest Generation
**Priority**: P0
**Capability**: data-preparation

The system SHALL provide a script to generate JSONL manifests for AGIQA-3K dataset including generation-specific metadata (prompt, model, sampling parameters).

#### Scenario: Generate AGIQA-3K Manifest
**Given** AGIQA-3K dataset at `data/raw/agiqa-3k/` with generated images
**And** metadata JSON with MOS and generation parameters
**When** `generate_agiqa3k_manifest.py` is executed
**Then** manifest includes generation metadata (generator_model, prompt, seed)
**And** manifest contains 3000 samples
**And** MOS values are mapped from original scale to 1-5

---

### Requirement: AgenticIQA-Eval Manifest Generation
**Priority**: P0
**Capability**: data-preparation

The system SHALL provide a script to generate JSONL manifests for AgenticIQA-Eval MCQ dataset split by task type (planner, executor_distortion, executor_tool, summarizer).

#### Scenario: Generate AgenticIQA-Eval Manifests
**Given** AgenticIQA-Eval raw MCQ questions with task_type annotations
**When** `generate_agenticiqa_eval_manifest.py` is executed
**Then** four JSONL files are created:
  - planner.jsonl (questions testing planner capability)
  - executor_distortion.jsonl (distortion detection questions)
  - executor_tool.jsonl (tool selection questions)
  - summarizer.jsonl (reasoning questions)
**And** each question includes: question, options (A/B/C/D), answer, task_type, reference_mode
**And** total questions = 750 NR + 250 FR = 1000

#### Scenario: Validate MCQ Options Format
**Given** an MCQ question with options ["A) Option 1", "B) Option 2", ...]
**When** manifest is generated
**Then** options are normalized to consistent format
**And** answer is validated to be one of the option keys (A/B/C/D)

---

### Requirement: Manifest Validation Tool
**Priority**: P0
**Capability**: data-preparation

The system SHALL provide a validation tool that checks JSONL manifests against schemas, verifies file path existence, validates value ranges, and reports errors with line numbers.

#### Scenario: Validate Manifest Against Schema
**Given** a JSONL manifest file
**And** corresponding JSON Schema file
**When** `validate_manifest.py --schema tid2013_schema.json --manifest data/processed/tid2013/manifest.jsonl` is executed
**Then** each line is validated against the schema
**And** validation errors report line number, field name, and constraint violation
**And** exit code is 0 if all lines valid, non-zero otherwise

#### Scenario: Validate File Path Existence
**Given** a manifest with image_path fields
**When** validation runs with `--check-paths` flag
**Then** each file path is checked for existence
**And** missing files are reported with sample_id
**And** summary shows count of missing files

#### Scenario: Strict Mode Validation
**Given** a manifest with one invalid line
**When** validation runs with `--strict` flag
**Then** validation stops at first error
**And** clear error message is displayed
**And** exit code indicates failure

---

### Requirement: Common Schema Fields
**Priority**: P1
**Capability**: data-preparation

The system SHALL define common field schemas shared across datasets to ensure consistency in sample_id format, file path patterns, split values, and metadata structures.

#### Scenario: Reuse Common Field Definitions
**Given** multiple dataset schemas (TID2013, BID, AGIQA-3K)
**When** defining sample_id field
**Then** all schemas reference common_fields.json for sample_id definition
**And** sample_id follows pattern: "<dataset>_<id>" (e.g., "tid2013_0001")

---

### Requirement: Incremental Manifest Generation
**Priority**: P2
**Capability**: data-preparation

The system SHALL support incremental manifest generation to skip already-processed samples when re-running generators on partial datasets.

#### Scenario: Resume Manifest Generation
**Given** a partially generated manifest with 1000 out of 3000 samples
**When** generator runs with `--resume` flag
**Then** existing samples are loaded and sample_ids tracked
**And** only new samples are processed and appended
**And** duplicate sample_ids are prevented
