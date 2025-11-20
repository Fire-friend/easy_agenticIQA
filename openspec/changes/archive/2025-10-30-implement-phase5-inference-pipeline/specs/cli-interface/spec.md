## ADDED Requirements

### Requirement: Configuration File Loading
**Priority**: P0
**Capability**: cli-interface

The system SHALL load configuration from a YAML file specified via `--config` argument, with defaults from `configs/pipeline.yaml`.

#### Scenario: Load Custom Config
**Given** a custom config at `configs/custom.yaml`
**When** the user runs `run_pipeline.py --config configs/custom.yaml`
**Then** settings are loaded from the custom file
**And** missing values fall back to defaults

---

### Requirement: Input/Output Paths
**Priority**: P0
**Capability**: cli-interface

The system SHALL accept input JSONL path via `--input` and output JSONL path via `--output` as required arguments.

#### Scenario: Specify I/O Paths
**Given** input at `data/processed/test.jsonl`
**When** the user runs with `--input data/processed/test.jsonl --output results/test_output.jsonl`
**Then** samples are read from input path
**And** results are written to output path

---

### Requirement: Configuration Override
**Priority**: P1
**Capability**: cli-interface

The system SHALL support runtime configuration override via `--backend-override KEY=VALUE` arguments using dot notation for nested keys.

#### Scenario: Override Planner Backend
**Given** config file specifies planner.backend=openai.gpt-4o
**When** the user runs with `--backend-override planner.backend=openai.gpt-4o-mini`
**Then** the planner uses GPT-4o-mini instead of GPT-4o

#### Scenario: Override Multiple Settings
**Given** default configuration
**When** the user runs with `--backend-override planner.temperature=0.5 --backend-override executor.backend=qwen2.5-vl`
**Then** both overrides are applied
**And** other settings remain unchanged

---

### Requirement: Resume Flag
**Priority**: P0
**Capability**: cli-interface

The system SHALL support a `--resume` flag to enable resuming from existing output file.

#### Scenario: Enable Resume Mode
**Given** an existing output file with 50 processed samples
**When** the user runs with `--resume`
**Then** those 50 samples are skipped
**And** only new samples are processed

---

### Requirement: Sample Limiting
**Priority**: P1
**Capability**: cli-interface

The system SHALL support `--max-samples N` to limit processing to the first N samples for testing.

#### Scenario: Process First 10 Samples
**Given** an input file with 1000 samples
**When** the user runs with `--max-samples 10`
**Then** only the first 10 samples are processed
**And** execution stops after sample 10

---

### Requirement: Verbose Logging
**Priority**: P1
**Capability**: cli-interface

The system SHALL support a `--verbose` flag to enable DEBUG-level logging for troubleshooting.

#### Scenario: Enable Verbose Mode
**Given** default logging level is INFO
**When** the user runs with `--verbose`
**Then** logging level is set to DEBUG
**And** prompts and responses are included in logs

---

### Requirement: Dry Run Mode
**Priority**: P2
**Capability**: cli-interface

The system SHALL support a `--dry-run` flag to validate configuration and data files without executing the pipeline.

#### Scenario: Validate Configuration
**Given** a configuration file and input JSONL
**When** the user runs with `--dry-run`
**Then** the config is loaded and validated
**And** the input file is parsed and validated
**And** estimated execution plan is printed (sample count, cost estimate)
**And** no pipeline execution occurs
