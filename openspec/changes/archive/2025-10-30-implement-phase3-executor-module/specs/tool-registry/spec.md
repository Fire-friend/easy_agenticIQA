# Capability: Tool Registry

## ADDED Requirements

### Requirement: Tool Metadata Management
The system SHALL provide a tool registry that loads and manages IQA tool metadata from a JSON file.

#### Scenario: Load tool metadata from JSON
- **WHEN** ToolRegistry initializes
- **THEN** it SHALL read tool metadata from `iqa_tools/metadata/tools.json`
- **AND** parse JSON into structured tool definitions
- **AND** raise clear error if file is missing or invalid JSON

#### Scenario: Tool metadata structure
- **WHEN** Tool metadata is loaded
- **THEN** each tool SHALL have:
  - `type`: "FR" (Full-Reference) or "NR" (No-Reference)
  - `strengths`: List of distortion types the tool excels at
  - `logistic_params`: Dictionary with beta1, beta2, beta3, beta4 parameters (optional)
- **AND** tool names SHALL match IQA-PyTorch tool identifiers

#### Scenario: Validate tool metadata
- **WHEN** ToolRegistry loads metadata
- **THEN** it SHALL validate that type is either "FR" or "NR"
- **AND** validate that strengths contains only valid distortion categories
- **AND** warn if logistic_params are missing (will use defaults)

### Requirement: Tool Execution Interface
The system SHALL provide a unified interface for executing IQA tools from IQA-PyTorch library.

#### Scenario: Execute Full-Reference tool
- **WHEN** execute_tool is called with FR tool and reference image
- **THEN** it SHALL load both test and reference images
- **AND** preprocess images to match tool requirements (size, color space)
- **AND** call IQA-PyTorch tool with both images
- **AND** return raw score as float

#### Scenario: Execute No-Reference tool
- **WHEN** execute_tool is called with NR tool and no reference
- **THEN** it SHALL load test image only
- **AND** preprocess image to match tool requirements
- **AND** call IQA-PyTorch tool with test image
- **AND** return raw score as float

#### Scenario: Handle image preprocessing
- **WHEN** execute_tool loads images for a tool
- **THEN** it SHALL convert images to RGB if needed
- **AND** resize images if tool has size constraints
- **AND** convert to numpy array or torch tensor as required by tool
- **AND** normalize pixel values to [0, 1] or [0, 255] based on tool

#### Scenario: Tool execution error handling
- **WHEN** Tool execution fails with exception
- **THEN** it SHALL catch the exception
- **AND** log error with tool name, image path, and exception message
- **AND** raise ToolExecutionError with original exception details

### Requirement: Score Normalization
The system SHALL normalize heterogeneous tool outputs to a unified [1, 5] scale using logistic transformation.

#### Scenario: Apply five-parameter logistic function
- **WHEN** normalize_score is called with tool_name and raw_score
- **THEN** it SHALL retrieve logistic parameters for the tool from metadata
- **AND** apply formula: f(x) = (β1 - β2) / (1 + exp(-(x - β3)/|β4|)) + β2
- **AND** return normalized score in [1, 5] range

#### Scenario: Handle missing logistic parameters
- **WHEN** Tool metadata lacks logistic_params
- **THEN** normalize_score SHALL use empirical defaults:
  - For "higher is better" tools: beta1=5.0, beta2=1.0, beta3=0.5, beta4=0.1
  - For "lower is better" tools: beta1=1.0, beta2=5.0, beta3=0.5, beta4=0.1
- **AND** log warning about using defaults

#### Scenario: Validate normalized scores
- **WHEN** normalize_score produces output
- **THEN** normalized score SHALL be within [1, 5] range
- **AND** if outside range, SHALL clip to [1, 5]
- **AND** log warning if clipping occurred

#### Scenario: Handle inverted tool outputs
- **WHEN** Tool outputs "lower is better" metric (e.g., distortion/error)
- **THEN** normalize_score SHALL detect tool direction from metadata
- **AND** invert the logistic function parameters (swap beta1 and beta2)
- **AND** produce final score where higher means better quality

### Requirement: Tool Output Caching
The system SHALL cache tool execution results to avoid redundant computations.

#### Scenario: Generate cache key
- **WHEN** Tool is about to execute
- **THEN** it SHALL compute cache key as: hash(image_bytes) + tool_name + hash(reference_bytes)
- **AND** hash SHALL be SHA256 of image binary content
- **AND** reference_bytes SHALL be empty string if no reference image

#### Scenario: Check cache before execution
- **WHEN** execute_tool is called
- **THEN** it SHALL check in-memory cache with cache key
- **AND** return cached result immediately if found
- **AND** skip tool execution if cache hit

#### Scenario: Store results in cache
- **WHEN** Tool execution completes successfully
- **THEN** it SHALL store result in cache with cache key
- **AND** cache SHALL persist for duration of program execution
- **AND** cache SHALL store both raw_score and normalized_score

#### Scenario: Cache size management
- **WHEN** Cache grows large (>1000 entries)
- **THEN** it SHALL evict oldest entries using LRU policy
- **AND** log cache eviction events
- **AND** maintain cache hit rate statistics

### Requirement: Tool Capability Querying
The system SHALL provide methods to query tool capabilities for tool selection logic.

#### Scenario: Get tools by distortion type
- **WHEN** get_tools_for_distortion is called with distortion type
- **THEN** it SHALL return list of tool names whose strengths include that distortion
- **AND** sort tools by type (FR tools first if reference available)

#### Scenario: Get tools by reference mode
- **WHEN** get_tools_by_type is called with "FR" or "NR"
- **THEN** it SHALL return list of tool names matching that type
- **AND** include all tools of that type regardless of strengths

#### Scenario: Check tool availability
- **WHEN** is_tool_available is called with tool_name
- **THEN** it SHALL check if tool is in loaded metadata
- **AND** optionally verify IQA-PyTorch can import the tool
- **AND** return boolean availability status

### Requirement: Tool Metadata Format
The system SHALL define a JSON schema for tool metadata files.

#### Scenario: JSON structure requirements
- **WHEN** Tool metadata file is created or validated
- **THEN** it SHALL be valid JSON object
- **AND** top-level keys SHALL be tool names (e.g., "TOPIQ_FR", "QAlign")
- **AND** each tool SHALL have required fields: type, strengths
- **AND** each tool MAY have optional field: logistic_params
- **AND** example structure:
```json
{
  "TOPIQ_FR": {
    "type": "FR",
    "strengths": ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"],
    "logistic_params": {"beta1": 5.0, "beta2": 1.0, "beta3": 0.5, "beta4": 0.1}
  },
  "QAlign": {
    "type": "NR",
    "strengths": ["Blurs", "Color distortions", "Noise", "Brightness change", "Spatial distortions", "Sharpness"],
    "logistic_params": {"beta1": 5.0, "beta2": 1.0, "beta3": 3.0, "beta4": 0.5}
  }
}
```
