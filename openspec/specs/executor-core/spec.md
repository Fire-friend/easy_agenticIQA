# executor-core Specification

## Purpose
TBD - created by archiving change implement-phase3-executor-module. Update Purpose after archive.
## Requirements
### Requirement: Distortion Detection Subtask
The system SHALL implement a distortion detection subtask that identifies potential distortion types from images when `distortion_source="Inferred"`.

#### Scenario: Detect distortions with VLM
- **WHEN** Executor receives a plan with `distortion_detection=true` and `distortion_source="Inferred"`
- **THEN** it SHALL use the following prompt template:
```text
System:
You are an expert in distortion detection. Based on the user's query, identify all possible distortions need to be
focused on to properly address the user's intent.
Return a valid JSON object in the following format:
{
  "distortion_set": {
    "<object or Global>": ["<distortion_1>", "<distortion_2>", ...]
  }
}
Instructions:
1. Focus your analysis on query scope. Describe distortions for each individually.
2. Only include distortion types from the following valid categories: ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"]

User:
User's query: {query}
The image: <image>
```

#### Scenario: Validate detected distortions
- **WHEN** Distortion detection subtask produces output
- **THEN** it SHALL validate that all distortion types belong to the valid categories list
- **AND** distortion object names SHALL align with Planner's `query_scope`
- **AND** it SHALL fallback to "Global" if object name is missing

#### Scenario: Handle detection errors
- **WHEN** VLM produces invalid JSON in distortion detection
- **THEN** it SHALL retry up to 3 times with stricter prompt ("Return ONLY valid JSON")
- **AND** it SHALL log the error if all retries fail
- **AND** it SHALL return null for distortion_set field

### Requirement: Distortion Analysis Subtask
The system SHALL implement a distortion analysis subtask that estimates distortion severity and visual impact.

#### Scenario: Analyze distortion severity
- **WHEN** Executor receives a plan with `distortion_analysis=true`
- **THEN** it SHALL use the following prompt template:
```text
System:
You are a distortion analysis expert. Your task is to assess the severity and visual impact of various distortion
types for different regions of an image or the entire image.
The distortion information: {distortion_set}
Return a valid JSON object in the following format:
{
  "distortion_analysis": {
    "<object or Global>": [
      {
        "type": "<distortion>",
        "severity": "<none/slight/moderate/severe/extreme>",
        "explanation": "<brief visual explanation>"
      }
    ]
  }
}
Instructions:
1. Base your analysis on the listed distortion types and consider the user question.
2. Use "none" if a distortion is barely or not visible.
3. Keep explanations short and focused on visual quality. Focus solely on analyzing visual distortion effects.

User:
User's query: {query}
The image: <image>
```

#### Scenario: Validate severity levels
- **WHEN** Distortion analysis subtask produces output
- **THEN** severity SHALL be one of: none, slight, moderate, severe, extreme
- **AND** explanation SHALL be non-empty string
- **AND** type SHALL match a distortion from distortion_set

#### Scenario: Handle analysis errors
- **WHEN** VLM produces invalid JSON in distortion analysis
- **THEN** it SHALL retry up to 3 times with stricter prompt
- **AND** it SHALL log the error if all retries fail
- **AND** it SHALL return null for distortion_analysis field

### Requirement: Tool Selection Subtask
The system SHALL implement a tool selection subtask that assigns appropriate IQA tools based on distortion types and tool capabilities.

#### Scenario: Select tools with VLM
- **WHEN** Executor receives a plan with `tool_selection=true`
- **THEN** it SHALL use the following prompt template:
```text
System:
You are a tool executor. Your task is to assign the most appropriate IQA tool to each visual distortion type,
based on the descriptions of the tools.
The distortion information: {distortion_set}.
The available tools: {tool_description}.
Return a valid JSON object in the following format:
{
  "selected_tools": {
    "<object or Global>": {
      "<distortion>": "<tool_name>"
    }
  }
}
Instructions:
For each distortion, choose the tool whose description suggests it performs best for that type of distortion.
```

#### Scenario: Prioritize FR tools when reference available
- **WHEN** Planner output has `reference_mode="Full-Reference"`
- **AND** Tool selection subtask is selecting tools
- **THEN** it SHALL prioritize Full-Reference (FR) tools over No-Reference (NR) tools
- **AND** it SHALL only select NR tools if no suitable FR tool exists

#### Scenario: Skip tool selection when required_tool specified
- **WHEN** Planner output has `required_tool` field set
- **THEN** Executor SHALL skip tool selection subtask
- **AND** use the required_tool for all distortions

#### Scenario: Handle selection errors
- **WHEN** VLM produces invalid JSON in tool selection
- **THEN** it SHALL retry up to 3 times
- **AND** it SHALL fallback to generic tools (QAlign for NR, TOPIQ_FR for FR) if all retries fail

### Requirement: Tool Execution Subtask
The system SHALL implement a tool execution subtask that runs selected IQA tools and normalizes scores.

#### Scenario: Execute tools and normalize scores
- **WHEN** Executor receives a plan with `tool_execution=true`
- **THEN** it SHALL iterate through selected_tools
- **AND** execute each tool via ToolRegistry
- **AND** normalize raw scores to [1, 5] range using logistic function
- **AND** record tool execution logs with tool name, raw score, normalized score

#### Scenario: Handle tool failures
- **WHEN** An IQA tool execution fails or returns NaN/Inf
- **THEN** it SHALL log the failure with error details
- **AND** fallback to generic NR tools (BRISQUE or NIQE)
- **AND** mark the fallback in tool_logs with `fallback=true` flag
- **AND** continue executing remaining tools

#### Scenario: Cache tool outputs
- **WHEN** Tool execution subtask runs the same tool on the same image
- **THEN** it SHALL check cache first using key: hash(image) + tool_name + hash(reference)
- **AND** return cached result if available
- **AND** store result in cache after execution if not cached

### Requirement: Executor Node Orchestration
The system SHALL implement an executor_node function that orchestrates all four subtasks based on Planner control flags.

#### Scenario: Conditional subtask execution
- **WHEN** executor_node receives AgenticIQAState with plan
- **THEN** it SHALL check plan.plan.distortion_detection flag
- **AND** execute distortion_detection_subtask if true
- **AND** check plan.plan.distortion_analysis flag
- **AND** execute distortion_analysis_subtask if true
- **AND** check plan.plan.tool_selection flag
- **AND** execute tool_selection_subtask if true
- **AND** check plan.plan.tool_execution flag
- **AND** execute tool_execution_subtask if true
- **AND** return ExecutorOutput with all subtask results

#### Scenario: Load executor configuration
- **WHEN** executor_node initializes
- **THEN** it SHALL load VLM configuration from configs/model_backends.yaml executor section
- **AND** create VLM client using configuration
- **AND** support custom API endpoints via base_url

#### Scenario: Handle subtask failures gracefully
- **WHEN** A subtask fails with exception
- **THEN** executor_node SHALL log the error
- **AND** set the corresponding output field to null
- **AND** continue executing remaining subtasks
- **AND** NOT fail the entire executor_node

### Requirement: Data Flow Between Subtasks
The system SHALL ensure proper data flow between subtasks to support conditional dependencies.

#### Scenario: Pass distortion_set to analysis
- **WHEN** distortion_detection produces distortion_set
- **THEN** distortion_analysis SHALL receive it in the prompt template
- **AND** use it to guide severity assessment

#### Scenario: Pass distortion_set and tool_metadata to selection
- **WHEN** tool_selection executes
- **THEN** it SHALL receive distortion_set (from detection or Planner)
- **AND** receive tool metadata from ToolRegistry
- **AND** format both into the prompt template

#### Scenario: Pass selected_tools to execution
- **WHEN** tool_execution executes
- **THEN** it SHALL receive selected_tools mapping
- **AND** iterate through each object → distortion → tool_name
- **AND** execute tools with image and optional reference

