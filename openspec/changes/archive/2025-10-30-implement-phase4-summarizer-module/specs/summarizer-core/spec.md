# Spec: summarizer-core

## Purpose
Implement core Summarizer agent logic with evidence integration, two prompt modes, and replanning decision logic.

## ADDED Requirements

### Requirement: The system SHALL implement explanation/QA mode prompt template
The system SHALL implement explanation/QA mode prompt template

#### Scenario: Use explanation/QA prompt template
- **WHEN** Planner output has `query_type != "IQA"` (e.g., "Explanation", "MCQ")
- **THEN** Summarizer SHALL use the following prompt template:
```text
System:
You are a visual quality assessment assistant. Your task is to select the most appropriate answer to the user's
question. You are given:
- Distortion analysis (severity and visual impact of listed distortions)
- Tool response (overall quality scores from IQA models)
- Image content
Decision process
1. First, understand what kind of visual information is needed to answer the user's question.
2. Check if the provided distortion analysis or tool response already contains the required information.
3. If the provided information is sufficient, use it to answer.
4. If the information is unclear or insufficient, analyze the image directly to determine the best answer.
Return a valid JSON object in the following format:
{
  "final_answer": "<one of the above letters>",
  "quality_reasoning": "<brief explanation, based on either distortion analysis, tool response, or direct visual observation>"
}
```

#### Scenario: Format evidence for explanation/QA mode
- **WHEN** Rendering explanation/QA prompt
- **THEN** distortion_analysis SHALL be formatted as JSON
- **AND** quality_scores SHALL be formatted as JSON with tool names and scores
- **AND** user query and available answer choices SHALL be embedded
- **AND** images SHALL be provided to VLM

### Requirement: The system SHALL implement scoring mode prompt template
The system SHALL implement scoring mode prompt template

#### Scenario: Use scoring prompt template
- **WHEN** Planner output has `query_type == "IQA"`
- **THEN** Summarizer SHALL use the following prompt template:
```text
System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion
analysis). Your task is to assess the image quality.
You must select one single answer from the following:
A. Excellent
B. Good
C. Fair
D. Poor
E. Bad
Return the JSON:
{
  "final_answer": "<one letter>",
  "quality_reasoning": "<concise justification referencing distortions or tool scores>"
}
```

#### Scenario: Apply score fusion in scoring mode
- **WHEN** Scoring mode is active and tool scores are available
- **THEN** Summarizer SHALL invoke ScoreFusion utility
- **AND** use fusion result to guide VLM toward appropriate quality level
- **AND** include fusion score in quality_reasoning if helpful

### Requirement: The system SHALL integrate Executor evidence into prompts
The system SHALL integrate Executor evidence into prompts

#### Scenario: Format distortion_analysis for VLM
- **WHEN** ExecutorOutput contains distortion_analysis
- **THEN** it SHALL be formatted as JSON structure:
```json
{
  "<object or Global>": [
    {"type": "<distortion>", "severity": "<level>", "explanation": "<text>"}
  ]
}
```
- **AND** embedded in User message

#### Scenario: Format quality_scores for VLM
- **WHEN** ExecutorOutput contains quality_scores
- **THEN** it SHALL be formatted as JSON structure:
```json
{
  "<object or Global>": {
    "<distortion>": ["<tool_name>", <score>]
  }
}
```
- **AND** embedded in User message

#### Scenario: Handle missing Executor evidence
- **WHEN** ExecutorOutput is None or fields are empty
- **THEN** Summarizer SHALL rely on direct visual analysis
- **AND** omit evidence sections from prompt
- **AND** note lack of evidence in quality_reasoning

### Requirement: The system SHALL implement replanning decision logic
The system SHALL implement replanning decision logic

#### Scenario: Detect insufficient evidence
- **WHEN** Evaluating evidence sufficiency
- **THEN** Summarizer SHALL check:
  - distortion_analysis coverage for query_scope
  - quality_scores availability for key distortions
  - Consistency between analysis and scores
- **AND** set need_replan=true if critical gaps exist

#### Scenario: Provide replan_reason
- **WHEN** need_replan=true
- **THEN** Summarizer SHALL populate replan_reason with specific gap description
- **EXAMPLES**:
  - "Missing tool scores for vehicle region"
  - "Distortion analysis does not cover all query_scope objects"
  - "Contradictory evidence: severe blur but high scores"

#### Scenario: Respect max iterations
- **WHEN** Checking if replanning should be triggered
- **THEN** Summarizer SHALL check state["iteration_count"]
- **AND** if iteration_count >= max_replan_iterations, set need_replan=false
- **AND** log warning about max iterations reached

### Requirement: support VLM client integration
The system SHALL support VLM client integration with retry logic

#### Scenario: Load VLM client from config
- **WHEN** Summarizer node initializes
- **THEN** it SHALL load configuration from `configs/model_backends.yaml`:
```yaml
summarizer:
  backend: openai.gpt-4o
  temperature: 0.0
  max_tokens: 512
```
- **AND** create VLM client using vlm_client.create_vlm_client()

#### Scenario: Retry on invalid JSON
- **WHEN** VLM produces invalid JSON output
- **THEN** Summarizer SHALL retry up to 3 times
- **AND** add stricter instruction: "Return ONLY valid JSON"
- **AND** log each retry attempt

#### Scenario: Fallback on all retries failed
- **WHEN** All 3 retries produce invalid JSON
- **THEN** Summarizer SHALL return SummarizerOutput with:
  - final_answer="Unable to determine"
  - quality_reasoning="VLM output parsing failed"
  - need_replan=false (to prevent loop)
- **AND** log error with full VLM output

### Requirement: The system SHALL validate VLM output against expected...
The system SHALL validate VLM output against expected JSON schema

#### Scenario: Parse and validate VLM JSON response
- **WHEN** VLM returns JSON string
- **THEN** Summarizer SHALL parse it as JSON
- **AND** validate it contains "final_answer" and "quality_reasoning" fields
- **AND** create SummarizerOutput via Pydantic validation
- **AND** catch ValidationError and trigger retry if validation fails

#### Scenario: Extract final_answer
- **WHEN** VLM JSON is valid
- **THEN** final_answer SHALL be extracted and stripped of whitespace
- **AND** if final_answer is empty, trigger retry

#### Scenario: Extract quality_reasoning
- **WHEN** VLM JSON is valid
- **THEN** quality_reasoning SHALL be extracted and trimmed
- **AND** if quality_reasoning is empty or only whitespace, trigger retry

### Requirement: The system SHALL implement the summarizer_node function as...
The system SHALL implement the summarizer_node function as a LangGraph node

#### Scenario: Accept state and config
- **WHEN** summarizer_node is called
- **THEN** it SHALL accept:
  - state: AgenticIQAState
  - config: Optional[RunnableConfig]
- **AND** return Dict[str, Any] containing summarizer_result

#### Scenario: Check prerequisites
- **WHEN** summarizer_node starts
- **THEN** it SHALL check state contains "plan"
- **AND** check state contains "executor_evidence"
- **AND** return error if either is missing

#### Scenario: Select prompt mode
- **WHEN** Processing task
- **THEN** it SHALL select prompt template based on plan.query_type:
  - "IQA" → scoring mode
  - Other → explanation/QA mode

#### Scenario: Load images
- **WHEN** Summarizer needs to generate response
- **THEN** it SHALL load test image from state["image_path"]
- **AND** optionally load reference image from state["reference_path"]
- **AND** handle image loading errors gracefully

#### Scenario: Return state update
- **WHEN** Summarizer completes
- **THEN** it SHALL return:
```python
{
    "summarizer_result": SummarizerOutput(...),
    "iteration_count": state.get("iteration_count", 0) + (1 if need_replan else 0)
}
```

### Requirement: The system SHALL log comprehensive execution information
The system SHALL log comprehensive execution information

#### Scenario: Log Summarizer invocation
- **WHEN** summarizer_node starts
- **THEN** it SHALL log:
  - Query type and mode selected
  - Evidence summary (number of distortions, tool scores)
  - Image paths

#### Scenario: Log prompt and response
- **WHEN** Calling VLM
- **THEN** it SHALL log (at DEBUG level):
  - Full prompt text
  - VLM response (truncated if very long)
  - Retry attempts if needed

#### Scenario: Log replanning decision
- **WHEN** Replanning is triggered
- **THEN** it SHALL log:
  - need_replan=true
  - replan_reason
  - Current iteration count

### Requirement: The system SHALL handle edge cases gracefully
The system SHALL handle edge cases gracefully

#### Scenario: Empty Executor evidence
- **WHEN** executor_evidence is None or all fields are None/empty
- **THEN** Summarizer SHALL proceed with direct visual analysis
- **AND** mention lack of tool evidence in reasoning

#### Scenario: Contradictory evidence
- **WHEN** distortion_analysis shows "severe" but quality_scores are high (>4.0)
- **THEN** Summarizer SHALL note the contradiction in quality_reasoning
- **AND** consider setting need_replan=true

#### Scenario: Missing query_scope coverage
- **WHEN** Planner specifies query_scope=["vehicle", "background"]
- **AND** Executor only provides analysis for "vehicle"
- **THEN** Summarizer SHALL detect missing "background" evidence
- **AND** set need_replan=true with appropriate reason
