# summarizer-state-models Spec Delta

## MODIFIED Requirements

### Requirement: The system SHALL support flexible output formats in SummarizerOutput
The system SHALL support flexible output formats for both numerical scores and categorical answers

#### Scenario: Update final_answer to support Union type
- **WHEN** SummarizerOutput is created
- **THEN** final_answer field SHALL accept Union[str, float]:
  - str: For MCQ queries (e.g., "C") or explanatory answers
  - float: For scoring queries (e.g., 2.73)
- **AND** type validation SHALL accept both
- **AND** docstring SHALL document both use cases

#### Scenario: Validate final_answer based on type
- **WHEN** final_answer is str
- **THEN** it SHALL be non-empty and stripped of whitespace
- **AND** if empty, raise ValidationError

- **WHEN** final_answer is float
- **THEN** it SHALL be in range [1.0, 5.0]
- **AND** if out of range, raise ValidationError
- **AND** it SHALL be finite (not NaN or inf)

## ADDED Requirements

### Requirement: The system SHALL add quality_score field to SummarizerOutput
The system SHALL add explicit quality_score field to carry numerical quality scores

#### Scenario: Add quality_score field
- **WHEN** SummarizerOutput model is defined
- **THEN** it SHALL include:
```python
quality_score: Optional[float] = Field(
    None,
    ge=1.0,
    le=5.0,
    description="Continuous quality score (1-5) for IQA scoring queries. None for MCQ/explanation queries."
)
```
- **AND** quality_score SHALL be constrained to [1.0, 5.0] when not None
- **AND** quality_score SHALL be None for MCQ queries
- **AND** quality_score SHALL be populated for scoring queries

#### Scenario: Populate quality_score for scoring queries
- **WHEN** Query type is SCORING
- **THEN** Summarizer SHALL set quality_score = <fused_score>
- **AND** final_answer = <fused_score> (same value)
- **AND** both fields carry the same numerical value for clarity

#### Scenario: Leave quality_score as None for MCQ
- **WHEN** Query type is MCQ or EXPLANATION
- **THEN** Summarizer SHALL set quality_score = None
- **AND** final_answer = <letter or text>
- **AND** downstream code can check quality_score to distinguish types

### Requirement: The system SHALL update examples in config_schema_examples
The system SHALL update config_schema_examples to reflect new output format

#### Scenario: Add scoring query example
- **WHEN** config_schema_examples() is called
- **THEN** it SHALL include example:
```python
{
    "final_answer": 2.73,
    "quality_score": 2.73,
    "quality_reasoning": "The image shows moderate blur (severity: moderate, tool score: 2.6) affecting sharpness. Tool-augmented fusion yields score of 2.73, indicating fair quality.",
    "need_replan": False
}
```

#### Scenario: Update MCQ example to show quality_score=None
- **WHEN** config_schema_examples() is called
- **THEN** it SHALL include MCQ example:
```python
{
    "final_answer": "C",
    "quality_score": None,
    "quality_reasoning": "The image shows moderate blur affecting sharpness, consistent with tool score of 2.6.",
    "need_replan": False
}
```

### Requirement: The system SHALL validate quality_score consistency
The system SHALL validate consistency between final_answer and quality_score

#### Scenario: Validate scoring query output consistency
- **WHEN** quality_score is not None
- **THEN** final_answer SHOULD be a float (not strictly enforced for flexibility)
- **AND** if final_answer is float, it SHALL equal quality_score (within tolerance 0.01)
- **AND** if inconsistent, log WARNING but don't fail validation

#### Scenario: Validate MCQ output consistency
- **WHEN** quality_score is None
- **THEN** final_answer SHOULD be a str (not strictly enforced)
- **AND** no consistency check needed

## MODIFIED Requirements (Existing)

### Requirement: Field validators for final_answer
Field validators for final_answer are updated to handle Union[str, float] type

#### Scenario: Validate string final_answer
- **WHEN** final_answer is str
- **THEN** validator SHALL strip whitespace
- **AND** check it's not empty
- **AND** return stripped str

#### Scenario: Validate float final_answer
- **WHEN** final_answer is float
- **THEN** validator SHALL check 1.0 ≤ value ≤ 5.0
- **AND** check value is finite
- **AND** return value as-is

#### Scenario: Reject invalid types
- **WHEN** final_answer is neither str nor float
- **THEN** validator SHALL raise TypeError
- **AND** error message SHALL say "final_answer must be str or float"
