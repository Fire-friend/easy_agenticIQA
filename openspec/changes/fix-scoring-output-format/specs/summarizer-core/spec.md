# summarizer-core Spec Delta

## MODIFIED Requirements

### Requirement: The system SHALL implement scoring mode prompt template
The system SHALL implement scoring mode prompt template to request probability distributions for numerical scoring

#### Scenario: Use probability distribution prompt for scoring queries
- **WHEN** Planner output has `query_type == "IQA"` AND query is a pure scoring request (e.g., "Rate the quality")
- **THEN** Summarizer SHALL use the following prompt template:
```text
System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion
analysis), assess the image quality and provide your confidence for each quality level.

Tool scores (1-5 scale, higher is better): {tool_scores}
Tool score mean: {tool_mean:.2f}

Distortion analysis: {distortion_analysis}

Your task:
1. Analyze the image quality based on the evidence
2. For EACH quality level, provide your log-probability (confidence):
   - Level 5 (Excellent): no visible distortions
   - Level 4 (Good): minor distortions, minimal impact
   - Level 3 (Fair): moderate distortions, noticeable impact
   - Level 2 (Poor): severe distortions, significant impact
   - Level 1 (Bad): extreme distortions, unusable quality

Return valid JSON:
{
  "quality_probs": {
    "1": <log_prob_1>,
    "2": <log_prob_2>,
    "3": <log_prob_3>,
    "4": <log_prob_4>,
    "5": <log_prob_5>
  },
  "quality_reasoning": "<concise justification referencing distortions and tool scores>"
}

The image: <image>
```

#### Scenario: Apply score fusion in scoring mode
- **WHEN** Scoring mode is active and tool scores are available
- **THEN** Summarizer SHALL invoke ScoreFusion utility
- **AND** extract VLM probability distribution from quality_probs
- **AND** compute fused score: q = Σ_c α_c · p_c · c
- **AND** return numerical score (1-5 continuous scale) as final_answer
- **AND** populate quality_score field with the same numerical value
- **AND** include fusion details in quality_reasoning

#### Scenario: Fallback when VLM doesn't provide quality_probs
- **WHEN** VLM response doesn't contain quality_probs dict
- **THEN** Summarizer SHALL retry with stricter prompt format
- **AND** if retry fails, extract quality level from reasoning text
- **AND** apply one-hot probability distribution (smoothed with epsilon=0.15)
- **AND** log warning about missing probabilities

#### Scenario: Handle MCQ queries with categorical options
- **WHEN** Query contains explicit options (e.g., "A) Excellent B) Good C) Fair")
- **THEN** Summarizer SHALL use categorical prompt template (existing EXPLANATION_PROMPT_TEMPLATE)
- **AND** return letter answer in final_answer
- **AND** set quality_score=None

### Requirement: The system SHALL detect query type to determine output format
The system SHALL detect query type to distinguish scoring queries from MCQ queries

#### Scenario: Detect pure scoring query
- **WHEN** Query contains "rate|score|assess|evaluate" + "quality" AND no explicit options
- **THEN** detect_query_type() SHALL return QueryType.SCORING
- **AND** Summarizer SHALL return numerical score
- **EXAMPLES**:
  - "Rate the perceptual quality of this image" → SCORING
  - "What is the quality score?" → SCORING
  - "Assess the image quality" → SCORING

#### Scenario: Detect MCQ query
- **WHEN** Query contains patterns like "A)" or "choose from" or explicit answer options
- **THEN** detect_query_type() SHALL return QueryType.MCQ
- **AND** Summarizer SHALL return letter answer
- **EXAMPLES**:
  - "Is quality: A) Excellent B) Good C) Fair?" → MCQ
  - "Choose from: A) High quality B) Low quality" → MCQ

#### Scenario: Detect explanation query
- **WHEN** Query contains "why|explain|describe|what"
- **THEN** detect_query_type() SHALL return QueryType.EXPLANATION
- **AND** Summarizer SHALL return descriptive text

### Requirement: The system SHALL return numerical scores for scoring queries
The system SHALL return numerical scores (1-5 continuous scale) for scoring queries instead of categorical ratings

#### Scenario: Return continuous numerical score
- **WHEN** Query type is SCORING
- **THEN** SummarizerOutput.final_answer SHALL be a float (e.g., 2.73)
- **AND** SummarizerOutput.quality_score SHALL be the same float value
- **AND** score SHALL be in range [1.0, 5.0]
- **AND** quality_reasoning SHALL reference tool scores and VLM assessment

#### Scenario: Return categorical letter for MCQ
- **WHEN** Query type is MCQ
- **THEN** SummarizerOutput.final_answer SHALL be a str (e.g., "C")
- **AND** SummarizerOutput.quality_score SHALL be None
- **AND** maintain backward compatibility with MCQ evaluation

### Requirement: The system SHALL extract VLM probability distributions
The system SHALL extract VLM probability distributions for quality levels from model output

#### Scenario: Extract probabilities from quality_probs dict
- **WHEN** VLM returns quality_probs: {"1": -3.2, "2": -0.5, "3": -0.1, "4": -2.1, "5": -4.5}
- **THEN** Summarizer SHALL apply softmax: p_c = exp(log p̂_c) / Σ exp(log p̂_j)
- **AND** verify Σ p_c ≈ 1.0
- **AND** pass probabilities to ScoreFusion.fuse()

#### Scenario: Fallback to classification extraction
- **WHEN** VLM doesn't provide quality_probs but reasoning suggests level (e.g., "Fair quality")
- **THEN** Summarizer SHALL extract level from text
- **AND** create one-hot distribution with smoothing: p_level=0.7, others=0.075 each
- **AND** log warning about approximate fusion

#### Scenario: Fallback to uniform distribution
- **WHEN** No quality_probs and no extractable level
- **THEN** Summarizer SHALL use uniform probabilities: p_c = 0.2 for all c
- **AND** log warning that fusion relies only on tool scores

## REMOVED Requirements

### Requirement: OLD scoring mode prompt with categorical selection
The old SCORING_PROMPT_TEMPLATE that forced selection from "A. Excellent, B. Good, C. Fair, D. Poor, E. Bad" is removed and replaced with probability distribution request.
