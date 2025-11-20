# Spec: score-fusion

## Purpose
Implement weighted score fusion algorithm that combines IQA tool scores with VLM probability distributions for scoring tasks.

## ADDED Requirements

### Requirement: implement perceptual weight calculation
The system SHALL implement perceptual weight calculation using Gaussian distribution centered at tool score mean

#### Scenario: Compute perceptual weights from tool scores
- **WHEN** Score fusion receives tool scores `[q̂_1, q̂_2, ..., q̂_n]`
- **THEN** it SHALL compute mean: `q̄ = (1/n) Σ q̂_i`
- **AND** compute Gaussian weights for each quality level c:
```
α_c = exp(-η(q̄ - c)²) / Σ_j∈C exp(-η(q̄ - j)²)
```
- **WHERE** η=1 (default), C={1,2,3,4,5}
- **AND** Σ α_c = 1 (weights sum to 1)

#### Scenario: Handle empty tool scores
- **WHEN** Tool scores list is empty
- **THEN** it SHALL return uniform weights: α_c = 0.2 for all c ∈ {1,2,3,4,5}
- **AND** log a warning about missing tool scores

#### Scenario: Handle single tool score
- **WHEN** Only one tool score is available
- **THEN** it SHALL use that score as q̄
- **AND** compute Gaussian weights normally

### Requirement: extract or estimate VLM probability distributions
The system SHALL extract or estimate VLM probability distributions for quality levels

#### Scenario: Extract probabilities from VLM logits
- **WHEN** VLM API returns logits for quality levels [log p̂_1, log p̂_2, ..., log p̂_5]
- **THEN** it SHALL apply softmax: `p_c = exp(log p̂_c) / Σ_j exp(log p̂_j)`
- **AND** validate Σ p_c ≈ 1

#### Scenario: Extract probabilities from classification output
- **WHEN** VLM returns single classification (e.g., "C" or "Fair")
- **THEN** it SHALL assign high probability to predicted class (default 0.7)
- **AND** distribute remaining probability uniformly among other classes
- **AND** log a warning that fusion is approximate without logits

#### Scenario: Fallback to uniform distribution
- **WHEN** VLM output doesn't contain probabilities or classification
- **THEN** it SHALL use uniform distribution: p_c = 0.2 for all c
- **AND** log a warning and disable fusion (use tool scores only)

### Requirement: apply fusion
The system SHALL apply fusion formula to compute final quality score

#### Scenario: Fuse tool scores with VLM probabilities
- **WHEN** Perceptual weights {α_c} and VLM probabilities {p_c} are available
- **THEN** it SHALL compute final score:
```
q = Σ_c∈C α_c · p_c · c
```
- **WHERE** c ∈ {1,2,3,4,5}
- **AND** q SHALL be in range [1, 5]

#### Scenario: Validate fusion output range
- **WHEN** Fusion formula produces score q
- **THEN** q SHALL satisfy 1 ≤ q ≤ 5
- **AND** if q < 1 or q > 5, it SHALL be clipped to [1, 5]
- **AND** log a warning if clipping occurs

#### Scenario: Handle numerical stability
- **WHEN** Computing exp() in softmax or Gaussian weights
- **THEN** it SHALL use numerically stable implementation (subtract max before exp)
- **AND** handle overflow/underflow gracefully

### Requirement: The system SHALL map continuous fusion scores to...
The system SHALL map continuous fusion scores to discrete quality levels

#### Scenario: Map to 5-level scale (1-5)
- **WHEN** Fusion produces continuous score q
- **THEN** it SHALL map to discrete level using rounding: level = round(q)
- **AND** ensure level ∈ {1, 2, 3, 4, 5}

#### Scenario: Map to letter grades (A-E)
- **WHEN** Fusion produces discrete level l ∈ {1,2,3,4,5}
- **THEN** it SHALL map to letters:
  - 5 → "A" (Excellent)
  - 4 → "B" (Good)
  - 3 → "C" (Fair)
  - 2 → "D" (Poor)
  - 1 → "E" (Bad)

#### Scenario: Support custom thresholds
- **WHEN** Custom level thresholds are provided
- **THEN** it SHALL use thresholds for mapping instead of rounding
- **EXAMPLE**: thresholds=[1.5, 2.5, 3.5, 4.5] → q=2.7 maps to level 3

### Requirement: The system SHALL provide configurable fusion parameters
The system SHALL provide configurable fusion parameters

#### Scenario: Configure eta parameter
- **WHEN** ScoreFusion is initialized
- **THEN** it SHALL accept eta parameter (default=1.0)
- **AND** higher eta SHALL make Gaussian weights more concentrated around tool mean
- **AND** lower eta SHALL make weights more uniform

#### Scenario: Configure quality levels
- **WHEN** ScoreFusion is initialized
- **THEN** it SHALL accept quality_levels parameter (default=[1,2,3,4,5])
- **AND** support alternative scales (e.g., [1,2,3,4,5,6,7] for 7-level scale)

#### Scenario: Configure VLM probability extraction mode
- **WHEN** ScoreFusion is initialized
- **THEN** it SHALL accept mode parameter: "logits", "classification", or "uniform"
- **AND** mode SHALL determine how VLM probabilities are extracted

### Requirement: provide comprehensive logging
The system SHALL provide comprehensive logging for fusion process

#### Scenario: Log fusion inputs
- **WHEN** Fusion is performed
- **THEN** it SHALL log:
  - Tool scores: [q̂_1, q̂_2, ..., q̂_n]
  - Tool score mean: q̄
  - Perceptual weights: {α_c}
  - VLM probabilities: {p_c}

#### Scenario: Log fusion output
- **WHEN** Fusion completes
- **THEN** it SHALL log:
  - Continuous score: q
  - Discrete level: l
  - Mapped letter grade (if applicable)

#### Scenario: Log warnings for edge cases
- **WHEN** Edge cases occur
- **THEN** it SHALL log warnings for:
  - Empty tool scores
  - Missing VLM probabilities
  - Clipped fusion scores
  - Numerical instability

### Requirement: The system SHALL validate fusion algorithm against known...
The system SHALL validate fusion algorithm against known test cases

#### Scenario: Test case 1 - Single tool score with uniform VLM
- **GIVEN** tool_scores=[2.6], vlm_probs={1:0.2, 2:0.2, 3:0.2, 4:0.2, 5:0.2}
- **WHEN** Fusion is computed
- **THEN** α weights SHALL peak at level 3 (closest to 2.6)
- **AND** final score q SHALL be close to 2.6 (dominated by tool score)

#### Scenario: Test case 2 - Multiple tool scores with VLM classification
- **GIVEN** tool_scores=[2.5, 2.7, 2.4], vlm_class="C" (level 3, prob=0.7)
- **WHEN** Fusion is computed
- **THEN** q̄ = 2.53
- **AND** α weights SHALL peak around level 3
- **AND** final score SHALL be between 2.5 and 3.0

#### Scenario: Test case 3 - Tool score and VLM disagree
- **GIVEN** tool_scores=[2.0], vlm_probs={1:0.1, 2:0.1, 3:0.1, 4:0.1, 5:0.6}
- **WHEN** Fusion is computed
- **THEN** α weights SHALL favor level 2 (tool mean)
- **AND** VLM probs favor level 5
- **AND** final score SHALL be weighted average leaning toward tool score
