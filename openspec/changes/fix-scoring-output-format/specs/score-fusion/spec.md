# score-fusion Spec Delta

## MODIFIED Requirements

### Requirement: The system SHALL fully implement fusion formula from paper
The system SHALL fully implement the paper's fusion formula: q = Σ_c α_c · p_c · c

#### Scenario: Apply complete fusion pipeline
- **WHEN** ScoreFusion.fuse(tool_scores, vlm_probs) is called
- **THEN** it SHALL:
  1. Compute perceptual weights: α_c = exp(-η(q̄-c)²) / Σ_j exp(-η(q̄-j)²)
  2. Validate VLM probabilities sum to 1.0 (within tolerance 0.01)
  3. Apply fusion formula: q = Σ_{c∈{1,2,3,4,5}} α_c · p_c · c
  4. Clip result to [1.0, 5.0]
  5. Return continuous score as float
- **AND** log all intermediate values (q̄, α, p, q)

#### Scenario: Compute perceptual weights with numerical stability
- **WHEN** Computing Gaussian weights α_c = exp(-η(q̄-c)²) / Σ exp(-η(q̄-j)²)
- **THEN** ScoreFusion SHALL use numerically stable implementation:
  - Compute exponents: exp_values = [exp(-η(q̄-c)²) for c in [1,2,3,4,5]]
  - Find max to avoid overflow: max_exp = max(exp_values)
  - Subtract max before exp: stable_exp = [exp(e - max_exp) for e in exp_values]
  - Normalize: α_c = stable_exp[c] / sum(stable_exp)
- **AND** verify Σ α_c = 1.0 (within tolerance 0.001)

#### Scenario: Validate fusion output range
- **WHEN** Fusion produces score q
- **THEN** ScoreFusion SHALL verify 1.0 ≤ q ≤ 5.0
- **AND** if q < 1.0 or q > 5.0, clip to [1.0, 5.0]
- **AND** log ERROR with details: "Fusion score {q:.3f} out of range, clipped to {clipped:.3f}"

### Requirement: The system SHALL implement VLM probability extraction
The system SHALL implement VLM probability extraction from model-provided log-probabilities

#### Scenario: Extract probabilities from quality_probs dict (primary method)
- **WHEN** VLM output contains quality_probs: {"1": log_p1, "2": log_p2, ...}
- **THEN** ScoreFusion.extract_vlm_probs() SHALL:
  - Parse log-probabilities for levels 1-5 from VLM JSON output
  - Apply softmax transformation: p_c = exp(log p̂_c) / Σ_j exp(log p̂_j)
  - Use numerically stable softmax (subtract max before exp)
  - Validate all log_probs are finite (not NaN or inf)
  - Return normalized probability dict: {1: p1, 2: p2, 3: p3, 4: p4, 5: p5}
- **AND** verify Σ p_c ≈ 1.0 (within tolerance 0.01)
- **NOTE**: These are VLM's self-assessed confidences per the paper's methodology

#### Scenario: Fallback to classification with smoothing
- **WHEN** VLM output doesn't have quality_probs but has quality_level (e.g., 3)
- **THEN** ScoreFusion SHALL create smoothed one-hot distribution:
  - Assign high probability to predicted level: p_level = 0.7
  - Distribute remaining uniformly: p_other = 0.075 each (0.3 / 4)
  - Ensures Σ p_c = 1.0
- **AND** log WARNING: "Using classification fallback with smoothing"

#### Scenario: Fallback to uniform distribution
- **WHEN** VLM output has neither quality_probs nor quality_level
- **THEN** ScoreFusion SHALL use uniform distribution:
  - p_c = 0.2 for all c ∈ {1, 2, 3, 4, 5}
- **AND** log WARNING: "No VLM probabilities, using uniform distribution"
- **AND** fusion relies entirely on tool score mean

### Requirement: The system SHALL provide detailed logging for debugging
The system SHALL provide detailed logging for score fusion debugging

#### Scenario: Log fusion inputs
- **WHEN** ScoreFusion.fuse() is called
- **THEN** it SHALL log at INFO level:
  - Tool scores: [q̂_1, q̂_2, ..., q̂_n]
  - Tool score count: n
  - Tool score mean: q̄ (2 decimal places)
  - VLM probabilities: {1: p1, 2: p2, 3: p3, 4: p4, 5: p5} (3 decimal places)

#### Scenario: Log perceptual weights
- **WHEN** Computing α weights
- **THEN** it SHALL log at INFO level:
  - Perceptual weights: {1: α1, 2: α2, 3: α3, 4: α4, 5: α5} (3 decimal places)
  - Peak weight level: argmax(α_c)
  - Weight entropy: H(α) = -Σ α_c log α_c (indicates concentration)

#### Scenario: Log fusion output
- **WHEN** Fusion completes
- **THEN** it SHALL log at INFO level:
  - Final score: q (2 decimal places)
  - Discrete level: round(q)
  - Contribution breakdown: Σ α_c · p_c · c for each c

#### Scenario: Log warnings for edge cases
- **WHEN** Edge cases occur
- **THEN** it SHALL log at WARNING level:
  - Empty tool scores → "No tool scores, using uniform α weights"
  - Missing VLM probs → "No VLM probabilities, using fallback"
  - Clipped score → "Score {q} clipped to [{min}, {max}]"
  - Numerical issues → "Numerical instability detected, applied stabilization"

## ADDED Requirements

### Requirement: The system SHALL support multiple probability extraction modes
The system SHALL support multiple modes for extracting VLM probability distributions

#### Scenario: Configure extraction mode
- **WHEN** ScoreFusion is initialized
- **THEN** it SHALL accept prob_mode parameter:
  - "logits": Extract from quality_probs (default)
  - "classification": Use one-hot with smoothing
  - "uniform": Always use uniform (for ablation studies)
- **AND** mode SHALL determine extraction strategy

#### Scenario: Override extraction mode per call
- **WHEN** ScoreFusion.fuse() is called with prob_mode parameter
- **THEN** it SHALL override default mode for this call
- **AND** log the mode used

### Requirement: The system SHALL validate input data quality
The system SHALL validate input data quality before applying fusion

#### Scenario: Validate tool scores
- **WHEN** tool_scores list is provided
- **THEN** ScoreFusion SHALL check:
  - All scores are finite (not NaN or inf)
  - All scores are in range [0, 6] (allow slight out-of-bound for raw scores)
  - If invalid scores found, log ERROR and filter them out
- **AND** if all scores invalid, return None and log ERROR

#### Scenario: Validate VLM probabilities
- **WHEN** VLM probabilities are extracted
- **THEN** ScoreFusion SHALL check:
  - All probabilities are in range [0, 1]
  - Sum of probabilities is approximately 1.0 (tolerance 0.01)
  - If sum != 1.0, renormalize and log WARNING
- **AND** if validation fails after renormalization, return None

#### Scenario: Handle fusion failure gracefully
- **WHEN** Fusion fails due to invalid inputs
- **THEN** ScoreFusion SHALL:
  - Return fallback score: mean(tool_scores) if available
  - Otherwise return 3.0 (neutral quality)
  - Log ERROR with full details
  - Set need_replan=true to request better evidence
