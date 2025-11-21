"""
Score fusion utility for combining IQA tool scores with VLM probability distributions.

Implements the weighted fusion algorithm from the AgenticIQA paper:
1. Compute perceptual weights using Gaussian distribution centered at tool score mean
2. Extract VLM probabilities from logits, classification, or use uniform distribution
3. Apply fusion formula: q = Σ_c α_c · p_c · c
4. Map continuous scores to discrete quality levels
"""

import logging
from typing import Dict, List, Optional, Union, Literal
import numpy as np

logger = logging.getLogger(__name__)


class ScoreFusion:
    """
    Weighted score fusion for combining tool scores with VLM probabilities.

    Args:
        eta: Gaussian concentration parameter (default=1.0)
        quality_levels: Quality levels to use (default=[1,2,3,4,5])
    """

    def __init__(
        self,
        eta: float = 1.0,
        quality_levels: Optional[List[int]] = None
    ):
        self.eta = eta
        self.quality_levels = quality_levels or [1, 2, 3, 4, 5]

        # Letter grade mapping (for 1-5 scale)
        self.level_to_letter = {
            5: "A",  # Excellent
            4: "B",  # Good
            3: "C",  # Fair
            2: "D",  # Poor
            1: "E"   # Bad
        }

    def compute_perceptual_weights(
        self,
        tool_scores: List[float]
    ) -> Dict[int, float]:
        """
        Compute Gaussian perceptual weights centered at tool score mean.

        Formula: α_c = exp(-η(q̄ - c)²) / Σ_j exp(-η(q̄ - j)²)

        Args:
            tool_scores: List of tool scores in [1, 5] range

        Returns:
            Dictionary mapping quality level to weight
        """
        if not tool_scores:
            logger.warning("Empty tool scores, using uniform weights")
            n_levels = len(self.quality_levels)
            return {level: 1.0 / n_levels for level in self.quality_levels}

        # Compute mean tool score
        q_bar = np.mean(tool_scores)
        logger.debug(f"Tool score mean: {q_bar:.3f} from {len(tool_scores)} scores")

        # Compute Gaussian weights for each quality level
        # Use numerically stable computation (subtract max before exp)
        exponents = [-self.eta * (q_bar - c) ** 2 for c in self.quality_levels]
        max_exp = max(exponents)
        exp_values = [np.exp(e - max_exp) for e in exponents]
        sum_exp = sum(exp_values)

        weights = {
            level: exp_val / sum_exp
            for level, exp_val in zip(self.quality_levels, exp_values)
        }

        logger.debug(f"Perceptual weights: {weights}")
        return weights

    def extract_vlm_probs(
        self,
        vlm_output: Dict
    ) -> Dict[int, float]:
        """
        Extract VLM probability distribution from model output.

        Tries multiple extraction methods in order:
        1. quality_probs dict with log-probabilities
        2. quality_level classification with smoothing
        3. Uniform distribution fallback

        Args:
            vlm_output: VLM JSON output dict

        Returns:
            Dictionary mapping quality level to probability
        """
        # Method 1: Extract from quality_probs (primary method)
        if isinstance(vlm_output, dict) and "quality_probs" in vlm_output:
            quality_probs = vlm_output["quality_probs"]

            # Parse log-probabilities for levels 1-5
            log_probs = []
            for level in self.quality_levels:
                log_p = quality_probs.get(str(level), -np.inf)
                if not np.isfinite(log_p):
                    logger.warning(f"Invalid log-prob for level {level}: {log_p}")
                    log_p = -10.0  # Use large negative value instead of -inf
                log_probs.append(log_p)

            # Apply numerically stable softmax
            max_log_p = max(log_probs)
            exp_values = [np.exp(lp - max_log_p) for lp in log_probs]
            sum_exp = sum(exp_values)

            if sum_exp == 0:
                logger.error("Softmax normalization failed, using uniform distribution")
                return self._uniform_probabilities()

            probs = {
                level: exp_val / sum_exp
                for level, exp_val in zip(self.quality_levels, exp_values)
            }

            # Validate probabilities sum to 1.0
            prob_sum = sum(probs.values())
            if not (0.99 <= prob_sum <= 1.01):
                logger.warning(f"VLM probs sum to {prob_sum:.3f}, renormalizing")
                probs = {level: p / prob_sum for level, p in probs.items()}

            logger.info(f"Extracted VLM probabilities from quality_probs: {probs}")
            return probs

        # Method 2: Fallback to classification with smoothing
        if isinstance(vlm_output, dict) and "quality_level" in vlm_output:
            predicted_level = vlm_output["quality_level"]

            if isinstance(predicted_level, str):
                predicted_level = self._parse_classification(predicted_level)

            if predicted_level in self.quality_levels:
                # One-hot with smoothing: 0.7 for predicted, 0.075 for others
                high_prob = 0.7
                low_prob = 0.075

                probs = {
                    level: high_prob if level == predicted_level else low_prob
                    for level in self.quality_levels
                }

                logger.warning("Using classification fallback with smoothing")
                logger.info(f"Classification-based probabilities (level {predicted_level}): {probs}")
                return probs

        # Method 3: Fallback to uniform distribution
        logger.warning("No VLM probabilities available, using uniform distribution")
        return self._uniform_probabilities()

    # Keep backward compatibility alias
    def extract_vlm_probabilities(
        self,
        vlm_output: Union[Dict, str, int],
        mode: Literal["logits", "classification", "uniform"] = "classification"
    ) -> Dict[int, float]:
        """
        Legacy method for backward compatibility.
        Redirects to extract_vlm_probs() with appropriate handling.
        """
        if mode == "logits" and isinstance(vlm_output, dict):
            return self.extract_vlm_probs({"quality_probs": vlm_output})
        elif mode == "classification":
            if isinstance(vlm_output, dict):
                return self.extract_vlm_probs(vlm_output)
            else:
                parsed_level = self._parse_classification(vlm_output)
                if parsed_level:
                    return self.extract_vlm_probs({"quality_level": parsed_level})

        return self._uniform_probabilities()

    def _uniform_probabilities(self) -> Dict[int, float]:
        """Return uniform probability distribution."""
        n_levels = len(self.quality_levels)
        return {level: 1.0 / n_levels for level in self.quality_levels}

    def _parse_classification(self, vlm_output: Union[str, int, Dict]) -> Optional[int]:
        """
        Parse VLM classification output to quality level.

        Handles:
        - Direct int: 3 → 3
        - Letter grade: "C" → 3
        - Quality name: "Fair" → 3
        - Dict with key: {"final_answer": "C"} → 3
        """
        if isinstance(vlm_output, int):
            if vlm_output in self.quality_levels:
                return vlm_output
            return None

        if isinstance(vlm_output, dict):
            # Try to extract from dict
            for key in ['final_answer', 'answer', 'level', 'classification']:
                if key in vlm_output:
                    return self._parse_classification(vlm_output[key])
            return None

        if isinstance(vlm_output, str):
            vlm_output = vlm_output.strip().upper()

            # Letter grade mapping
            letter_to_level = {
                "A": 5, "EXCELLENT": 5,
                "B": 4, "GOOD": 4,
                "C": 3, "FAIR": 3,
                "D": 2, "POOR": 2,
                "E": 1, "BAD": 1
            }

            if vlm_output in letter_to_level:
                return letter_to_level[vlm_output]

            # Try to parse as int
            try:
                level = int(vlm_output)
                if level in self.quality_levels:
                    return level
            except ValueError:
                pass

        return None

    def fuse(
        self,
        tool_scores: List[float],
        vlm_probs: Dict[int, float]
    ) -> float:
        """
        Apply fusion formula to compute final quality score.

        Formula:
            w_c = α_c · p_c
            q = Σ(w_c · c) / Σ(w_c)

        where:
        - α_c: perceptual weights (sum to 1.0)
        - p_c: VLM probabilities (sum to 1.0)
        - w_c: joint weights (normalized)
        - c: quality level

        Args:
            tool_scores: List of tool scores
            vlm_probs: VLM probability distribution

        Returns:
            Fused quality score in [1, 5] range
        """
        # Compute perceptual weights α from tool scores
        alpha = self.compute_perceptual_weights(tool_scores)

        # Validate inputs
        prob_sum = sum(vlm_probs.values())
        if not (0.99 <= prob_sum <= 1.01):
            logger.warning(f"VLM probs sum to {prob_sum:.3f}, should be ~1.0")

        alpha_sum = sum(alpha.values())
        if not (0.99 <= alpha_sum <= 1.01):
            logger.error(f"Alpha weights sum to {alpha_sum:.3f}, should be 1.0")

        # Compute joint weights: w_c = α_c · p_c
        joint_weights = {}
        for c in self.quality_levels:
            joint_weights[c] = alpha[c] * vlm_probs.get(c, 0)

        # Normalize joint weights
        weight_sum = sum(joint_weights.values())

        if weight_sum < 1e-10:
            # Fallback: if joint weights are too small, use tool mean
            logger.warning("Joint weights sum to near zero, using tool score mean")
            q = np.mean(tool_scores) if tool_scores else 3.0
        else:
            # Apply normalized fusion formula: q = Σ(w_c · c) / Σ(w_c)
            numerator = sum(joint_weights[c] * c for c in self.quality_levels)
            q = numerator / weight_sum

        # Logging
        tool_mean = np.mean(tool_scores) if tool_scores else 0.0
        logger.info(f"Score Fusion: q={q:.3f} (tool_mean={tool_mean:.3f})")
        logger.debug(f"  Alpha weights: {alpha}")
        logger.debug(f"  VLM probs: {vlm_probs}")
        logger.debug(f"  Joint weights sum: {weight_sum:.6f}")

        return float(q)

    # Keep backward compatibility alias
    def fuse_scores(
        self,
        tool_scores: List[float],
        vlm_probabilities: Dict[int, float]
    ) -> float:
        """Legacy method for backward compatibility. Redirects to fuse()."""
        return self.fuse(tool_scores, vlm_probabilities)

    def map_to_level(
        self,
        score: float,
        thresholds: Optional[List[float]] = None
    ) -> int:
        """
        Map continuous score to discrete quality level.

        Args:
            score: Continuous score
            thresholds: Optional custom thresholds (default: rounding)

        Returns:
            Discrete quality level
        """
        if thresholds is None:
            # Simple rounding
            level = round(score)
        else:
            # Use custom thresholds
            # Example: [1.5, 2.5, 3.5, 4.5] → score=2.7 maps to level 3
            level = 1
            for i, threshold in enumerate(thresholds):
                if score > threshold:
                    level = i + 2  # Levels start at 1

        # Ensure level is valid
        level = max(min(self.quality_levels), min(level, max(self.quality_levels)))

        return level

    def map_to_letter(self, level: int) -> str:
        """
        Map quality level to letter grade.

        Args:
            level: Quality level (1-5)

        Returns:
            Letter grade (A-E)
        """
        return self.level_to_letter.get(level, "C")  # Default to C if invalid

    def get_fusion_stats(self) -> Dict[str, float]:
        """
        Get fusion statistics for logging/debugging.

        Returns:
            Dictionary with fusion statistics
        """
        return {
            "eta": self.eta,
            "n_levels": len(self.quality_levels),
            "levels": self.quality_levels
        }
