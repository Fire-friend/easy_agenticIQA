"""
Unit tests for ScoreFusion utility in src/agentic/score_fusion.py
"""

import pytest
import numpy as np
from src.agentic.score_fusion import ScoreFusion


class TestScoreFusionInit:
    """Tests for ScoreFusion initialization."""

    def test_default_init(self):
        """Test default initialization."""
        fusion = ScoreFusion()
        assert fusion.eta == 1.0
        assert fusion.quality_levels == [1, 2, 3, 4, 5]
        assert fusion.level_to_letter == {5: "A", 4: "B", 3: "C", 2: "D", 1: "E"}

    def test_custom_eta(self):
        """Test initialization with custom eta."""
        fusion = ScoreFusion(eta=2.0)
        assert fusion.eta == 2.0

    def test_custom_quality_levels(self):
        """Test initialization with custom quality levels."""
        fusion = ScoreFusion(quality_levels=[1, 2, 3, 4, 5, 6, 7])
        assert fusion.quality_levels == [1, 2, 3, 4, 5, 6, 7]


class TestPerceptualWeights:
    """Tests for compute_perceptual_weights."""

    def test_empty_tool_scores(self):
        """Test that empty scores return uniform weights."""
        fusion = ScoreFusion()
        weights = fusion.compute_perceptual_weights([])

        # Should be uniform
        assert len(weights) == 5
        for level in [1, 2, 3, 4, 5]:
            assert abs(weights[level] - 0.2) < 1e-6

    def test_single_tool_score(self):
        """Test with single tool score."""
        fusion = ScoreFusion(eta=1.0)
        weights = fusion.compute_perceptual_weights([3.0])

        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # Weight at level 3 should be highest
        assert weights[3] > weights[2]
        assert weights[3] > weights[4]

    def test_multiple_tool_scores(self):
        """Test with multiple tool scores."""
        fusion = ScoreFusion(eta=1.0)
        weights = fusion.compute_perceptual_weights([2.5, 2.7, 2.4])

        # Mean = 2.53, so weight at level 3 should be highest
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert weights[3] > weights[2]
        assert weights[3] > weights[4]

    def test_higher_eta_more_concentrated(self):
        """Test that higher eta makes weights more concentrated."""
        scores = [3.0]

        fusion_low = ScoreFusion(eta=0.5)
        weights_low = fusion_low.compute_perceptual_weights(scores)

        fusion_high = ScoreFusion(eta=2.0)
        weights_high = fusion_high.compute_perceptual_weights(scores)

        # Higher eta should have more weight at level 3
        assert weights_high[3] > weights_low[3]

        # And less weight at extremes
        assert weights_high[1] < weights_low[1]
        assert weights_high[5] < weights_low[5]


class TestVLMProbabilities:
    """Tests for extract_vlm_probabilities."""

    def test_logits_mode(self):
        """Test extraction from logits using softmax."""
        fusion = ScoreFusion()
        logits = {1: -2.0, 2: -1.0, 3: 2.0, 4: 1.0, 5: 0.0}

        probs = fusion.extract_vlm_probabilities(logits, mode="logits")

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-6

        # Level 3 has highest logit, should have highest prob
        assert probs[3] > probs[2]
        assert probs[3] > probs[4]

    def test_logits_mode_invalid_input(self):
        """Test logits mode with invalid input falls back to uniform."""
        fusion = ScoreFusion()
        probs = fusion.extract_vlm_probabilities("not a dict", mode="logits")

        # Should be uniform fallback
        for level in [1, 2, 3, 4, 5]:
            assert abs(probs[level] - 0.2) < 1e-6

    def test_classification_mode_letter(self):
        """Test extraction from classification (letter)."""
        fusion = ScoreFusion()
        probs = fusion.extract_vlm_probabilities("C", mode="classification")

        # Should sum to 1
        assert abs(sum(probs.values()) - 1.0) < 1e-6

        # Level 3 (C) should have high probability (0.7)
        assert abs(probs[3] - 0.7) < 1e-6

        # Others should share remaining 0.3
        assert abs(probs[1] - 0.075) < 1e-6

    def test_classification_mode_quality_name(self):
        """Test extraction from classification (quality name)."""
        fusion = ScoreFusion()
        probs = fusion.extract_vlm_probabilities("Excellent", mode="classification")

        # Level 5 (Excellent) should have high probability
        assert probs[5] > 0.6

    def test_classification_mode_integer(self):
        """Test extraction from classification (integer)."""
        fusion = ScoreFusion()
        probs = fusion.extract_vlm_probabilities(4, mode="classification")

        # Level 4 should have high probability
        assert abs(probs[4] - 0.7) < 1e-6

    def test_classification_mode_dict(self):
        """Test extraction from dict with final_answer key."""
        fusion = ScoreFusion()
        vlm_output = {"final_answer": "B"}
        probs = fusion.extract_vlm_probabilities(vlm_output, mode="classification")

        # Level 4 (B) should have high probability
        assert abs(probs[4] - 0.7) < 1e-6

    def test_uniform_mode(self):
        """Test uniform mode returns uniform distribution."""
        fusion = ScoreFusion()
        probs = fusion.extract_vlm_probabilities("anything", mode="uniform")

        # Should be uniform
        for level in [1, 2, 3, 4, 5]:
            assert abs(probs[level] - 0.2) < 1e-6


class TestFuseScores:
    """Tests for fuse_scores."""

    def test_basic_fusion(self):
        """Test basic fusion with tool scores and VLM probs."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [3.0]
        vlm_probs = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Should be in valid range
        assert 1.0 <= fused <= 5.0

        # Should be close to 3 (both tool and VLM favor level 3)
        assert 2.5 <= fused <= 3.5

    def test_fusion_with_disagreement(self):
        """Test fusion when tool and VLM disagree."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [2.0]  # Tool favors low quality
        vlm_probs = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.6}  # VLM favors high

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Should be between tool mean and VLM preference
        assert 2.0 <= fused <= 5.0

        # Weighted average should lean toward tool score due to Gaussian weights
        assert fused < 4.0

    def test_fusion_clipping(self):
        """Test that fusion clips to [1, 5] range."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [5.0]
        # Extreme VLM distribution favoring high
        vlm_probs = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 1.0}

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Should be clipped to 5.0
        assert fused <= 5.0
        assert fused >= 1.0

    def test_fusion_with_empty_tools(self):
        """Test fusion with no tool scores (uniform alpha)."""
        fusion = ScoreFusion()
        vlm_probs = {1: 0.1, 2: 0.2, 3: 0.4, 4: 0.2, 5: 0.1}

        fused = fusion.fuse_scores([], vlm_probs)

        # Should still compute valid score
        assert 1.0 <= fused <= 5.0


class TestMapping:
    """Tests for score mapping functions."""

    def test_map_to_level_default(self):
        """Test default rounding mapping."""
        fusion = ScoreFusion()

        assert fusion.map_to_level(2.4) == 2
        assert fusion.map_to_level(2.5) == 2  # Python rounds 0.5 to even
        assert fusion.map_to_level(2.6) == 3
        assert fusion.map_to_level(3.5) == 4

    def test_map_to_level_custom_thresholds(self):
        """Test custom threshold mapping."""
        fusion = ScoreFusion()
        thresholds = [1.5, 2.5, 3.5, 4.5]

        assert fusion.map_to_level(2.7, thresholds) == 3
        assert fusion.map_to_level(1.3, thresholds) == 1
        assert fusion.map_to_level(4.8, thresholds) == 5

    def test_map_to_level_clipping(self):
        """Test that mapping clips to valid levels."""
        fusion = ScoreFusion()

        # These should be clipped to valid range
        assert fusion.map_to_level(0.5) in [1, 2, 3, 4, 5]
        assert fusion.map_to_level(6.0) in [1, 2, 3, 4, 5]

    def test_map_to_letter(self):
        """Test mapping levels to letter grades."""
        fusion = ScoreFusion()

        assert fusion.map_to_letter(5) == "A"
        assert fusion.map_to_letter(4) == "B"
        assert fusion.map_to_letter(3) == "C"
        assert fusion.map_to_letter(2) == "D"
        assert fusion.map_to_letter(1) == "E"

    def test_map_to_letter_invalid_level(self):
        """Test that invalid level returns default 'C'."""
        fusion = ScoreFusion()
        assert fusion.map_to_letter(99) == "C"


class TestIntegrationScenarios:
    """Integration test cases from spec."""

    def test_case_1_single_tool_uniform_vlm(self):
        """Test case: Single tool score with uniform VLM."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [2.6]
        vlm_probs = {1: 0.2, 2: 0.2, 3: 0.2, 4: 0.2, 5: 0.2}

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Alpha weights peak at level 3 (closest to 2.6)
        # With uniform VLM, result should be close to 2.6
        assert 2.4 <= fused <= 2.8

    def test_case_2_multiple_tools_classification_vlm(self):
        """Test case: Multiple tool scores with VLM classification."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [2.5, 2.7, 2.4]  # Mean = 2.53
        vlm_probs = fusion.extract_vlm_probabilities("C", mode="classification")  # Level 3

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Should be between tool mean (2.53) and VLM preference (3)
        assert 2.5 <= fused <= 3.0

    def test_case_3_tool_vlm_disagree(self):
        """Test case: Tool and VLM disagree significantly."""
        fusion = ScoreFusion(eta=1.0)
        tool_scores = [2.0]  # Tool favors level 2
        vlm_probs = {1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.6}  # VLM favors level 5

        fused = fusion.fuse_scores(tool_scores, vlm_probs)

        # Alpha weights favor level 2 (Gaussian centered at 2.0)
        # Final score should lean toward tool score
        assert fused < 4.0


class TestGetFusionStats:
    """Tests for get_fusion_stats."""

    def test_fusion_stats(self):
        """Test getting fusion statistics."""
        fusion = ScoreFusion(eta=1.5, quality_levels=[1, 2, 3, 4, 5])
        stats = fusion.get_fusion_stats()

        assert stats["eta"] == 1.5
        assert stats["n_levels"] == 5
        assert stats["levels"] == [1, 2, 3, 4, 5]
