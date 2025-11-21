"""Tests for Tool Registry."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.agentic.tool_registry import ToolRegistry, ToolExecutionError


@pytest.fixture
def temp_tools_json():
    """Create temporary tools.json file with 5-parameter logistic coefficients."""
    tools_data = {
        "QAlign": {
            "type": "NR",
            "strengths": ["Blurs", "Noise"],
            "logistic_params": {
                "beta1": 28.8204,
                "beta2": 0.1469,
                "beta3": 6.1941,
                "beta4": -0.1906,
                "beta5": 6.7863
            }
        },
        "TOPIQ_FR": {
            "type": "FR",
            "strengths": ["Blurs", "Compression"],
            "logistic_params": {
                "beta1": 21.73,
                "beta2": 0.1147,
                "beta3": 0.4721,
                "beta4": 3.5654,
                "beta5": 1.0094
            }
        },
        "BRISQUE": {
            "type": "NR",
            "strengths": ["Blurs", "Compression", "Noise"],
            "logistic_params": {
                "beta1": -2.2106,
                "beta2": 0.0684,
                "beta3": 54.3418,
                "beta4": 0.0050,
                "beta5": 2.2728
            }
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(tools_data, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestToolRegistryInit:
    """Tests for ToolRegistry initialization."""

    def test_load_metadata(self, temp_tools_json):
        """Test loading tool metadata from JSON."""
        registry = ToolRegistry(metadata_path=temp_tools_json)
        assert len(registry.tools) == 3
        assert "QAlign" in registry.tools
        assert "TOPIQ_FR" in registry.tools

    def test_missing_metadata_file(self):
        """Test handling of missing metadata file."""
        registry = ToolRegistry(metadata_path=Path("/nonexistent/tools.json"))
        assert len(registry.tools) == 0

    def test_invalid_json(self):
        """Test handling of invalid JSON."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            temp_path = Path(f.name)

        with pytest.raises(ValueError, match="Invalid JSON"):
            ToolRegistry(metadata_path=temp_path)

        temp_path.unlink()

    def test_validate_tool_type(self, temp_tools_json):
        """Test validation of tool type field."""
        # Load tools with invalid type
        with open(temp_tools_json, 'r') as f:
            tools_data = json.load(f)

        tools_data["BadTool"] = {"type": "INVALID", "strengths": []}

        with open(temp_tools_json, 'w') as f:
            json.dump(tools_data, f)

        with pytest.raises(ValueError, match="invalid type"):
            ToolRegistry(metadata_path=temp_tools_json)


class TestToolRegistryQueries:
    """Tests for tool registry query methods."""

    def test_get_tools_for_distortion(self, temp_tools_json):
        """Test getting tools for a specific distortion."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Get tools for Blurs
        tools = registry.get_tools_for_distortion("Blurs", reference_available=False)
        assert "QAlign" in tools
        assert "TOPIQ_FR" in tools
        assert "BRISQUE" in tools

    def test_get_tools_prioritize_fr(self, temp_tools_json):
        """Test that FR tools are prioritized when reference available."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        tools = registry.get_tools_for_distortion("Blurs", reference_available=True)
        # TOPIQ_FR should come before NR tools
        fr_index = tools.index("TOPIQ_FR") if "TOPIQ_FR" in tools else -1
        nr_index = tools.index("QAlign") if "QAlign" in tools else -1

        if fr_index >= 0 and nr_index >= 0:
            assert fr_index < nr_index

    def test_get_tools_by_type(self, temp_tools_json):
        """Test getting tools by type."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        fr_tools = registry.get_tools_by_type("FR")
        assert "TOPIQ_FR" in fr_tools
        assert "QAlign" not in fr_tools

        nr_tools = registry.get_tools_by_type("NR")
        assert "QAlign" in nr_tools
        assert "BRISQUE" in nr_tools
        assert "TOPIQ_FR" not in nr_tools

    def test_is_tool_available(self, temp_tools_json):
        """Test checking tool availability."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        assert registry.is_tool_available("QAlign") is True
        assert registry.is_tool_available("NonexistentTool") is False


class TestScoreNormalization:
    """Tests for score normalization."""

    def test_normalize_score_basic(self, temp_tools_json):
        """Test basic score normalization."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Test normalization for QAlign
        normalized = registry.normalize_score("QAlign", 3.5)
        assert 1.0 <= normalized <= 5.0

    def test_normalize_extreme_values(self, temp_tools_json):
        """Test normalization with extreme values."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Very high score - should still be in valid range and clipped if needed
        normalized_high = registry.normalize_score("QAlign", 100.0)
        assert 1.0 <= normalized_high <= 5.0
        assert np.isfinite(normalized_high)

        # Very low score - should still be in valid range and clipped if needed
        normalized_low = registry.normalize_score("QAlign", -100.0)
        assert 1.0 <= normalized_low <= 5.0
        assert np.isfinite(normalized_low)

    def test_normalize_requires_5_params(self, temp_tools_json):
        """Test that normalization requires all 5 beta parameters."""
        # Create tool with incomplete parameters
        with open(temp_tools_json, 'r') as f:
            tools_data = json.load(f)

        # Add tool with only 4 parameters
        tools_data["IncompleteTool"] = {
            "type": "NR",
            "strengths": ["Blurs"],
            "logistic_params": {
                "beta1": 5.0,
                "beta2": 1.0,
                "beta3": 0.5,
                "beta4": 0.1
                # Missing beta5
            }
        }

        with open(temp_tools_json, 'w') as f:
            json.dump(tools_data, f)

        # Should raise ValueError on load due to missing beta5
        with pytest.raises(ValueError, match="missing logistic parameters.*beta5"):
            ToolRegistry(metadata_path=temp_tools_json)

    def test_normalize_unknown_tool(self, temp_tools_json):
        """Test normalization with unknown tool raises error."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        with pytest.raises(ValueError, match="Unknown tool"):
            registry.normalize_score("UnknownTool", 2.0)

    def test_normalize_clipping(self, temp_tools_json):
        """Test that scores outside [1, 5] are clipped."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Test with various raw scores to ensure output is always clipped to [1, 5]
        # The 5-parameter formula should ensure output is in range, with clipping as fallback
        test_scores = [-1000, -100, -10, 0, 0.5, 1, 5, 10, 100, 1000]
        for raw_score in test_scores:
            normalized = registry.normalize_score("QAlign", raw_score)
            assert 1.0 <= normalized <= 5.0, f"Score {normalized} out of range for raw_score={raw_score}"
            assert np.isfinite(normalized)

    def test_normalize_5param_formula(self, temp_tools_json):
        """Test that 5-parameter formula is applied correctly."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Test with a known raw score
        raw_score = 0.5
        normalized = registry.normalize_score("TOPIQ_FR", raw_score)

        # Manually compute expected value using 5-parameter formula
        # q̂ = β₁(½ - 1/(exp(β₂(q̃ - β₃)))) + β₄q̃ + β₅
        beta1, beta2, beta3, beta4, beta5 = 21.73, 0.1147, 0.4721, 3.5654, 1.0094
        exponent = beta2 * (raw_score - beta3)
        exp_term = np.exp(exponent)
        expected = beta1 * (0.5 - 1.0 / exp_term) + beta4 * raw_score + beta5

        # Clip to [1, 5]
        expected = np.clip(expected, 1.0, 5.0)

        # Should match within floating point tolerance
        assert abs(normalized - expected) < 1e-6

    def test_normalize_numerical_stability_large_exponent(self, temp_tools_json):
        """Test numerical stability with very large positive exponent."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Create scenario with large positive exponent
        # For TOPIQ_FR with very large raw_score
        normalized = registry.normalize_score("TOPIQ_FR", 1000.0)

        # Should handle gracefully and return valid score
        assert 1.0 <= normalized <= 5.0
        assert np.isfinite(normalized)

    def test_normalize_numerical_stability_small_exponent(self, temp_tools_json):
        """Test numerical stability with very large negative exponent."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        # Create scenario with large negative exponent
        # For TOPIQ_FR with very small raw_score
        normalized = registry.normalize_score("TOPIQ_FR", -1000.0)

        # Should handle gracefully and return valid score
        assert 1.0 <= normalized <= 5.0
        assert np.isfinite(normalized)

    def test_missing_logistic_params_raises_error(self, temp_tools_json):
        """Test that missing logistic_params raises ValueError."""
        # Create tool without logistic_params
        with open(temp_tools_json, 'r') as f:
            tools_data = json.load(f)

        tools_data["NoParamsTool"] = {
            "type": "NR",
            "strengths": ["Blurs"]
            # No logistic_params
        }

        with open(temp_tools_json, 'w') as f:
            json.dump(tools_data, f)

        # Should raise ValueError on load
        with pytest.raises(ValueError, match="missing 'logistic_params'"):
            ToolRegistry(metadata_path=temp_tools_json)


class TestToolExecution:
    """Tests for tool execution."""

    @pytest.fixture
    def mock_pyiqa(self):
        """Mock pyiqa module."""
        with patch('pyiqa.create_metric') as mock_create:
            mock_metric = MagicMock()
            mock_metric.return_value.item.return_value = 0.75
            mock_create.return_value = mock_metric
            yield mock_create

    @pytest.mark.skipif(True, reason="Requires pyiqa installation or proper mocking")
    def test_execute_nr_tool(self, temp_tools_json, mock_pyiqa, tmp_path):
        """Test executing No-Reference tool."""
        pass

    @pytest.mark.skipif(True, reason="Requires pyiqa installation or proper mocking")
    def test_execute_fr_tool(self, temp_tools_json, mock_pyiqa, tmp_path):
        """Test executing Full-Reference tool."""
        pass

    @pytest.mark.skipif(True, reason="Requires pyiqa installation or proper mocking")
    def test_execute_fr_without_reference(self, temp_tools_json, mock_pyiqa, tmp_path):
        """Test that FR tool without reference raises error."""
        pass

    def test_execute_unknown_tool(self, temp_tools_json):
        """Test executing unknown tool raises error."""
        registry = ToolRegistry(metadata_path=temp_tools_json)

        with pytest.raises(ToolExecutionError, match="Unknown tool"):
            registry.execute_tool("UnknownTool", "/path/to/image.jpg")

    @pytest.mark.skipif(True, reason="Requires pyiqa installation or proper mocking")
    def test_execute_with_nan_score(self, temp_tools_json, tmp_path):
        """Test handling of NaN score from tool."""
        pass


class TestCaching:
    """Tests for tool output caching."""

    @pytest.mark.skipif(True, reason="Requires pyiqa installation for caching tests")
    def test_cache_hit(self):
        """Test cache hit on second call."""
        pass

    @pytest.mark.skipif(True, reason="Requires pyiqa installation for caching tests")
    def test_cache_different_images(self):
        """Test that different images have different cache keys."""
        pass

    @pytest.mark.skipif(True, reason="Requires pyiqa installation for caching tests")
    def test_cache_lru_eviction(self):
        """Test LRU cache eviction."""
        pass

    @pytest.mark.skipif(True, reason="Requires pyiqa installation for caching tests")
    def test_cache_stats(self):
        """Test cache statistics."""
        pass
