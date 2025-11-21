"""
Unit tests for Planner node in src/agentic/nodes/planner.py
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.agentic.nodes.planner import (
    construct_planner_prompt,
    parse_planner_output,
    planner_node,
    PLANNER_PROMPT_TEMPLATE
)
from src.agentic.state import AgenticIQAState, PlannerOutput


class TestPromptConstruction:
    """Tests for prompt construction."""

    def test_basic_prompt(self):
        """Test constructing basic prompt without reference."""
        prompt = construct_planner_prompt("Is this image blurry?", has_reference=False)
        assert "User's query: Is this image blurry?" in prompt
        assert "System:" in prompt
        assert "Planner in an Image Quality Assessment" in prompt

    def test_prompt_with_reference(self):
        """Test constructing prompt with reference image."""
        prompt = construct_planner_prompt("Compare quality", has_reference=True)
        assert "User's query: Compare quality" in prompt
        assert "reference image is also provided" in prompt


class TestOutputParsing:
    """Tests for JSON output parsing."""

    def test_parse_valid_json(self):
        """Test parsing valid planner JSON."""
        json_str = '''
        {
            "task_type": "IQA",
            "reference_type": "No-Reference",
            "required_object_names": ["vehicle"],
            "required_distortions": {"vehicle": ["Sharpness"]},
            "required_tools": null,
            "distortion_source": "explicit",
            "plan": {
                "distortion_detection": false,
                "distortion_analysis": true,
                "tool_selection": false,
                "tool_execute": false
            }
        }
        '''
        output = parse_planner_output(json_str)
        assert isinstance(output, PlannerOutput)
        assert output.task_type == "IQA"
        assert output.required_object_names == ["vehicle"]

    def test_parse_invalid_json(self):
        """Test that invalid JSON raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            parse_planner_output("not valid json")
        assert "Invalid JSON" in str(exc_info.value)

    def test_parse_invalid_schema(self):
        """Test that JSON not matching schema raises ValueError."""
        json_str = '{"task_type": "INVALID", "required_object_names": null}'
        with pytest.raises(ValueError) as exc_info:
            parse_planner_output(json_str)
        assert "Failed to parse planner output" in str(exc_info.value)


class TestPlannerNode:
    """Tests for planner_node function."""

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image."""
        image_path = tmp_path / "test_image.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        return str(image_path)

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        config = MagicMock()
        config.planner.backend = "openai.gpt-4o"
        config.planner.temperature = 0.0
        config.planner.max_tokens = 2048
        config.planner.top_p = 0.1
        return config

    @pytest.fixture
    def valid_plan_json(self):
        """Valid plan JSON response."""
        return json.dumps({
            "task_type": "IQA",
            "reference_type": "No-Reference",
            "required_object_names": None,
            "required_distortions": None,
            "required_tools": None,
            "distortion_source": "inferred",
            "plan": {
                "distortion_detection": True,
                "distortion_analysis": True,
                "tool_selection": True,
                "tool_execute": True
            }
        })

    def test_planner_node_success(self, temp_image, mock_config, valid_plan_json):
        """Test successful planner node execution."""
        state: AgenticIQAState = {
            "query": "What's the quality?",
            "image_path": temp_image
        }

        # Mock VLM client
        with patch('src.agentic.nodes.planner.create_vlm_client') as mock_create:
            with patch('src.agentic.nodes.planner.load_model_backends') as mock_load_config:
                mock_load_config.return_value = mock_config

                mock_client = MagicMock()
                mock_client.backend_name = "openai"
                mock_client.generate.return_value = valid_plan_json
                mock_create.return_value = mock_client

                # Execute node
                result = planner_node(state)

                # Verify result
                assert "plan" in result
                assert isinstance(result["plan"], PlannerOutput)
                assert result["plan"].task_type == "IQA"

    def test_planner_node_with_reference(self, temp_image, mock_config, valid_plan_json):
        """Test planner with reference image."""
        state: AgenticIQAState = {
            "query": "Compare images",
            "image_path": temp_image,
            "reference_path": temp_image  # Use same image for test
        }

        with patch('src.agentic.nodes.planner.create_vlm_client') as mock_create:
            with patch('src.agentic.nodes.planner.load_model_backends') as mock_load_config:
                mock_load_config.return_value = mock_config

                mock_client = MagicMock()
                mock_client.backend_name = "openai"

                # Update JSON to reflect Full-Reference mode
                plan_data = json.loads(valid_plan_json)
                plan_data["reference_type"] = "Full-Reference"
                mock_client.generate.return_value = json.dumps(plan_data)

                mock_create.return_value = mock_client

                result = planner_node(state)

                assert "plan" in result
                assert result["plan"].reference_type == "Full-Reference"

    def test_planner_node_image_not_found(self):
        """Test error handling for missing image."""
        state: AgenticIQAState = {
            "query": "Test query",
            "image_path": "/nonexistent/image.jpg"
        }

        with patch('src.agentic.nodes.planner.load_model_backends'):
            result = planner_node(state)

            assert "error" in result
            assert "Failed to load images" in result["error"]

    def test_planner_node_invalid_json_retry(self, temp_image, mock_config):
        """Test retry logic on invalid JSON."""
        state: AgenticIQAState = {
            "query": "Test query",
            "image_path": temp_image
        }

        with patch('src.agentic.nodes.planner.create_vlm_client') as mock_create:
            with patch('src.agentic.nodes.planner.load_model_backends') as mock_load_config:
                mock_load_config.return_value = mock_config

                mock_client = MagicMock()
                mock_client.backend_name = "openai"
                # First two attempts return invalid JSON, third succeeds
                valid_json = json.dumps({
                    "task_type": "IQA",
                    "reference_type": "No-Reference",
                    "required_object_names": None,
                    "required_distortions": None,
                    "required_tools": None,
                    "distortion_source": "inferred",
                    "plan": {
                        "distortion_detection": True,
                        "distortion_analysis": True,
                        "tool_selection": True,
                        "tool_execute": True
                    }
                })
                mock_client.generate.side_effect = [
                    "invalid json",
                    "still invalid",
                    valid_json
                ]
                mock_create.return_value = mock_client

                result = planner_node(state, max_retries=3)

                # Should eventually succeed
                assert "plan" in result or "error" in result
                # Verify retry attempts were made
                assert mock_client.generate.call_count == 3

    def test_planner_node_auth_error(self, temp_image, mock_config):
        """Test authentication error handling."""
        state: AgenticIQAState = {
            "query": "Test query",
            "image_path": temp_image
        }

        with patch('src.agentic.nodes.planner.create_vlm_client') as mock_create:
            with patch('src.agentic.nodes.planner.load_model_backends') as mock_load_config:
                mock_load_config.return_value = mock_config

                mock_client = MagicMock()
                mock_client.generate.side_effect = Exception("401 authentication failed")
                mock_create.return_value = mock_client

                result = planner_node(state, max_retries=1)

                assert "error" in result
                assert "authentication failed" in result["error"].lower()
