"""Integration tests for Planner→Executor flow."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

from src.agentic.graph import create_agentic_graph, compile_graph, run_pipeline
from src.agentic.state import AgenticIQAState, PlannerOutput, PlanControlFlags


@pytest.fixture
def temp_image():
    """Create temporary test image."""
    from PIL import Image
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create simple test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f.name)
        temp_path = Path(f.name)

    yield str(temp_path)

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def valid_plan():
    """Create valid Planner output."""
    return PlannerOutput(
        query_type="IQA",
        query_scope=["vehicle"],
        distortion_source="Explicit",
        distortions={"vehicle": ["Blurs"]},
        reference_mode="No-Reference",
        required_tool=None,
        plan=PlanControlFlags(
            distortion_detection=False,
            distortion_analysis=True,
            tool_selection=True,
            tool_execution=True
        )
    )


class TestPlannerToExecutorFlow:
    """Tests for Planner→Executor integration."""

    def test_graph_creation_with_executor(self):
        """Test that graph includes Executor node."""
        graph = create_agentic_graph()

        # Check nodes exist
        assert "planner" in graph.nodes
        assert "executor" in graph.nodes

    def test_graph_compilation(self):
        """Test that graph with Executor compiles successfully."""
        graph = create_agentic_graph()
        compiled = compile_graph(graph)

        assert compiled is not None

    @patch('src.agentic.nodes.planner.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    def test_end_to_end_flow(
        self,
        mock_tool_registry,
        mock_executor_config,
        mock_executor_vlm,
        mock_planner_config,
        mock_planner_vlm,
        temp_image,
        valid_plan
    ):
        """Test end-to-end Planner→Executor flow with mocks."""
        # Mock Planner
        mock_planner_backend = MagicMock()
        mock_planner_backend.planner.backend = "openai.gpt-4o"
        mock_planner_backend.planner.temperature = 0.0
        mock_planner_backend.planner.max_tokens = 2048
        mock_planner_config.return_value = mock_planner_backend

        mock_planner_client = MagicMock()
        mock_planner_client.backend_name = "openai"
        mock_planner_client.generate.return_value = valid_plan.model_dump_json()
        mock_planner_vlm.return_value = mock_planner_client

        # Mock Executor
        mock_executor_backend = MagicMock()
        mock_executor_backend.executor.backend = "openai.gpt-4o"
        mock_executor_backend.executor.temperature = 0.0
        mock_executor_config.return_value = mock_executor_backend

        mock_executor_client = MagicMock()
        mock_executor_client.backend_name = "openai"

        # Mock distortion analysis response
        mock_executor_client.generate.return_value = '''
        {
          "distortion_analysis": {
            "vehicle": [
              {
                "type": "Blurs",
                "severity": "severe",
                "explanation": "Strong motion blur"
              }
            ]
          }
        }
        '''
        mock_executor_vlm.return_value = mock_executor_client

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.tools = {
            "QAlign": {"type": "NR", "strengths": ["Blurs"]}
        }
        mock_registry_instance.is_tool_available.return_value = True
        mock_registry_instance.execute_tool.return_value = (2.5, 3.2)
        mock_registry_instance.get_tools_for_distortion.return_value = ["QAlign"]  # Return list of strings
        mock_registry_instance.get_cache_stats.return_value = {
            'hits': 0, 'misses': 1, 'hit_rate': 0.0
        }
        mock_tool_registry.return_value = mock_registry_instance

        # Run pipeline
        final_state = run_pipeline(
            query="What's the quality of the vehicle?",
            image_path=temp_image
        )

        # Verify state contains both plan and executor_evidence
        assert "plan" in final_state
        assert "executor_evidence" in final_state
        assert final_state["plan"].query_type == "IQA"
        assert final_state["executor_evidence"] is not None

    def test_executor_handles_missing_plan(self, temp_image):
        """Test that Executor handles missing plan gracefully."""
        from src.agentic.nodes.executor import executor_node

        # State without plan
        state: AgenticIQAState = {
            "query": "Test query",
            "image_path": temp_image
        }

        result = executor_node(state)

        # Should return error
        assert "error" in result
        assert "No plan found" in result["error"]

    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    def test_executor_with_all_flags_disabled(
        self,
        mock_tool_registry,
        mock_config,
        mock_vlm,
        temp_image,
        valid_plan
    ):
        """Test Executor with all control flags disabled."""
        from src.agentic.nodes.executor import executor_node

        # Disable all flags
        plan_no_tasks = PlannerOutput(
            query_type="IQA",
            query_scope="Global",
            distortion_source="Explicit",
            distortions={"Global": ["Noise"]},
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=False,
                distortion_analysis=False,
                tool_selection=False,
                tool_execution=False
            )
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image,
            "plan": plan_no_tasks
        }

        # Mock config
        mock_backend = MagicMock()
        mock_backend.executor.backend = "openai.gpt-4o"
        mock_config.return_value = mock_backend

        # Mock VLM client
        mock_client = MagicMock()
        mock_vlm.return_value = mock_client

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.tools = {}
        mock_tool_registry.return_value = mock_registry_instance

        result = executor_node(state)

        # Should complete without errors
        assert "executor_evidence" in result
        assert result["executor_evidence"].distortion_set is not None  # From plan
        assert result["executor_evidence"].distortion_analysis is None
        assert result["executor_evidence"].selected_tools is None
        assert result["executor_evidence"].quality_scores is None

    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    def test_executor_with_tool_execution_only(
        self,
        mock_tool_registry,
        mock_config,
        mock_vlm,
        temp_image
    ):
        """Test Executor with only tool execution enabled."""
        from src.agentic.nodes.executor import executor_node

        # Create plan with preselected tools
        plan_with_tools = PlannerOutput(
            query_type="IQA",
            query_scope=["vehicle"],
            distortion_source="Explicit",
            distortions={"vehicle": ["Blurs"]},
            reference_mode="No-Reference",
            required_tool="QAlign",
            plan=PlanControlFlags(
                distortion_detection=False,
                distortion_analysis=False,
                tool_selection=True,
                tool_execution=True
            )
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image,
            "plan": plan_with_tools
        }

        # Mock config
        mock_backend = MagicMock()
        mock_backend.executor.backend = "openai.gpt-4o"
        mock_config.return_value = mock_backend

        # Mock VLM client
        mock_client = MagicMock()
        mock_vlm.return_value = mock_client

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.tools = {"QAlign": {"type": "NR", "strengths": ["Blurs"]}}
        mock_registry_instance.is_tool_available.return_value = True
        mock_registry_instance.execute_tool.return_value = (2.5, 3.2)
        mock_registry_instance.get_cache_stats.return_value = {'hits': 0, 'misses': 1}
        mock_tool_registry.return_value = mock_registry_instance

        result = executor_node(state)

        # Should have tool execution results
        assert "executor_evidence" in result
        evidence = result["executor_evidence"]
        assert evidence.selected_tools is not None
        assert evidence.quality_scores is not None
        assert len(evidence.tool_logs) > 0

        # Verify tool was executed
        mock_registry_instance.execute_tool.assert_called()


class TestExecutorErrorHandling:
    """Tests for Executor error handling."""

    @patch('src.utils.config.load_model_backends')
    def test_executor_handles_image_load_error(self, mock_config):
        """Test that Executor handles image loading errors."""
        from src.agentic.nodes.executor import executor_node

        # Mock config to avoid config loading errors
        mock_backend = MagicMock()
        mock_backend.executor.backend = "openai.gpt-4o"
        mock_backend.executor.temperature = 0.0
        mock_config.return_value = mock_backend

        plan = PlannerOutput(
            query_type="IQA",
            query_scope="Global",
            distortion_source="Explicit",
            distortions={"Global": ["Noise"]},
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=False,
                distortion_analysis=False,
                tool_selection=False,
                tool_execution=False
            )
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": "/nonexistent/image.jpg",
            "plan": plan
        }

        result = executor_node(state)

        # Should return error
        assert "error" in result
        assert "Failed to load images" in result["error"]

    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.create_vlm_client')
    def test_executor_handles_vlm_client_creation_error(self, mock_vlm, mock_config, temp_image):
        """Test that Executor handles VLM client creation errors."""
        from src.agentic.nodes.executor import executor_node

        # Mock config to avoid config loading errors
        mock_backend = MagicMock()
        mock_backend.executor.backend = "openai.gpt-4o"
        mock_backend.executor.temperature = 0.0
        mock_config.return_value = mock_backend

        plan = PlannerOutput(
            query_type="IQA",
            query_scope="Global",
            distortion_source="Explicit",
            distortions={"Global": ["Noise"]},
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=False,
                distortion_analysis=True,
                tool_selection=False,
                tool_execution=False
            )
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image,
            "plan": plan
        }

        # Make VLM client creation fail
        mock_vlm.side_effect = Exception("VLM client creation failed")

        result = executor_node(state)

        # Should return error
        assert "error" in result
        assert "Failed to create VLM client" in result["error"]
