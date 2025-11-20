"""
Integration tests for Phase 4: Planner→Executor→Summarizer flow with replanning.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agentic.graph import create_agentic_graph, compile_graph, run_pipeline
from src.agentic.state import (
    AgenticIQAState,
    PlannerOutput,
    PlanControlFlags,
    ExecutorOutput,
    DistortionAnalysis
)


@pytest.fixture
def temp_image():
    """Create temporary test image."""
    from PIL import Image
    import numpy as np

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(f.name)
        temp_path = Path(f.name)

    yield str(temp_path)

    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def valid_plan_json():
    """Valid plan JSON for IQA task."""
    plan = {
        "query_type": "IQA",
        "query_scope": "Global",
        "distortion_source": "Inferred",
        "distortions": None,
        "reference_mode": "No-Reference",
        "required_tool": None,
        "plan": {
            "distortion_detection": True,
            "distortion_analysis": True,
            "tool_selection": True,
            "tool_execution": True
        }
    }
    return json.dumps(plan)


@pytest.fixture
def executor_json_response():
    """Valid Executor JSON response."""
    executor_response = {
        "distortion_set": {"Global": ["Noise", "Blurs"]},
        "distortion_analysis": {
            "Global": [
                {
                    "type": "Noise",
                    "severity": "slight",
                    "explanation": "Minor noise visible in uniform areas."
                },
                {
                    "type": "Blurs",
                    "severity": "moderate",
                    "explanation": "Moderate blur affecting edges."
                }
            ]
        },
        "selected_tools": {
            "Global": {
                "Noise": "BRISQUE",
                "Blurs": "QAlign"
            }
        },
        "quality_scores": {
            "Global": {
                "Noise": ["BRISQUE", 3.2],
                "Blurs": ["QAlign", 2.8]
            }
        }
    }
    return json.dumps(executor_response)


@pytest.fixture
def summarizer_json_response_no_replan():
    """Valid Summarizer JSON response without replanning."""
    return json.dumps({
        "final_answer": "C",
        "quality_reasoning": "Fair quality due to moderate blur and slight noise, consistent with tool scores (BRISQUE: 3.2, QAlign: 2.8)."
    })


@pytest.fixture
def summarizer_json_response_with_replan():
    """Valid Summarizer JSON response requesting replanning."""
    return json.dumps({
        "final_answer": "Unable to determine",
        "quality_reasoning": "Insufficient evidence for complete assessment."
    })


class TestPhase4GraphStructure:
    """Tests for Phase 4 graph structure."""

    def test_graph_includes_all_nodes(self):
        """Test that graph includes Planner, Executor, and Summarizer."""
        graph = create_agentic_graph()

        assert "planner" in graph.nodes
        assert "executor" in graph.nodes
        assert "summarizer" in graph.nodes

    def test_graph_compiles_successfully(self):
        """Test that Phase 4 graph compiles without errors."""
        graph = create_agentic_graph()
        compiled = compile_graph(graph)

        assert compiled is not None


class TestFullPipelineWithoutReplanning:
    """Tests for complete pipeline execution without replanning."""

    @patch('src.agentic.nodes.planner.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.planner.load_image')
    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.load_image')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_end_to_end_scoring_task(
        self,
        mock_summ_load_image,
        mock_summ_config,
        mock_summ_vlm,
        mock_tool_registry,
        mock_exec_load_image,
        mock_exec_config,
        mock_exec_vlm,
        mock_plan_load_image,
        mock_plan_config,
        mock_plan_vlm,
        temp_image,
        valid_plan_json,
        executor_json_response,
        summarizer_json_response_no_replan
    ):
        """Test end-to-end pipeline for IQA scoring task."""
        # Mock Planner
        mock_plan_backend = MagicMock()
        mock_plan_backend.planner.backend = "openai.gpt-4o"
        mock_plan_backend.planner.temperature = 0.0
        mock_plan_backend.planner.max_tokens = 2048
        mock_plan_config.return_value = mock_plan_backend

        mock_plan_client = MagicMock()
        mock_plan_client.backend_name = "openai"
        mock_plan_client.generate.return_value = valid_plan_json
        mock_plan_vlm.return_value = mock_plan_client

        mock_plan_image = MagicMock()
        mock_plan_load_image.return_value = mock_plan_image

        # Mock Executor
        mock_exec_backend = MagicMock()
        mock_exec_backend.executor.backend = "openai.gpt-4o"
        mock_exec_backend.executor.temperature = 0.0
        mock_exec_backend.executor.max_tokens = 4096
        mock_exec_config.return_value = mock_exec_backend

        mock_exec_client = MagicMock()
        mock_exec_client.backend_name = "openai"
        # Return JSON responses for subtasks
        mock_exec_client.generate.side_effect = [
            executor_json_response,  # First subtask
            executor_json_response,  # Subsequent subtasks
            executor_json_response,
            executor_json_response
        ]
        mock_exec_vlm.return_value = mock_exec_client

        mock_exec_image = MagicMock()
        mock_exec_load_image.return_value = mock_exec_image

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.execute_tool.return_value = (2.5, 3.0)
        mock_registry_instance.get_tools_for_distortion.return_value = ["QAlign"]
        mock_registry_instance.is_tool_available.return_value = True
        mock_registry_instance.get_cache_stats.return_value = {'hits': 0, 'misses': 1}
        mock_tool_registry.return_value = mock_registry_instance

        # Mock Summarizer
        mock_summ_backend = MagicMock()
        mock_summ_backend.summarizer.backend = "openai.gpt-4o"
        mock_summ_backend.summarizer.temperature = 0.0
        mock_summ_backend.summarizer.max_tokens = 512
        mock_summ_config.return_value = mock_summ_backend

        mock_summ_client = MagicMock()
        mock_summ_client.backend_name = "openai"
        mock_summ_client.generate.return_value = summarizer_json_response_no_replan
        mock_summ_vlm.return_value = mock_summ_client

        mock_summ_image = MagicMock()
        mock_summ_load_image.return_value = mock_summ_image

        # Run pipeline
        final_state = run_pipeline(
            query="What is the quality?",
            image_path=temp_image,
            max_replan_iterations=2
        )

        # Verify complete flow
        assert "plan" in final_state
        assert final_state["plan"].query_type == "IQA"

        assert "executor_evidence" in final_state
        assert final_state["executor_evidence"] is not None

        assert "summarizer_result" in final_state
        assert final_state["summarizer_result"].final_answer == "C"
        assert final_state["summarizer_result"].need_replan is False

        # Verify no replanning occurred
        assert final_state["iteration_count"] == 0
        assert len(final_state["replan_history"]) == 0


class TestPipelineWithReplanning:
    """Tests for pipeline with replanning scenarios."""

    @patch('src.agentic.nodes.planner.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.planner.load_image')
    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.load_image')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_pipeline_with_single_replan(
        self,
        mock_summ_load_image,
        mock_summ_config,
        mock_summ_vlm,
        mock_tool_registry,
        mock_exec_load_image,
        mock_exec_config,
        mock_exec_vlm,
        mock_plan_load_image,
        mock_plan_config,
        mock_plan_vlm,
        temp_image
    ):
        """Test pipeline with one replanning iteration."""
        # First plan - incomplete
        plan_json_1 = json.dumps({
            "query_type": "IQA",
            "query_scope": ["car", "person"],
            "distortion_source": "Inferred",
            "distortions": None,
            "reference_mode": "No-Reference",
            "required_tool": None,
            "plan": {
                "distortion_detection": True,
                "distortion_analysis": True,
                "tool_selection": True,
                "tool_execution": True
            }
        })

        # Second plan - after replan
        plan_json_2 = json.dumps({
            "query_type": "IQA",
            "query_scope": ["car", "person"],
            "distortion_source": "Inferred",
            "distortions": {"car": ["Blurs"], "person": ["Noise"]},
            "reference_mode": "No-Reference",
            "required_tool": None,
            "plan": {
                "distortion_detection": False,
                "distortion_analysis": True,
                "tool_selection": True,
                "tool_execution": True
            }
        })

        # Mock Planner - return different plans
        mock_plan_backend = MagicMock()
        mock_plan_backend.planner.backend = "openai.gpt-4o"
        mock_plan_backend.planner.temperature = 0.0
        mock_plan_backend.planner.max_tokens = 2048
        mock_plan_config.return_value = mock_plan_backend

        mock_plan_client = MagicMock()
        mock_plan_client.backend_name = "openai"
        mock_plan_client.generate.side_effect = [plan_json_1, plan_json_2]
        mock_plan_vlm.return_value = mock_plan_client

        mock_plan_load_image.return_value = MagicMock()

        # Mock Executor - incomplete first time, complete second time
        executor_json_incomplete = json.dumps({
            "distortion_analysis": {
                "car": [{"type": "Blurs", "severity": "moderate", "explanation": "Blur detected"}]
                # Missing 'person'
            },
            "quality_scores": {
                "car": {"Blurs": ["QAlign", 2.8]}
            }
        })

        executor_json_complete = json.dumps({
            "distortion_analysis": {
                "car": [{"type": "Blurs", "severity": "moderate", "explanation": "Blur detected"}],
                "person": [{"type": "Noise", "severity": "slight", "explanation": "Noise detected"}]
            },
            "quality_scores": {
                "car": {"Blurs": ["QAlign", 2.8]},
                "person": {"Noise": ["BRISQUE", 3.5]}
            }
        })

        mock_exec_backend = MagicMock()
        mock_exec_backend.executor.backend = "openai.gpt-4o"
        mock_exec_backend.executor.temperature = 0.0
        mock_exec_backend.executor.max_tokens = 4096
        mock_exec_config.return_value = mock_exec_backend

        mock_exec_client = MagicMock()
        mock_exec_client.backend_name = "openai"
        # First executor run (incomplete), then second run (complete)
        mock_exec_client.generate.side_effect = [
            executor_json_incomplete, executor_json_incomplete,
            executor_json_incomplete, executor_json_incomplete,
            executor_json_complete, executor_json_complete,
            executor_json_complete, executor_json_complete
        ]
        mock_exec_vlm.return_value = mock_exec_client

        mock_exec_load_image.return_value = MagicMock()

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.execute_tool.return_value = (2.5, 3.0)
        mock_registry_instance.get_tools_for_distortion.return_value = ["QAlign"]
        mock_registry_instance.is_tool_available.return_value = True
        mock_registry_instance.get_cache_stats.return_value = {'hits': 0, 'misses': 1}
        mock_tool_registry.return_value = mock_registry_instance

        # Mock Summarizer - request replan once, then success
        # Note: Only need 2 summarizer calls since we expect 1 replan total
        # Call 1 (iteration 0): Request replan due to incomplete evidence
        # Call 2 (iteration 1): Success with complete evidence
        summarizer_replan = json.dumps({
            "final_answer": "Unable to determine",
            "quality_reasoning": "Insufficient evidence for person region.",
            "need_replan": True,
            "replan_reason": "Missing evidence for person region"
        })

        summarizer_success = json.dumps({
            "final_answer": "C",
            "quality_reasoning": "Fair quality overall.",
            "need_replan": False
        })

        mock_summ_backend = MagicMock()
        mock_summ_backend.summarizer.backend = "openai.gpt-4o"
        mock_summ_backend.summarizer.temperature = 0.0
        mock_summ_backend.summarizer.max_tokens = 512
        mock_summ_config.return_value = mock_summ_backend

        mock_summ_client = MagicMock()
        mock_summ_client.backend_name = "openai"
        # Provide enough responses for initial run + up to 2 replans
        # (in case evidence collection issues cause multiple replan requests)
        mock_summ_client.generate.side_effect = [
            summarizer_replan,
            summarizer_success,
            summarizer_success  # Extra response in case needed
        ]
        mock_summ_vlm.return_value = mock_summ_client

        mock_summ_load_image.return_value = MagicMock()

        # Run pipeline
        final_state = run_pipeline(
            query="Check quality",
            image_path=temp_image,
            max_replan_iterations=2
        )

        # Verify replanning occurred and eventually succeeded
        assert final_state["summarizer_result"].final_answer == "C"
        assert final_state["summarizer_result"].need_replan is False
        assert final_state["iteration_count"] == 1  # One replan iteration
        # Note: replan_history may contain multiple entries if evidence gaps
        # were detected in multiple iterations before finally succeeding
        assert len(final_state["replan_history"]) >= 1


class TestMaxIterationsEnforcement:
    """Tests for max iteration limit enforcement."""

    def test_max_iterations_prevents_infinite_loop(self, temp_image):
        """Test that max iterations prevents infinite replanning."""
        # This test verifies the logic without full mocking
        # The decide_next_node function should prevent infinite loops

        from src.agentic.graph import decide_next_node
        from src.agentic.state import SummarizerOutput

        # Simulate repeated replan requests
        summarizer_result = SummarizerOutput(
            final_answer="Unable to determine",
            quality_reasoning="Still insufficient",
            need_replan=True,
            replan_reason="Missing evidence"
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image,
            "summarizer_result": summarizer_result,
            "iteration_count": 2,
            "max_replan_iterations": 2
        }

        # Should return END despite need_replan=True
        decision = decide_next_node(state)
        assert decision == "__end__"


class TestDisabledReplanning:
    """Tests for running pipeline with replanning disabled."""

    @patch('src.agentic.nodes.planner.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.planner.load_image')
    @patch('src.agentic.nodes.executor.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.executor.load_image')
    @patch('src.agentic.nodes.executor.ToolRegistry')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_pipeline_with_max_iterations_zero(
        self,
        mock_summ_load_image,
        mock_summ_config,
        mock_summ_vlm,
        mock_tool_registry,
        mock_exec_load_image,
        mock_exec_config,
        mock_exec_vlm,
        mock_plan_load_image,
        mock_plan_config,
        mock_plan_vlm,
        temp_image,
        valid_plan_json,
        executor_json_response
    ):
        """Test that max_replan_iterations=0 disables replanning."""
        # Use multi-object plan to test incomplete evidence
        plan_multi_objects = json.dumps({
            "query_type": "IQA",
            "query_scope": ["car", "person"],  # Multiple objects required
            "distortion_source": "Explicit",
            "distortions": {"car": ["Blurs"], "person": ["Noise"]},
            "reference_mode": "No-Reference",
            "required_tool": None,
            "plan": {
                "distortion_detection": False,
                "distortion_analysis": True,
                "tool_selection": True,
                "tool_execution": True
            }
        })

        # Executor provides evidence only for 'car', missing 'person'
        incomplete_executor_response = json.dumps({
            "distortion_analysis": {
                "car": [{"type": "Blurs", "severity": "moderate", "explanation": "Blur detected"}]
                # Missing 'person'
            },
            "quality_scores": {
                "car": {"Blurs": ["QAlign", 2.8]}
                # Missing 'person'
            }
        })

        # Setup mocks
        mock_plan_backend = MagicMock()
        mock_plan_backend.planner.backend = "openai.gpt-4o"
        mock_plan_backend.planner.temperature = 0.0
        mock_plan_backend.planner.max_tokens = 2048
        mock_plan_config.return_value = mock_plan_backend

        mock_plan_client = MagicMock()
        mock_plan_client.backend_name = "openai"
        mock_plan_client.generate.return_value = plan_multi_objects
        mock_plan_vlm.return_value = mock_plan_client
        mock_plan_load_image.return_value = MagicMock()

        mock_exec_backend = MagicMock()
        mock_exec_backend.executor.backend = "openai.gpt-4o"
        mock_exec_backend.executor.temperature = 0.0
        mock_exec_backend.executor.max_tokens = 4096
        mock_exec_config.return_value = mock_exec_backend

        mock_exec_client = MagicMock()
        mock_exec_client.backend_name = "openai"
        mock_exec_client.generate.return_value = incomplete_executor_response
        mock_exec_vlm.return_value = mock_exec_client
        mock_exec_load_image.return_value = MagicMock()

        # Mock tool registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.execute_tool.return_value = (2.5, 3.0)
        mock_registry_instance.get_tools_for_distortion.return_value = ["QAlign"]
        mock_registry_instance.is_tool_available.return_value = True
        mock_registry_instance.get_cache_stats.return_value = {'hits': 0, 'misses': 1}
        mock_tool_registry.return_value = mock_registry_instance

        # Summarizer requests replan
        summarizer_replan = json.dumps({
            "final_answer": "Unable to determine",
            "quality_reasoning": "Insufficient evidence.",
            "need_replan": True,
            "replan_reason": "Insufficient evidence"
        })

        mock_summ_backend = MagicMock()
        mock_summ_backend.summarizer.backend = "openai.gpt-4o"
        mock_summ_backend.summarizer.temperature = 0.0
        mock_summ_backend.summarizer.max_tokens = 512
        mock_summ_config.return_value = mock_summ_backend

        mock_summ_client = MagicMock()
        mock_summ_client.backend_name = "openai"
        mock_summ_client.generate.return_value = summarizer_replan
        mock_summ_vlm.return_value = mock_summ_client
        mock_summ_load_image.return_value = MagicMock()

        # Run pipeline with max_replan_iterations=0
        final_state = run_pipeline(
            query="Test",
            image_path=temp_image,
            max_replan_iterations=0
        )

        # Should complete without replanning
        assert final_state["iteration_count"] == 0
        assert final_state["summarizer_result"].need_replan is True  # Still set
        # But no actual replanning occurred (iteration_count remains 0)
