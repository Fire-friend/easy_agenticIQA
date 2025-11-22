"""
Unit tests for Summarizer node in src/agentic/nodes/summarizer.py
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.agentic.nodes.summarizer import (
    format_evidence_unified,
    check_evidence_sufficiency,
    summarizer_node,
    PAPER_UNIFIED_PROMPT_TEMPLATE
)
from src.agentic.state import (
    AgenticIQAState,
    PlannerOutput,
    PlanControlFlags,
    ExecutorOutput,
    DistortionAnalysis,
    SummarizerOutput
)


class TestPromptTemplates:
    """Tests for unified prompt template (paper specification)."""

    def test_unified_template_has_required_placeholders(self):
        """Test that unified template has required placeholders from paper."""
        assert "{query}" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "{distortion_analysis}" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "{tool_scores}" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "{reference_type}" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "{prior_answer}" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "<image>" in PAPER_UNIFIED_PROMPT_TEMPLATE

    def test_unified_template_has_paper_elements(self):
        """Test that unified template matches paper specification."""
        # Check for paper's system message elements
        assert "summarizer assistant" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "Image Quality Assessment" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "quality_reasoning" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "final_answer" in PAPER_UNIFIED_PROMPT_TEMPLATE
        # Check guidelines section
        assert "Guidelines" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "key distortions" in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "Reference tool scores" in PAPER_UNIFIED_PROMPT_TEMPLATE

    def test_unified_template_format(self):
        """Test that unified template can be formatted with all parameters."""
        prompt = PAPER_UNIFIED_PROMPT_TEMPLATE.format(
            query="Is this image blurry?",
            distortion_analysis='{"vehicle": [{"type": "Blurs", "severity": "moderate"}]}',
            tool_scores='{"vehicle": {"Blurs": {"tool": "QAlign", "score": 2.8}}}',
            reference_type="No-Reference",
            prior_answer="This is the first iteration"
        )
        assert "Is this image blurry?" in prompt
        assert "QAlign" in prompt
        assert "No-Reference" in prompt
        assert "first iteration" in prompt


class TestEvidenceFormatting:
    """Tests for unified evidence formatting function."""

    def test_format_evidence_unified_no_evidence(self):
        """Test formatting with no evidence (first iteration, NR mode)."""
        evidence = format_evidence_unified(
            executor_output=None,
            reference_path=None,
            iteration_count=0,
            replan_history=[],
            previous_result=None
        )

        assert evidence["distortion_analysis"] == "No distortion analysis available"
        assert evidence["tool_scores"] == "No tool scores available"
        assert evidence["reference_type"] == "No-Reference"
        assert "first iteration" in evidence["prior_answer"]

    def test_format_evidence_unified_with_data(self):
        """Test formatting with complete evidence."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "vehicle": [
                    DistortionAnalysis(
                        type="Blurs",
                        severity="moderate",
                        explanation="Vehicle edges show motion blur."
                    )
                ]
            },
            quality_scores={
                "vehicle": {
                    "Blurs": ("QAlign", 2.8)
                }
            }
        )

        evidence = format_evidence_unified(
            executor_output=executor_output,
            reference_path=None,
            iteration_count=0,
            replan_history=[],
            previous_result=None
        )

        # Should be valid JSON
        distortion_data = json.loads(evidence["distortion_analysis"])
        tool_data = json.loads(evidence["tool_scores"])

        assert "vehicle" in distortion_data
        assert distortion_data["vehicle"][0]["type"] == "Blurs"
        assert distortion_data["vehicle"][0]["severity"] == "moderate"

        assert "vehicle" in tool_data
        assert tool_data["vehicle"]["Blurs"]["tool"] == "QAlign"
        assert tool_data["vehicle"]["Blurs"]["score"] == 2.8

        assert evidence["reference_type"] == "No-Reference"
        assert "first iteration" in evidence["prior_answer"]

    def test_format_evidence_unified_full_reference(self):
        """Test formatting with full-reference mode."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "Global": [
                    DistortionAnalysis(
                        type="Noise",
                        severity="slight",
                        explanation="Minor noise visible."
                    )
                ]
            },
            quality_scores={
                "Global": {"Noise": ("LPIPS", 3.5)}
            }
        )

        evidence = format_evidence_unified(
            executor_output=executor_output,
            reference_path="/path/to/reference.jpg",
            iteration_count=0,
            replan_history=[],
            previous_result=None
        )

        assert evidence["reference_type"] == "Full-Reference"
        assert "LPIPS" in evidence["tool_scores"]

    def test_format_evidence_unified_with_prior_answer(self):
        """Test formatting with prior answer from replanning iteration."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "Global": [
                    DistortionAnalysis(
                        type="Noise",
                        severity="slight",
                        explanation="Minor noise visible."
                    )
                ]
            },
            quality_scores={
                "Global": {"Noise": ("BRISQUE", 3.5)}
            }
        )

        previous_result = SummarizerOutput(
            final_answer="C",
            quality_score=3.0,
            quality_reasoning="Moderate quality detected",
            need_replan=True,
            replan_reason="Missing analysis for person"
        )

        evidence = format_evidence_unified(
            executor_output=executor_output,
            reference_path=None,
            iteration_count=1,
            replan_history=["[Iteration 0] Missing analysis for person"],
            previous_result=previous_result
        )

        # Should contain prior answer information
        assert "Previous answer" in evidence["prior_answer"]
        assert "C" in evidence["prior_answer"]
        assert "Moderate quality" in evidence["prior_answer"]
        assert "Missing analysis for person" in evidence["prior_answer"]


class TestEvidenceSufficiency:
    """Tests for check_evidence_sufficiency."""

    def test_max_iterations_reached(self):
        """Test that check_evidence_sufficiency reports true state regardless of iterations.

        Note: Iteration limit enforcement is now handled by decide_next_node() in the graph,
        not by check_evidence_sufficiency(). This function only assesses evidence quality.
        """
        executor_output = ExecutorOutput(
            distortion_analysis={},
            quality_scores={}
        )

        need_replan, reason = check_evidence_sufficiency(
            executor_output=executor_output,
            query_scope="Global",
            max_iterations=2,
            current_iteration=2
        )

        # Function reports true evidence state (insufficient - no tool scores)
        # even when iterations are exhausted. The graph enforces iteration limits.
        assert need_replan is True
        assert "No tool scores" in reason

    def test_no_executor_evidence(self):
        """Test that missing executor output triggers replan."""
        need_replan, reason = check_evidence_sufficiency(
            executor_output=None,
            query_scope="Global",
            max_iterations=2,
            current_iteration=0
        )

        assert need_replan is True
        assert "No Executor evidence" in reason

    def test_missing_distortion_analysis_for_scope(self):
        """Test that missing analysis for query scope triggers replan."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "car": [
                    DistortionAnalysis(type="Blurs", severity="moderate", explanation="Blur detected")
                ]
            },
            quality_scores={"car": {"Blurs": ("QAlign", 2.5)}}
        )

        need_replan, reason = check_evidence_sufficiency(
            executor_output=executor_output,
            query_scope=["car", "person"],  # person is missing
            max_iterations=2,
            current_iteration=0
        )

        assert need_replan is True
        assert "person" in reason

    def test_no_tool_scores_triggers_replan(self):
        """Test that missing tool scores triggers replan."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "Global": [
                    DistortionAnalysis(type="Noise", severity="slight", explanation="Noise present")
                ]
            },
            quality_scores=None  # No scores
        )

        need_replan, reason = check_evidence_sufficiency(
            executor_output=executor_output,
            query_scope="Global",
            max_iterations=2,
            current_iteration=0
        )

        assert need_replan is True
        assert "No tool scores" in reason

    def test_sufficient_evidence_no_replan(self):
        """Test that sufficient evidence doesn't trigger replan."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "vehicle": [
                    DistortionAnalysis(type="Blurs", severity="moderate", explanation="Blur detected")
                ]
            },
            quality_scores={
                "vehicle": {"Blurs": ("QAlign", 2.8)}
            }
        )

        need_replan, reason = check_evidence_sufficiency(
            executor_output=executor_output,
            query_scope=["vehicle"],
            max_iterations=2,
            current_iteration=0
        )

        assert need_replan is False
        assert reason == ""

    def test_global_scope_handling(self):
        """Test that Global scope is handled correctly."""
        executor_output = ExecutorOutput(
            distortion_analysis={
                "Global": [
                    DistortionAnalysis(type="Noise", severity="slight", explanation="Noise present")
                ]
            },
            quality_scores={
                "Global": {"Noise": ("BRISQUE", 3.5)}
            }
        )

        need_replan, reason = check_evidence_sufficiency(
            executor_output=executor_output,
            query_scope="Global",
            max_iterations=2,
            current_iteration=0
        )

        assert need_replan is False


class TestSummarizerNode:
    """Integration tests for summarizer_node."""

    @pytest.fixture
    def temp_image(self, tmp_path):
        """Create a temporary test image."""
        from PIL import Image
        image_path = tmp_path / "test_image.jpg"
        img = Image.new('RGB', (100, 100), color='red')
        img.save(image_path)
        return str(image_path)

    @pytest.fixture
    def mock_vlm_client(self):
        """Mock VLM client."""
        client = Mock()
        client.backend_name = "mock_backend"
        client.generate = Mock(return_value='{"final_answer": "C", "quality_reasoning": "Moderate quality"}')
        return client

    @pytest.fixture
    def base_state(self, temp_image):
        """Create base state for tests."""
        plan = PlannerOutput(
            query_type="IQA",
            query_scope="Global",
            distortion_source="Inferred",
            distortions=None,
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=True,
                distortion_analysis=True,
                tool_selection=True,
                tool_execute=True
            )
        )

        executor_output = ExecutorOutput(
            distortion_analysis={
                "Global": [
                    DistortionAnalysis(
                        type="Noise",
                        severity="slight",
                        explanation="Minor noise present"
                    )
                ]
            },
            quality_scores={
                "Global": {"Noise": ("BRISQUE", 3.5)}
            }
        )

        state: AgenticIQAState = {
            "query": "What is the quality?",
            "image_path": temp_image,
            "plan": plan,
            "executor_evidence": executor_output,
            "iteration_count": 0,
            "max_replan_iterations": 2,
            "replan_history": []
        }

        return state

    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_summarizer_node_scoring_mode(
        self,
        mock_load_image,
        mock_create_vlm,
        mock_load_config,
        base_state,
        mock_vlm_client
    ):
        """Test summarizer node in scoring mode."""
        # Setup mocks
        mock_config = Mock()
        mock_config.summarizer = Mock(backend="mock_backend", temperature=0.0, max_tokens=512)
        mock_load_config.return_value = mock_config

        mock_create_vlm.return_value = mock_vlm_client

        mock_image = Mock()
        mock_load_image.return_value = mock_image

        # Run node
        result = summarizer_node(base_state)

        # Verify result
        assert "summarizer_result" in result
        assert result["summarizer_result"].final_answer == "C"
        assert result["summarizer_result"].need_replan is False

    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_summarizer_node_explanation_mode(
        self,
        mock_load_image,
        mock_create_vlm,
        mock_load_config,
        base_state,
        mock_vlm_client,
        temp_image
    ):
        """Test summarizer node in explanation/QA mode."""
        # Change to Other query type for explanation mode
        base_state["plan"].query_type = "Other"

        # Setup mocks
        mock_config = Mock()
        mock_config.summarizer = Mock(backend="mock_backend", temperature=0.0, max_tokens=512)
        mock_load_config.return_value = mock_config

        mock_create_vlm.return_value = mock_vlm_client

        mock_image = Mock()
        mock_load_image.return_value = mock_image

        # Run node
        result = summarizer_node(base_state)

        # Verify result
        assert "summarizer_result" in result
        assert isinstance(result["summarizer_result"], SummarizerOutput)

    def test_summarizer_node_missing_plan(self, temp_image):
        """Test that missing plan returns error."""
        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image
        }

        result = summarizer_node(state)

        assert "error" in result
        assert "No plan" in result["error"]

    def test_summarizer_node_missing_executor_evidence(self, temp_image):
        """Test that missing executor evidence returns error."""
        plan = PlannerOutput(
            query_type="IQA",
            query_scope="Global",
            distortion_source="Inferred",
            distortions=None,
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=True,
                distortion_analysis=True,
                tool_selection=True,
                tool_execute=True
            )
        )

        state: AgenticIQAState = {
            "query": "Test",
            "image_path": temp_image,
            "plan": plan
        }

        result = summarizer_node(state)

        assert "error" in result
        assert "No executor_evidence" in result["error"]

    @patch('src.utils.config.load_model_backends')
    @patch('src.agentic.nodes.summarizer.create_vlm_client')
    @patch('src.agentic.nodes.summarizer.load_image')
    def test_summarizer_node_triggers_replan(
        self,
        mock_load_image,
        mock_create_vlm,
        mock_load_config,
        temp_image,
        mock_vlm_client
    ):
        """Test that insufficient evidence triggers replanning."""
        # Setup state with insufficient evidence
        plan = PlannerOutput(
            query_type="IQA",
            query_scope=["car", "person"],  # Multiple objects
            distortion_source="Inferred",
            distortions=None,
            reference_mode="No-Reference",
            required_tool=None,
            plan=PlanControlFlags(
                distortion_detection=True,
                distortion_analysis=True,
                tool_selection=True,
                tool_execute=True
            )
        )

        # Only has evidence for 'car', missing 'person'
        executor_output = ExecutorOutput(
            distortion_analysis={
                "car": [
                    DistortionAnalysis(
                        type="Blurs",
                        severity="moderate",
                        explanation="Blur detected"
                    )
                ]
            },
            quality_scores={
                "car": {"Blurs": ("QAlign", 2.8)}
            }
        )

        state: AgenticIQAState = {
            "query": "Check quality",
            "image_path": temp_image,
            "plan": plan,
            "executor_evidence": executor_output,
            "iteration_count": 0,
            "max_replan_iterations": 2,
            "replan_history": []
        }

        # Setup mocks
        mock_config = Mock()
        mock_config.summarizer = Mock(backend="mock_backend", temperature=0.0, max_tokens=512)
        mock_load_config.return_value = mock_config

        mock_create_vlm.return_value = mock_vlm_client

        mock_image = Mock()
        mock_load_image.return_value = mock_image

        # Run node
        result = summarizer_node(state)

        # Should trigger replan
        assert result["summarizer_result"].need_replan is True
        assert "person" in result["summarizer_result"].replan_reason
        # Note: iteration_count is now managed by planner, not summarizer
        assert len(result["replan_history"]) == 1
