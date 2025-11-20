"""Tests for Executor state models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.agentic.state import (
    DistortionAnalysis,
    ToolExecutionLog,
    ExecutorOutput,
    AgenticIQAState
)


class TestDistortionAnalysis:
    """Tests for DistortionAnalysis model."""

    def test_valid_distortion_analysis(self):
        """Test creating valid DistortionAnalysis."""
        analysis = DistortionAnalysis(
            type="Blurs",
            severity="moderate",
            explanation="The image shows noticeable motion blur"
        )
        assert analysis.type == "Blurs"
        assert analysis.severity == "moderate"
        assert analysis.explanation == "The image shows noticeable motion blur"

    def test_all_severity_levels(self):
        """Test all valid severity levels."""
        severity_levels = ["none", "slight", "moderate", "severe", "extreme"]
        for severity in severity_levels:
            analysis = DistortionAnalysis(
                type="Noise",
                severity=severity,
                explanation="Test explanation"
            )
            assert analysis.severity == severity

    def test_invalid_severity(self):
        """Test that invalid severity raises error."""
        with pytest.raises(ValidationError):
            DistortionAnalysis(
                type="Blurs",
                severity="very_bad",  # Invalid
                explanation="Test"
            )

    def test_empty_explanation(self):
        """Test that empty explanation raises error."""
        with pytest.raises(ValidationError):
            DistortionAnalysis(
                type="Blurs",
                severity="moderate",
                explanation=""
            )

    def test_whitespace_explanation(self):
        """Test that whitespace-only explanation raises error."""
        with pytest.raises(ValidationError):
            DistortionAnalysis(
                type="Blurs",
                severity="moderate",
                explanation="   "
            )

    def test_explanation_trimming(self):
        """Test that explanation is trimmed."""
        analysis = DistortionAnalysis(
            type="Blurs",
            severity="moderate",
            explanation="  Test explanation  "
        )
        assert analysis.explanation == "Test explanation"

    def test_json_serialization(self):
        """Test JSON serialization."""
        analysis = DistortionAnalysis(
            type="Blurs",
            severity="moderate",
            explanation="Test"
        )
        json_str = analysis.model_dump_json()
        assert '"type":"Blurs"' in json_str or '"type": "Blurs"' in json_str

    def test_json_deserialization(self):
        """Test JSON deserialization."""
        json_data = '{"type": "Noise", "severity": "severe", "explanation": "Heavy noise"}'
        analysis = DistortionAnalysis.model_validate_json(json_data)
        assert analysis.type == "Noise"
        assert analysis.severity == "severe"


class TestToolExecutionLog:
    """Tests for ToolExecutionLog model."""

    def test_valid_tool_log(self):
        """Test creating valid ToolExecutionLog."""
        log = ToolExecutionLog(
            tool_name="QAlign",
            object_name="vehicle",
            distortion="Blurs",
            raw_score=2.5,
            normalized_score=3.2,
            execution_time=1.5
        )
        assert log.tool_name == "QAlign"
        assert log.normalized_score == 3.2
        assert log.fallback is False

    def test_normalized_score_clipping(self):
        """Test that normalized score outside [1, 5] is clipped."""
        log = ToolExecutionLog(
            tool_name="QAlign",
            object_name="Global",
            distortion="Noise",
            raw_score=10.0,
            normalized_score=6.5,  # Outside range
            execution_time=1.0
        )
        # Should be clipped to 5.0
        assert 1.0 <= log.normalized_score <= 5.0

    def test_negative_execution_time(self):
        """Test that negative execution time raises error."""
        with pytest.raises(ValidationError):
            ToolExecutionLog(
                tool_name="QAlign",
                object_name="Global",
                distortion="Blurs",
                raw_score=2.0,
                normalized_score=3.0,
                execution_time=-1.0  # Invalid
            )

    def test_with_error(self):
        """Test log with error message."""
        log = ToolExecutionLog(
            tool_name="TOPIQ_FR",
            object_name="Global",
            distortion="Blurs",
            raw_score=0.0,
            normalized_score=1.0,
            error="Tool execution failed"
        )
        assert log.error == "Tool execution failed"

    def test_with_fallback(self):
        """Test log with fallback flag."""
        log = ToolExecutionLog(
            tool_name="BRISQUE",
            object_name="Global",
            distortion="Noise",
            raw_score=45.0,
            normalized_score=2.8,
            fallback=True
        )
        assert log.fallback is True

    def test_timestamp_auto_generated(self):
        """Test that timestamp is auto-generated."""
        log = ToolExecutionLog(
            tool_name="QAlign",
            object_name="Global",
            distortion="Blurs",
            raw_score=2.0,
            normalized_score=3.0
        )
        assert isinstance(log.timestamp, datetime)


class TestExecutorOutput:
    """Tests for ExecutorOutput model."""

    def test_empty_executor_output(self):
        """Test creating empty ExecutorOutput."""
        output = ExecutorOutput()
        assert output.distortion_set is None
        assert output.distortion_analysis is None
        assert output.selected_tools is None
        assert output.quality_scores is None
        assert len(output.tool_logs) == 0

    def test_with_distortion_set(self):
        """Test ExecutorOutput with distortion_set."""
        output = ExecutorOutput(
            distortion_set={"vehicle": ["Blurs", "Noise"]}
        )
        assert output.distortion_set == {"vehicle": ["Blurs", "Noise"]}

    def test_with_distortion_analysis(self):
        """Test ExecutorOutput with distortion_analysis."""
        analysis = DistortionAnalysis(
            type="Blurs",
            severity="severe",
            explanation="Strong motion blur"
        )
        output = ExecutorOutput(
            distortion_analysis={"vehicle": [analysis]}
        )
        assert len(output.distortion_analysis["vehicle"]) == 1
        assert output.distortion_analysis["vehicle"][0].type == "Blurs"

    def test_with_selected_tools(self):
        """Test ExecutorOutput with selected_tools."""
        output = ExecutorOutput(
            selected_tools={
                "vehicle": {"Blurs": "QAlign"}
            }
        )
        assert output.selected_tools["vehicle"]["Blurs"] == "QAlign"

    def test_with_quality_scores(self):
        """Test ExecutorOutput with quality_scores."""
        output = ExecutorOutput(
            quality_scores={
                "vehicle": {"Blurs": ("QAlign", 2.15)}
            }
        )
        tool_name, score = output.quality_scores["vehicle"]["Blurs"]
        assert tool_name == "QAlign"
        assert score == 2.15

    def test_with_tool_logs(self):
        """Test ExecutorOutput with tool_logs."""
        log = ToolExecutionLog(
            tool_name="QAlign",
            object_name="vehicle",
            distortion="Blurs",
            raw_score=2.0,
            normalized_score=2.15
        )
        output = ExecutorOutput(tool_logs=[log])
        assert len(output.tool_logs) == 1
        assert output.tool_logs[0].tool_name == "QAlign"

    def test_complete_output(self):
        """Test complete ExecutorOutput with all fields."""
        analysis = DistortionAnalysis(
            type="Blurs",
            severity="severe",
            explanation="Strong blur"
        )
        log = ToolExecutionLog(
            tool_name="QAlign",
            object_name="vehicle",
            distortion="Blurs",
            raw_score=2.0,
            normalized_score=2.15
        )

        output = ExecutorOutput(
            distortion_set={"vehicle": ["Blurs"]},
            distortion_analysis={"vehicle": [analysis]},
            selected_tools={"vehicle": {"Blurs": "QAlign"}},
            quality_scores={"vehicle": {"Blurs": ("QAlign", 2.15)}},
            tool_logs=[log]
        )

        assert output.distortion_set is not None
        assert output.distortion_analysis is not None
        assert output.selected_tools is not None
        assert output.quality_scores is not None
        assert len(output.tool_logs) == 1

    def test_json_serialization_round_trip(self):
        """Test JSON serialization and deserialization."""
        analysis = DistortionAnalysis(
            type="Noise",
            severity="moderate",
            explanation="Visible noise"
        )
        output = ExecutorOutput(
            distortion_set={"Global": ["Noise"]},
            distortion_analysis={"Global": [analysis]}
        )

        # Serialize
        json_str = output.model_dump_json()

        # Deserialize
        output2 = ExecutorOutput.model_validate_json(json_str)

        assert output2.distortion_set == output.distortion_set
        assert output2.distortion_analysis["Global"][0].type == "Noise"


class TestAgenticIQAStateWithExecutor:
    """Tests for AgenticIQAState with executor_evidence field."""

    def test_state_with_executor_evidence(self):
        """Test AgenticIQAState with executor_evidence."""
        state: AgenticIQAState = {
            "query": "What's the quality?",
            "image_path": "/path/to/image.jpg",
            "executor_evidence": ExecutorOutput(
                distortion_set={"Global": ["Blurs"]}
            )
        }

        assert "executor_evidence" in state
        assert state["executor_evidence"].distortion_set == {"Global": ["Blurs"]}

    def test_state_without_executor_evidence(self):
        """Test AgenticIQAState without executor_evidence (Phase 2 compatibility)."""
        state: AgenticIQAState = {
            "query": "What's the quality?",
            "image_path": "/path/to/image.jpg"
        }

        assert "executor_evidence" not in state
        # This should not raise any errors
