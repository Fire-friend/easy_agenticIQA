"""
Pydantic models for LangGraph state management in the AgenticIQA pipeline.
Defines type-safe state models for Planner, Executor, and Summarizer modules.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from typing_extensions import NotRequired, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


# ==================== Planner Models ====================

class PlanControlFlags(BaseModel):
    """Control flags for Executor subtasks."""
    distortion_detection: bool = Field(
        ...,
        description="Whether to detect distortions in the image"
    )
    distortion_analysis: bool = Field(
        ...,
        description="Whether to analyze distortion severity and impact"
    )
    tool_selection: bool = Field(
        ...,
        description="Whether to select appropriate IQA tools"
    )
    tool_execution: bool = Field(
        ...,
        description="Whether to execute selected IQA tools"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "distortion_detection": False,
                    "distortion_analysis": True,
                    "tool_selection": False,
                    "tool_execution": False
                }
            ]
        }
    }


class PlannerOutput(BaseModel):
    """Structured output from the Planner module."""
    query_type: Literal["IQA", "Other"] = Field(
        ...,
        description="Type of query: IQA (Image Quality Assessment) or Other"
    )
    query_scope: Union[List[str], Literal["Global"]] = Field(
        ...,
        description="Scope of the query: list of specific objects or 'Global'"
    )
    distortion_source: Literal["Explicit", "Inferred"] = Field(
        ...,
        description="Whether distortions are explicitly mentioned or need to be inferred"
    )
    distortions: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Map of objects/Global to their distortion types"
    )
    reference_mode: Literal["Full-Reference", "No-Reference"] = Field(
        ...,
        description="Whether a reference image is provided"
    )
    required_tool: Optional[str] = Field(
        None,
        description="Specific tool required if user explicitly requested one"
    )
    plan: PlanControlFlags = Field(
        ...,
        description="Control flags for Executor subtasks"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query_type": "IQA",
                    "query_scope": ["vehicle"],
                    "distortion_source": "Explicit",
                    "distortions": {"vehicle": ["Blurs"]},
                    "reference_mode": "No-Reference",
                    "required_tool": None,
                    "plan": {
                        "distortion_detection": False,
                        "distortion_analysis": True,
                        "tool_selection": False,
                        "tool_execution": False
                    }
                }
            ]
        }
    }


class PlannerInput(BaseModel):
    """Input data for the Planner module."""
    query: str = Field(..., min_length=1, description="User's natural language query")
    image_path: str = Field(..., description="Path to the image to assess")
    reference_path: Optional[str] = Field(None, description="Optional path to reference image")
    prior_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional context from previous conversation turns"
    )

    @field_validator('image_path')
    @classmethod
    def validate_image_path(cls, v: str) -> str:
        """Validate that image file exists."""
        path = Path(v)
        if not path.exists():
            raise ValueError(f"Image file not found: {v}")
        if not path.is_file():
            raise ValueError(f"Image path is not a file: {v}")

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Invalid image format: {path.suffix}. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )
        return v

    @field_validator('reference_path')
    @classmethod
    def validate_reference_path(cls, v: Optional[str]) -> Optional[str]:
        """Validate that reference image file exists if provided."""
        if v is None:
            return v

        path = Path(v)
        if not path.exists():
            raise ValueError(f"Reference image file not found: {v}")
        if not path.is_file():
            raise ValueError(f"Reference image path is not a file: {v}")

        # Check file extension
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        if path.suffix.lower() not in valid_extensions:
            raise ValueError(
                f"Invalid reference image format: {path.suffix}. "
                f"Supported formats: {', '.join(valid_extensions)}"
            )
        return v

    @field_validator('query')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate that query is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace")
        return v.strip()


class PlannerError(BaseModel):
    """Error information from Planner execution."""
    error_type: str = Field(..., description="Type of error (e.g., 'validation_error', 'api_error')")
    message: str = Field(..., description="Human-readable error description")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error context")
    retry_count: int = Field(default=0, description="Number of retries attempted")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

    @classmethod
    def from_exception(cls, exc: Exception, error_type: str = "unknown_error", retry_count: int = 0) -> "PlannerError":
        """Create PlannerError from an exception."""
        return cls(
            error_type=error_type,
            message=str(exc),
            details={
                "exception_type": type(exc).__name__,
                "exception_args": exc.args
            },
            retry_count=retry_count,
            timestamp=datetime.now()
        )


# ==================== Executor Models ====================

class DistortionAnalysis(BaseModel):
    """Distortion severity analysis result."""
    type: str = Field(..., description="Distortion type (e.g., 'Blurs', 'Noise')")
    severity: Literal["none", "slight", "moderate", "severe", "extreme"] = Field(
        ...,
        description="Severity level of the distortion"
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Brief visual explanation of the distortion impact"
    )

    @field_validator('explanation')
    @classmethod
    def validate_explanation(cls, v: str) -> str:
        """Validate that explanation is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Explanation cannot be empty or whitespace")
        return v.strip()

    @field_validator('type')
    @classmethod
    def validate_distortion_type(cls, v: str) -> str:
        """Validate that distortion type is from valid categories."""
        valid_types = {
            "Blurs", "Color distortions", "Compression", "Noise",
            "Brightness change", "Sharpness", "Contrast", "Spatial distortions"
        }
        if v not in valid_types:
            # Log warning but don't fail - VLM might use slight variations
            import logging
            logging.getLogger(__name__).warning(
                f"Distortion type '{v}' not in standard categories: {valid_types}"
            )
        return v


class ToolExecutionLog(BaseModel):
    """Record of a single IQA tool execution."""
    tool_name: str = Field(..., description="IQA tool identifier")
    object_name: str = Field(..., description="Query scope object or 'Global'")
    distortion: str = Field(..., description="Distortion type being assessed")
    raw_score: Optional[float] = Field(None, description="Unnormalized tool output")
    normalized_score: Optional[float] = Field(None, description="Normalized score in [1, 5] range")
    execution_time: float = Field(default=0.0, ge=0.0, description="Tool runtime in seconds")
    fallback: bool = Field(default=False, description="Whether tool was a fallback")
    error: Optional[str] = Field(None, description="Error message if tool failed")
    timestamp: datetime = Field(default_factory=datetime.now, description="Execution timestamp")

    @field_validator('normalized_score')
    @classmethod
    def validate_normalized_score(cls, v: Optional[float]) -> Optional[float]:
        """Validate that normalized score is in [1, 5] range."""
        if v is None:
            return None
        if not (1.0 <= v <= 5.0):
            import logging
            logging.getLogger(__name__).warning(
                f"Normalized score {v} outside [1, 5] range, clipping"
            )
            return max(1.0, min(5.0, v))
        return v


class ExecutorOutput(BaseModel):
    """Structured output from the Executor module."""
    distortion_set: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Detected distortions per object/Global"
    )
    distortion_analysis: Optional[Dict[str, List[DistortionAnalysis]]] = Field(
        None,
        description="Severity analysis per object/Global"
    )
    selected_tools: Optional[Dict[str, Dict[str, str]]] = Field(
        None,
        description="Selected tools per object → distortion → tool_name"
    )
    quality_scores: Optional[Dict[str, Dict[str, tuple]]] = Field(
        None,
        description="Quality scores per object → distortion → (tool_name, score)"
    )
    tool_logs: List[ToolExecutionLog] = Field(
        default_factory=list,
        description="Execution logs for all tool runs"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "distortion_set": None,
                    "distortion_analysis": {
                        "vehicle": [
                            {
                                "type": "Blurs",
                                "severity": "severe",
                                "explanation": "The vehicle edges appear smeared with strong motion blur."
                            }
                        ]
                    },
                    "selected_tools": {
                        "vehicle": {
                            "Blurs": "QAlign"
                        }
                    },
                    "quality_scores": {
                        "vehicle": {
                            "Blurs": ("QAlign", 2.15)
                        }
                    },
                    "tool_logs": []
                }
            ]
        }
    }


# ==================== Summarizer Models ====================

class SummarizerOutput(BaseModel):
    """Structured output from the Summarizer module."""
    final_answer: Union[str, float] = Field(
        ...,
        description="Final answer: letter for MCQ (e.g., 'C'), float for scoring (e.g., 2.73)"
    )
    quality_score: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Continuous quality score (1-5) for IQA scoring queries. None for MCQ/explanation queries."
    )
    quality_reasoning: str = Field(
        ...,
        min_length=1,
        description="Evidence-based explanation for the answer"
    )
    need_replan: bool = Field(
        default=False,
        description="Whether replanning is needed due to insufficient evidence"
    )
    replan_reason: Optional[str] = Field(
        None,
        description="Reason for replanning if need_replan=True"
    )
    used_evidence: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional tracking of evidence items referenced"
    )

    @field_validator('final_answer')
    @classmethod
    def validate_final_answer(cls, v: Union[str, float]) -> Union[str, float]:
        """Validate final_answer based on type (str or float)."""
        if isinstance(v, str):
            v = v.strip()
            if not v:
                raise ValueError("final_answer cannot be empty when string")
            return v
        elif isinstance(v, float):
            if not (1.0 <= v <= 5.0):
                raise ValueError(f"final_answer must be in range [1.0, 5.0], got {v}")
            if not (v == v):  # Check for NaN
                raise ValueError("final_answer cannot be NaN")
            return v
        else:
            raise TypeError("final_answer must be str or float")

    @field_validator('quality_reasoning')
    @classmethod
    def validate_quality_reasoning(cls, v: str) -> str:
        """Validate that quality_reasoning is not empty or whitespace."""
        v = v.strip()
        if not v:
            raise ValueError("quality_reasoning cannot be empty or whitespace")
        return v

    @field_validator('replan_reason')
    @classmethod
    def validate_replan_reason(cls, v: Optional[str], info) -> Optional[str]:
        """Auto-set replan_reason if need_replan=True but reason is missing."""
        # Access need_replan from the model being validated
        if info.data.get('need_replan') and not v:
            import logging
            logging.getLogger(__name__).warning(
                "need_replan=True but replan_reason is missing, auto-setting"
            )
            return "No reason provided"
        return v.strip() if v else v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "final_answer": 2.73,
                    "quality_score": 2.73,
                    "quality_reasoning": "The image shows moderate blur (severity: moderate, tool score: 2.6) affecting sharpness. Tool-augmented fusion yields score of 2.73, indicating fair quality.",
                    "need_replan": False
                },
                {
                    "final_answer": "C",
                    "quality_score": None,
                    "quality_reasoning": "The image shows moderate blur affecting sharpness, consistent with tool score of 2.6.",
                    "need_replan": False
                },
                {
                    "final_answer": "Unable to determine",
                    "quality_score": None,
                    "quality_reasoning": "Insufficient evidence for vehicle region.",
                    "need_replan": True,
                    "replan_reason": "Missing tool scores for vehicle region"
                }
            ]
        }
    }


# ==================== LangGraph State ====================

class AgenticIQAState(TypedDict):
    """
    LangGraph state for the AgenticIQA pipeline.

    Phase 2 fields (Planner):
    - query: User's natural language query
    - image_path: Path to image to assess
    - reference_path: Optional path to reference image
    - plan: Structured plan from Planner
    - error: Error message if any step fails

    Phase 3 fields (Executor):
    - executor_evidence: Structured evidence from Executor

    Phase 4 fields (Summarizer):
    - summarizer_result: Final answer and reasoning from Summarizer
    - iteration_count: Current replanning iteration (starts at 0)
    - max_replan_iterations: Maximum allowed replanning iterations (default 2)
    - replan_history: History of replan reasons for debugging
    """
    query: str
    image_path: str
    reference_path: NotRequired[str]
    plan: NotRequired[PlannerOutput]
    error: NotRequired[str]

    # Phase 3: Executor outputs
    executor_evidence: NotRequired[ExecutorOutput]

    # Phase 4: Summarizer outputs and iteration tracking
    summarizer_result: NotRequired[SummarizerOutput]
    iteration_count: NotRequired[int]
    max_replan_iterations: NotRequired[int]
    replan_history: NotRequired[List[str]]


# ==================== Utility Functions ====================

def merge_plan_state(current: AgenticIQAState, update: Dict[str, Any]) -> AgenticIQAState:
    """
    State reducer for merging plan updates.

    Args:
        current: Current state
        update: Partial state update

    Returns:
        Merged state
    """
    # For Phase 2, we use simple replacement strategy
    # The 'plan' field is completely replaced, not merged
    merged = current.copy()
    merged.update(update)
    return merged
