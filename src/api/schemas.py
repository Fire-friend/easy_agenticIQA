"""
Pydantic models for FastAPI request/response validation.
"""

from datetime import datetime
from typing import Optional, Union
from pydantic import BaseModel, Field


class AssessPathRequest(BaseModel):
    """Request model for path-based image assessment."""
    query: str = Field(..., min_length=1, description="User's natural language query")
    image_path: str = Field(..., description="Path to the image to assess")
    reference_path: Optional[str] = Field(None, description="Optional path to reference image")
    max_replan_iterations: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum replanning iterations allowed"
    )


class ExecutionMetadata(BaseModel):
    """Metadata about pipeline execution."""
    iteration_count: int = Field(..., description="Number of replanning iterations")
    replan_history: list[str] = Field(default_factory=list, description="History of replan reasons")
    execution_time_seconds: float = Field(..., ge=0.0, description="Total execution time")


class AssessResponse(BaseModel):
    """Response model for image quality assessment."""
    final_answer: Union[str, float] = Field(
        ...,
        description="Final answer: letter for MCQ (e.g., 'C') or float for scoring (e.g., 2.73)"
    )
    quality_score: Optional[float] = Field(
        None,
        ge=1.0,
        le=5.0,
        description="Continuous quality score (1-5) for IQA scoring queries. None for MCQ/explanation."
    )
    quality_reasoning: str = Field(
        ...,
        min_length=1,
        description="Evidence-based explanation for the answer"
    )
    need_replan: bool = Field(
        default=False,
        description="Whether replanning was needed"
    )
    execution_metadata: ExecutionMetadata = Field(
        ...,
        description="Metadata about pipeline execution"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorDetail(BaseModel):
    """Error response detail."""
    error_type: str = Field(..., description="Type of error")
    message: str = Field(..., description="Human-readable error message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")


class ErrorResponse(BaseModel):
    """Standard error response format."""
    detail: ErrorDetail
