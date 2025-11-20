"""
Pipeline wrapper for FastAPI integration.
Converts API requests to pipeline inputs and state to API responses.
"""

import time
import logging
from typing import Optional, Dict, Any

from fastapi import HTTPException

from src.agentic.graph import run_pipeline
from src.agentic.state import AgenticIQAState
from src.api.schemas import AssessResponse, ExecutionMetadata

logger = logging.getLogger(__name__)


class PipelineWrapper:
    """Wrapper for AgenticIQA pipeline execution."""

    def __init__(self, pipeline_config: Optional[Dict[str, Any]] = None):
        """
        Initialize pipeline wrapper.

        Args:
            pipeline_config: Optional configuration for pipeline execution
        """
        # Load model backends configuration
        if pipeline_config is None:
            try:
                from src.utils.config import load_model_backends
                backends_config = load_model_backends()
                self.pipeline_config = {
                    "model_backends": backends_config.model_dump()
                }
                logger.info("Loaded model backends configuration")
            except Exception as e:
                logger.warning(f"Failed to load model backends config: {e}")
                self.pipeline_config = {}
        else:
            self.pipeline_config = pipeline_config

        logger.info("PipelineWrapper initialized")

    async def assess_image(
        self,
        query: str,
        image_path: str,
        reference_path: Optional[str] = None,
        max_replan_iterations: int = 2
    ) -> AssessResponse:
        """
        Execute pipeline and convert result to API response.

        Args:
            query: User's natural language query
            image_path: Path to image to assess
            reference_path: Optional path to reference image
            max_replan_iterations: Maximum replanning iterations

        Returns:
            AssessResponse with final answer and metadata

        Raises:
            HTTPException: If pipeline execution fails
        """
        start_time = time.time()

        try:
            logger.info(f"Running pipeline: query='{query}', image='{image_path}'")

            # Execute pipeline
            final_state: AgenticIQAState = run_pipeline(
                query=query,
                image_path=image_path,
                reference_path=reference_path,
                max_replan_iterations=max_replan_iterations,
                config=self.pipeline_config
            )

            # Calculate execution time
            execution_time = time.time() - start_time

            # Convert state to API response
            response = self._state_to_response(final_state, execution_time)

            logger.info(f"Pipeline completed in {execution_time:.2f}s")
            return response

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise

        except ValueError as e:
            # Handle validation errors from pipeline
            logger.error(f"Pipeline validation error: {e}")
            raise HTTPException(status_code=400, detail={
                "error_type": "validation_error",
                "message": str(e)
            })

        except FileNotFoundError as e:
            # Handle missing files
            logger.error(f"File not found: {e}")
            raise HTTPException(status_code=400, detail={
                "error_type": "file_not_found",
                "message": str(e)
            })

        except Exception as e:
            # Handle other pipeline errors
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)

            # Extract more detailed error info if available
            error_detail = str(e)
            if hasattr(e, '__cause__') and e.__cause__:
                error_detail += f" (caused by: {e.__cause__})"

            raise HTTPException(status_code=500, detail={
                "error_type": "pipeline_error",
                "message": f"Pipeline execution failed: {error_detail}"
            })

    def _state_to_response(
        self,
        state: AgenticIQAState,
        execution_time: float
    ) -> AssessResponse:
        """
        Convert AgenticIQAState to AssessResponse.

        Args:
            state: Final pipeline state
            execution_time: Total execution time in seconds

        Returns:
            AssessResponse model

        Raises:
            HTTPException: If state is invalid or missing required fields
        """
        try:
            # Extract summarizer result
            summarizer_result = state.get("summarizer_result")
            if not summarizer_result:
                # Check if there's an error in state
                error_msg = state.get("error", "Unknown error")
                logger.error(f"Pipeline state: {list(state.keys())}")
                raise HTTPException(status_code=500, detail={
                    "error_type": "missing_result",
                    "message": f"Pipeline completed but no summarizer result available. Error: {error_msg}"
                })

            # Extract metadata
            iteration_count = state.get("iteration_count", 0)
            replan_history = state.get("replan_history", [])

            # Build execution metadata
            metadata = ExecutionMetadata(
                iteration_count=iteration_count,
                replan_history=replan_history,
                execution_time_seconds=round(execution_time, 2)
            )

            # Build response
            response = AssessResponse(
                final_answer=summarizer_result.final_answer,
                quality_score=summarizer_result.quality_score,
                quality_reasoning=summarizer_result.quality_reasoning,
                need_replan=summarizer_result.need_replan,
                execution_metadata=metadata
            )

            return response

        except AttributeError as e:
            logger.error(f"Invalid state structure: {e}")
            raise HTTPException(status_code=500, detail={
                "error_type": "invalid_state",
                "message": f"Invalid pipeline state structure: {str(e)}"
            })

        except Exception as e:
            logger.error(f"Failed to convert state to response: {e}")
            raise HTTPException(status_code=500, detail={
                "error_type": "conversion_error",
                "message": f"Failed to convert pipeline result: {str(e)}"
            })
