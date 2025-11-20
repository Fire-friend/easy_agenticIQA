"""
Planner node implementation for AgenticIQA pipeline.
Analyzes user queries and images to generate structured task plans.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from PIL import Image
from langchain_core.runnables import RunnableConfig

from src.agentic.state import AgenticIQAState, PlannerOutput, PlannerError
from src.agentic.vlm_client import create_vlm_client, load_image
from src.utils.config import load_model_backends

# Configure logging
logger = logging.getLogger(__name__)

# Planner prompt template from docs/02_module_planner.md (Paper Appendix A.2)
PLANNER_PROMPT_TEMPLATE = """System:
You are a planner in an image quality assessment (IQA) system. Your task is to analyze the user's query and
generate a structured plan for downstream assessment.
Return a valid JSON object in the following format:
{{
  "query_type": "IQA" or "Other",
  "query_scope": ["<object1>", "<object2>", ...] or "Global",
  "distortion_source": "Explicit" or "Inferred",
  "distortions": {{"<object_or_Global>": ["<distortion1>", "<distortion2>"]}} or null,
  "reference_mode": "Full-Reference" or "No-Reference",
  "required_tool": null,
  "plan": {{
    "distortion_detection": true or false,
    "distortion_analysis": true or false,
    "tool_selection": true or false,
    "tool_execution": true or false
  }}
}}

FIELD SPECIFICATIONS:
- "distortions": If user explicitly mentions distortions, use format {{"Global": ["Blurs", "Noise"]}} or {{"vehicle": ["Blurs"]}}.
  Valid distortion types: "Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast", "Spatial distortions"
  If distortions need to be inferred, set to null and set "distortion_detection" to true.

IMPORTANT RULES:
1. "required_tool" should ALWAYS be set to null. Tool selection will be handled by a specialized downstream module.
2. Set "tool_selection" to true in the plan to enable automatic tool selection.
3. Do NOT invent tool names or use generic names like "No-Reference IQA Tool" or "BlurDetectionTool".
4. For "distortions" field: MUST be either null or a dict with object names as keys and lists of distortion types as values.

EXAMPLES:
- Query about blur: {{"distortions": {{"Global": ["Blurs"]}}}}
- Query about quality (no specific distortion): {{"distortions": null}}
- Multiple distortions: {{"distortions": {{"Global": ["Blurs", "Noise", "Compression"]}}}}

User:
User's query: {query}
The image: <image>"""


def construct_planner_prompt(query: str, has_reference: bool = False) -> str:
    """
    Construct planner prompt from template.

    Args:
        query: User's natural language query
        has_reference: Whether a reference image is provided

    Returns:
        Formatted prompt string
    """
    prompt = PLANNER_PROMPT_TEMPLATE.format(query=query)

    # Add note about reference image if provided
    if has_reference:
        prompt += "\nNote: A reference image is also provided for Full-Reference quality assessment."

    return prompt


def parse_planner_output(json_str: str) -> PlannerOutput:
    """
    Parse and validate planner JSON output.

    Args:
        json_str: JSON string from VLM (may be wrapped in markdown code blocks)

    Returns:
        Validated PlannerOutput instance

    Raises:
        ValueError: If JSON is invalid or doesn't match schema
    """
    from src.utils.json_utils import parse_json_response

    try:
        # Parse JSON with automatic extraction
        data = parse_json_response(json_str)

        # Validate with Pydantic
        return PlannerOutput.model_validate(data)

    except ValueError as e:
        # Re-raise with more context
        raise ValueError(f"Failed to parse planner output: {e}")
    except Exception as e:
        raise ValueError(f"Failed to validate planner output: {e}")


def planner_node(
    state: AgenticIQAState,
    config: Optional[RunnableConfig] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Planner node function for LangGraph.

    Args:
        state: Current pipeline state
        config: Optional configuration override
        max_retries: Maximum number of retry attempts

    Returns:
        State update dict with 'plan' field or 'error' field

    Raises:
        Exception: If all retries fail
    """
    logger.info("Planner node starting")

    # Extract inputs from state
    query = state["query"]
    image_path = state["image_path"]
    reference_path = state.get("reference_path")

    # Load configuration
    try:
        if config is None:
            backends_config = load_model_backends()
            planner_config = backends_config.planner
        else:
            # Access custom config via RunnableConfig's "configurable" dict
            configurable = config.get("configurable", {})
            if "planner" in configurable:
                planner_config = configurable["planner"]
            else:
                # Fallback to loading from YAML if not in config
                backends_config = load_model_backends()
                planner_config = backends_config.planner

        logger.info(f"Using backend: {planner_config.backend}")

    except Exception as e:
        error_msg = f"Failed to load configuration: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Load images
    try:
        images = [load_image(image_path)]
        if reference_path:
            images.append(load_image(reference_path))
        logger.info(f"Loaded {len(images)} image(s)")

    except Exception as e:
        error_msg = f"Failed to load images: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Create VLM client
    try:
        client = create_vlm_client(
            planner_config.backend,
            {
                "api_key": None,  # Will use environment variables
                "base_url": planner_config.base_url if hasattr(planner_config, 'base_url') else None
            }
        )
        logger.info(f"Created VLM client: {client.backend_name}")

    except Exception as e:
        error_msg = f"Failed to create VLM client: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Construct prompt
    prompt = construct_planner_prompt(query, has_reference=reference_path is not None)

    # Attempt generation with retries
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Planner generation attempt {attempt}/{max_retries}")

            # Generate response
            response = client.generate(
                prompt,
                images,
                temperature=planner_config.temperature,
                max_tokens=planner_config.max_tokens if hasattr(planner_config, 'max_tokens') else 2048,
                top_p=planner_config.top_p if hasattr(planner_config, 'top_p') else 0.1
            )

            logger.debug(f"VLM response: {response[:200]}...")

            # Parse and validate output
            plan = parse_planner_output(response)

            logger.info(f"Planner succeeded: query_type={plan.query_type}, scope={plan.query_scope}")

            # Check if this is a replanning iteration
            # If "plan" already exists in state, this is a replan - increment iteration
            is_replan = "plan" in state and state["plan"] is not None
            current_iteration = state.get("iteration_count", 0)

            if is_replan:
                new_iteration = current_iteration + 1
                logger.info(f"Replanning: iteration {current_iteration} -> {new_iteration}")
                return {"plan": plan, "iteration_count": new_iteration}
            else:
                logger.info("Initial planning")
                return {"plan": plan}

        except ValueError as e:
            # Parsing/validation error - try again with stricter prompt
            last_error = e
            logger.warning(f"Attempt {attempt} failed with validation error: {e}")

            if attempt < max_retries:
                # Add stricter instruction for next attempt
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."

        except Exception as e:
            # Other errors (API errors, etc.)
            last_error = e
            logger.error(f"Attempt {attempt} failed with error: {e}")

            # Don't retry on authentication errors
            if "authentication" in str(e).lower() or "api key" in str(e).lower():
                error_msg = f"Planner authentication failed: {e}"
                logger.error(error_msg)
                return {"error": error_msg}

    # All retries failed
    error_msg = f"Planner failed after {max_retries} attempts. Last error: {last_error}"
    logger.error(error_msg)
    return {"error": error_msg}
