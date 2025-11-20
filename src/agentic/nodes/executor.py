"""
Executor node implementation for AgenticIQA pipeline.
Collects distortion evidence and tool scores based on Planner's control flags.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from langchain_core.runnables import RunnableConfig
from PIL import Image

from src.agentic.state import AgenticIQAState, ExecutorOutput, DistortionAnalysis, ToolExecutionLog
from src.agentic.vlm_client import create_vlm_client, load_image
from src.agentic.tool_registry import ToolRegistry, ToolExecutionError
from src.utils.config import load_model_backends
from src.utils.json_utils import parse_json_response

# Configure logging
logger = logging.getLogger(__name__)

# Valid distortion categories
VALID_DISTORTION_TYPES = {
    "Blurs", "Color distortions", "Compression", "Noise",
    "Brightness change", "Sharpness", "Contrast", "Spatial distortions"
}

# ==================== Prompt Templates ====================

DISTORTION_DETECTION_PROMPT_TEMPLATE = """System:
You are an expert in distortion detection. Based on the user's query, identify all possible distortions need to be
focused on to properly address the user's intent.
Return a valid JSON object in the following format:
{{
  "distortion_set": {{
    "<object or Global>": ["<distortion_1>", "<distortion_2>", ...]
  }}
}}
Instructions:
1. Focus your analysis on query scope. Describe distortions for each individually.
2. Only include distortion types from the following valid categories: ["Blurs", "Color distortions", "Compression", "Noise", "Brightness change", "Sharpness", "Contrast"]

User:
User's query: {query}
The image: <image>"""

DISTORTION_ANALYSIS_PROMPT_TEMPLATE = """System:
You are a distortion analysis expert. Your task is to assess the severity and visual impact of various distortion
types for different regions of an image or the entire image.
The distortion information: {distortion_set}
Return a valid JSON object in the following format:
{{
  "distortion_analysis": {{
    "<object or Global>": [
      {{
        "type": "<distortion>",
        "severity": "<none/slight/moderate/severe/extreme>",
        "explanation": "<brief visual explanation>"
      }}
    ]
  }}
}}
Instructions:
1. Base your analysis on the listed distortion types and consider the user question.
2. Use "none" if a distortion is barely or not visible.
3. Keep explanations short and focused on visual quality. Focus solely on analyzing visual distortion effects.

User:
User's query: {query}
The image: <image>"""

TOOL_SELECTION_PROMPT_TEMPLATE = """System:
You are a tool executor. Your task is to assign the most appropriate IQA tool to each visual distortion type,
based on the descriptions of the tools.
The distortion information: {distortion_set}.
The available tools: {tool_description}.
Return a valid JSON object in the following format:
{{
  "selected_tools": {{
    "<object or Global>": {{
      "<distortion>": "<tool_name>"
    }}
  }}
}}
Instructions:
For each distortion, choose the tool whose description suggests it performs best for that type of distortion."""


# ==================== Subtask Functions ====================

def distortion_detection_subtask(
    query: str,
    images: List[Image.Image],
    vlm_client,
    max_retries: int = 3
) -> Optional[Dict[str, List[str]]]:
    """
    Detect distortions in the image using VLM.

    Args:
        query: User's query
        images: List of images (test + optional reference)
        vlm_client: VLM client instance
        max_retries: Maximum retry attempts

    Returns:
        Distortion set dict or None if detection fails
    """
    prompt = DISTORTION_DETECTION_PROMPT_TEMPLATE.format(query=query)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Distortion detection attempt {attempt}/{max_retries}")

            # Generate VLM response
            response = vlm_client.generate(prompt, images, temperature=0.0, max_tokens=2048)
            logger.debug(f"Detection response: {response[:200]}...")

            # Parse JSON with automatic extraction
            data = parse_json_response(response)
            distortion_set = data.get('distortion_set')

            if distortion_set is None:
                raise ValueError("Missing 'distortion_set' in response")

            # Validate distortion types
            for obj, distortions in distortion_set.items():
                for distortion in distortions:
                    if distortion not in VALID_DISTORTION_TYPES:
                        logger.warning(f"Invalid distortion type: {distortion}")

            logger.info(f"Detected distortions: {distortion_set}")
            return distortion_set

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."
        except Exception as e:
            logger.error(f"Distortion detection error: {e}")
            break

    logger.error("Distortion detection failed after all retries")
    return None


def distortion_analysis_subtask(
    query: str,
    images: List[Image.Image],
    distortion_set: Dict[str, List[str]],
    vlm_client,
    max_retries: int = 3
) -> Optional[Dict[str, List[DistortionAnalysis]]]:
    """
    Analyze distortion severity using VLM.

    Args:
        query: User's query
        images: List of images
        distortion_set: Detected distortions
        vlm_client: VLM client instance
        max_retries: Maximum retry attempts

    Returns:
        Distortion analysis dict or None if analysis fails
    """
    prompt = DISTORTION_ANALYSIS_PROMPT_TEMPLATE.format(
        query=query,
        distortion_set=json.dumps(distortion_set)
    )

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Distortion analysis attempt {attempt}/{max_retries}")

            # Generate VLM response
            response = vlm_client.generate(prompt, images, temperature=0.0, max_tokens=2048)
            logger.debug(f"Analysis response: {response[:200]}...")

            # Parse JSON with automatic extraction
            data = parse_json_response(response)
            analysis_data = data.get('distortion_analysis')

            if analysis_data is None:
                raise ValueError("Missing 'distortion_analysis' in response")

            # Validate with Pydantic
            result = {}
            for obj, analyses in analysis_data.items():
                result[obj] = [DistortionAnalysis(**item) for item in analyses]

            logger.info(f"Analysis complete for {len(result)} objects")
            return result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."
        except Exception as e:
            logger.error(f"Distortion analysis error: {e}")
            break

    logger.error("Distortion analysis failed after all retries")
    return None


def tool_selection_subtask(
    query: str,
    images: List[Image.Image],
    distortion_set: Dict[str, List[str]],
    tool_registry: ToolRegistry,
    reference_available: bool,
    vlm_client,
    max_retries: int = 3
) -> Optional[Dict[str, Dict[str, str]]]:
    """
    Select appropriate IQA tools for each distortion using VLM.

    Args:
        query: User's query
        images: List of images
        distortion_set: Detected distortions
        tool_registry: Tool registry instance
        reference_available: Whether reference image is available
        vlm_client: VLM client instance
        max_retries: Maximum retry attempts

    Returns:
        Selected tools dict or None if selection fails
    """
    # Format tool descriptions for VLM
    tool_descriptions = {}
    for tool_name, metadata in tool_registry.tools.items():
        tool_descriptions[tool_name] = {
            "type": metadata['type'],
            "strengths": metadata.get('strengths', [])
        }

    prompt = TOOL_SELECTION_PROMPT_TEMPLATE.format(
        distortion_set=json.dumps(distortion_set),
        tool_description=json.dumps(tool_descriptions, indent=2)
    )

    # Add FR/NR guidance
    if reference_available:
        prompt += "\n\nIMPORTANT: A reference image is available. Prefer Full-Reference (FR) tools when possible."
    else:
        prompt += "\n\nIMPORTANT: NO reference image is available. You MUST select ONLY No-Reference (NR) tools. Do NOT select Full-Reference (FR) tools."

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Tool selection attempt {attempt}/{max_retries}")

            # Generate VLM response
            response = vlm_client.generate(prompt, images, temperature=0.0, max_tokens=2048)
            logger.debug(f"Selection response: {response[:200]}...")

            # Parse JSON with automatic extraction
            data = parse_json_response(response)
            selected_tools = data.get('selected_tools')

            if selected_tools is None:
                raise ValueError("Missing 'selected_tools' in response")

            # Validate tool names exist
            for obj, distortions in selected_tools.items():
                for distortion, tool_name in distortions.items():
                    if not tool_registry.is_tool_available(tool_name):
                        error_msg = f"Tool selection error: Unknown tool '{tool_name}' selected for {obj}/{distortion}. Available tools: {list(tool_registry.tools.keys())}"
                        logger.error(error_msg)
                        raise ValueError(error_msg)

            logger.info(f"Selected tools for {len(selected_tools)} objects")
            return selected_tools

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                prompt += "\n\nIMPORTANT: Return ONLY valid JSON with no additional text."
        except Exception as e:
            logger.error(f"Tool selection error: {e}")
            break

    # Fallback: use default tools
    logger.error("Tool selection failed after all retries")
    # Return None to indicate failure - don't use automatic fallback
    return None


def tool_execution_subtask(
    selected_tools: Dict[str, Dict[str, str]],
    image_path: str,
    reference_path: Optional[str],
    tool_registry: ToolRegistry
) -> Tuple[Dict[str, Dict[str, Tuple[str, float]]], List[ToolExecutionLog]]:
    """
    Execute selected IQA tools and normalize scores.

    Args:
        selected_tools: Selected tools mapping
        image_path: Path to test image
        reference_path: Optional reference image path
        tool_registry: Tool registry instance

    Returns:
        Tuple of (quality_scores, tool_logs)
    """
    quality_scores = {}
    tool_logs = []

    for object_name, distortions in selected_tools.items():
        quality_scores[object_name] = {}

        for distortion, tool_name in distortions.items():
            start_time = time.time()
            log_entry = ToolExecutionLog(
                tool_name=tool_name,
                object_name=object_name,
                distortion=distortion,
                raw_score=None,
                normalized_score=None,
                execution_time=0.0,
                fallback=False,
                error=None
            )

            try:
                # Execute tool
                raw_score, normalized_score = tool_registry.execute_tool(
                    tool_name, image_path, reference_path
                )

                # Record success
                log_entry.raw_score = raw_score
                log_entry.normalized_score = normalized_score
                log_entry.execution_time = time.time() - start_time

                quality_scores[object_name][distortion] = (tool_name, normalized_score)
                logger.info(
                    f"Tool {tool_name} for {object_name}/{distortion}: "
                    f"raw={raw_score:.4f}, normalized={normalized_score:.2f}"
                )

            except ToolExecutionError as e:
                # Tool execution failed - record error without fallback
                log_entry.error = f"Tool execution failed: {e}"
                log_entry.execution_time = time.time() - start_time
                logger.error(f"Tool {tool_name} execution failed for {object_name}/{distortion}: {e}")

            except Exception as e:
                log_entry.error = str(e)
                log_entry.execution_time = time.time() - start_time
                logger.error(f"Tool execution error: {e}")

            tool_logs.append(log_entry)

    return quality_scores, tool_logs


# ==================== Executor Node ====================

def executor_node(
    state: AgenticIQAState,
    config: Optional[RunnableConfig] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Executor node function for LangGraph.

    Args:
        state: Current pipeline state
        config: Optional configuration override
        max_retries: Maximum retry attempts for VLM calls

    Returns:
        State update dict with 'executor_evidence' field or 'error' field
    """
    logger.info("Executor node starting")

    # Extract inputs from state
    query = state["query"]
    image_path = state["image_path"]
    reference_path = state.get("reference_path")
    plan = state.get("plan")

    # Check for upstream errors
    if "error" in state:
        logger.error(f"Skipping Executor due to upstream error: {state['error']}")
        return {"error": state["error"]}

    if plan is None:
        error_msg = "No plan found in state. Planner must run before Executor."
        logger.error(error_msg)
        return {"error": error_msg}

    # Load configuration
    try:
        if config is None:
            backends_config = load_model_backends()
            executor_config = backends_config.executor
        else:
            # Access custom config via RunnableConfig's "configurable" dict
            configurable = config.get("configurable", {})
            if "executor" in configurable:
                executor_config = configurable["executor"]
            else:
                backends_config = load_model_backends()
                executor_config = backends_config.executor

        logger.info(f"Using backend: {executor_config.backend}")

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
        vlm_client = create_vlm_client(
            executor_config.backend,
            {
                "api_key": None,  # Will use environment variables
                "base_url": executor_config.base_url if hasattr(executor_config, 'base_url') else None
            }
        )
        logger.info(f"Created VLM client: {vlm_client.backend_name}")

    except Exception as e:
        error_msg = f"Failed to create VLM client: {e}"
        logger.error(error_msg)
        return {"error": error_msg}

    # Create tool registry
    try:
        tool_registry = ToolRegistry()
        logger.info(f"Loaded {len(tool_registry.tools)} tools")

    except Exception as e:
        logger.warning(f"Failed to load tool registry: {e}")
        tool_registry = None

    # Initialize executor output
    executor_output = ExecutorOutput()

    # Execute subtasks based on control flags
    control_flags = plan.plan

    # 1. Distortion Detection
    if control_flags.distortion_detection:
        logger.info("Running distortion detection subtask")
        distortion_set = distortion_detection_subtask(
            query, images, vlm_client, max_retries
        )
        executor_output.distortion_set = distortion_set
    else:
        # Use distortions from Planner if available
        executor_output.distortion_set = plan.distortions

    # 2. Distortion Analysis
    if control_flags.distortion_analysis:
        logger.info("Running distortion analysis subtask")
        distortion_set_for_analysis = executor_output.distortion_set or plan.distortions

        if distortion_set_for_analysis:
            distortion_analysis = distortion_analysis_subtask(
                query, images, distortion_set_for_analysis, vlm_client, max_retries
            )
            executor_output.distortion_analysis = distortion_analysis

    # 3. Tool Selection
    if control_flags.tool_selection and tool_registry is not None:
        logger.info("Running tool selection subtask")
        distortion_set_for_selection = executor_output.distortion_set or plan.distortions

        if distortion_set_for_selection:
            # Check if Planner specified required_tool
            if plan.required_tool:
                logger.info(f"Using required tool: {plan.required_tool}")
                selected_tools = {}
                for obj, distortions in distortion_set_for_selection.items():
                    selected_tools[obj] = {dist: plan.required_tool for dist in distortions}
                executor_output.selected_tools = selected_tools
            else:
                selected_tools = tool_selection_subtask(
                    query, images, distortion_set_for_selection,
                    tool_registry, reference_path is not None,
                    vlm_client, max_retries
                )
                if selected_tools is None:
                    error_msg = "Tool selection failed after all retries"
                    logger.error(error_msg)
                    return {"error": error_msg}
                executor_output.selected_tools = selected_tools

    # 4. Tool Execution
    if control_flags.tool_execution and tool_registry is not None:
        logger.info("Running tool execution subtask")

        if executor_output.selected_tools:
            quality_scores, tool_logs = tool_execution_subtask(
                executor_output.selected_tools,
                image_path, reference_path,
                tool_registry
            )
            executor_output.quality_scores = quality_scores
            executor_output.tool_logs = tool_logs

            # Log cache stats
            cache_stats = tool_registry.get_cache_stats()
            logger.info(f"Tool cache stats: {cache_stats}")

    logger.info("Executor node completed successfully")
    return {"executor_evidence": executor_output}
