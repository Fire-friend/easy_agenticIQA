"""
Summarizer agent node for the AgenticIQA pipeline.

Synthesizes Executor evidence with VLM visual understanding to produce final answers
and quality reasoning. Supports two modes (explanation/QA and scoring) and implements
conditional replanning when evidence is insufficient.
"""

import json
import logging
import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from PIL import Image
from langchain_core.runnables import RunnableConfig

from src.agentic.state import AgenticIQAState, SummarizerOutput, ExecutorOutput
from src.agentic.vlm_client import create_vlm_client, load_image
from src.agentic.score_fusion import ScoreFusion
from src.utils.config import load_model_backends
from src.utils.json_utils import parse_json_response

logger = logging.getLogger(__name__)


# ==================== Query Type Detection ====================

class QueryType(Enum):
    """Query type classification for determining output format."""
    SCORING = "scoring"           # Pure quality scoring → numerical score
    MCQ = "mcq"                   # Multiple choice question → letter
    EXPLANATION = "explanation"   # Descriptive explanation → text


def detect_query_type_with_vlm(query: str, vlm_client) -> QueryType:
    """
    Detect query type using VLM for more accurate intent recognition.

    Args:
        query: User's natural language query
        vlm_client: VLM client for inference

    Returns:
        QueryType enum value
    """
    # Quick rule-based detection for MCQ (explicit options are obvious)
    if re.search(r'[A-E]\)', query) or "choose from" in query.lower():
        logger.info(f"Detected MCQ query (explicit options): {query[:50]}")
        return QueryType.MCQ

    # Use VLM for intent classification
    classification_prompt = f"""Analyze this image quality assessment query and classify its type.

Query: "{query}"

Types:
1. SCORING - User wants a numerical quality score (1-5 scale)
   Examples: "Rate the quality", "What is the quality score", "Assess the image quality"

2. EXPLANATION - User wants descriptive information or yes/no answer
   Examples: "What is the major distortion", "Does this have blur", "Why is it blurry", "What problems exist"

Return ONLY a JSON object:
{{
  "type": "SCORING" or "EXPLANATION",
  "reasoning": "<brief explanation>"
}}"""

    try:
        response = vlm_client.generate(
            prompt=classification_prompt,
            images=[],
            temperature=0.0,
            max_tokens=100
        )

        from src.utils.json_utils import parse_json_response
        result = parse_json_response(response)

        query_type_str = result.get("type", "EXPLANATION").upper()
        reasoning = result.get("reasoning", "")

        if query_type_str == "SCORING":
            logger.info(f"VLM detected SCORING query: {query[:50]} | Reasoning: {reasoning}")
            return QueryType.SCORING
        else:
            logger.info(f"VLM detected EXPLANATION query: {query[:50]} | Reasoning: {reasoning}")
            return QueryType.EXPLANATION

    except Exception as e:
        logger.warning(f"VLM query type detection failed: {e}, falling back to rule-based")
        return detect_query_type_rule_based(query)


def detect_query_type_rule_based(query: str) -> QueryType:
    """
    Fallback rule-based query type detection.

    Args:
        query: User's natural language query

    Returns:
        QueryType enum value
    """
    query_lower = query.lower()

    # MCQ: explicit options
    if re.search(r'[A-E]\)', query) or "choose from" in query_lower:
        return QueryType.MCQ

    # SCORING: rate/score/assess/evaluate keywords
    if re.search(r'\b(rate|score|assess|evaluate)\b', query_lower):
        return QueryType.SCORING

    # SCORING: "what is the quality/score" patterns
    if re.search(r'\bwhat\s+(is|are)\s+the\s+(quality\s+)?(score|level)', query_lower):
        return QueryType.SCORING

    # Default: EXPLANATION
    return QueryType.EXPLANATION


# Keep old function name for compatibility
def detect_query_type(query: str, vlm_client=None) -> QueryType:
    """
    Detect query type to determine output format.
    Uses VLM if provided, otherwise falls back to rule-based detection.

    Args:
        query: User's natural language query
        vlm_client: Optional VLM client for better detection

    Returns:
        QueryType enum value
    """
    if vlm_client is not None:
        return detect_query_type_with_vlm(query, vlm_client)
    else:
        return detect_query_type_rule_based(query)


# ==================== Prompt Templates ====================

# Paper's unified prompt template for all query types
# Source: Original AgenticIQA paper - Summarizer system message
PAPER_UNIFIED_PROMPT_TEMPLATE = """You are a **summarizer assistant** in an Image Quality Assessment (IQA) agent system. Your task is to integrate information from prior distortion analysis and computed IQA tool scores to produce a comprehensive quality interpretation and directly answer the user query.

**User Query:** {query}

**Your Input Includes:**

1. **Distortion Analysis:** Severity, category, and explanation of detected distortions per region or globally.
2. **IQA Tool Scores:** Quality scores (range 1 to 5) assigned by specific IQA tools based on the distortions.
3. **Reference Type:** {reference_type}
4. **Optional Prior Answer:** {prior_answer}
5. **Optional Image Input:** You may also infer the answer from images and query directly.

**Return a valid JSON object in the following format:**

```json
{{
  "quality_reasoning": "<Summary of the reasoning process, combining distortions, severity, descriptions, and IQA scores>",
  "final_answer": "<Concise and direct response to the user query based on the full reasoning>"
}}
```

**Guidelines:**

* In `"quality_reasoning"`:
  * Summarize the key distortions and their visual impact.
  * Reference tool scores to support your conclusion.
  * Ensure the logic connecting observations and conclusions are clear and interpretable.

* In `"final_answer"`:
  * Provide a direct and concise judgment regarding the user query.
  * Use natural and human-readable phrasing.

* Only return the JSON object. Do not include any markdown, commentary, or additional text.

**Distortion Analysis:**
{distortion_analysis}

**IQA Tool Scores:**
{tool_scores}

The image: <image>"""


# ==================== Evidence Formatting ====================

def format_evidence_unified(
    executor_output: Optional[ExecutorOutput],
    reference_path: Optional[str],
    iteration_count: int,
    replan_history: List[str],
    previous_result: Optional[SummarizerOutput] = None
) -> Dict[str, str]:
    """
    Format Executor evidence for the unified prompt template.

    Generates all evidence fields required by PAPER_UNIFIED_PROMPT_TEMPLATE:
    - distortion_analysis: JSON formatted distortion analysis
    - tool_scores: JSON formatted tool scores (1-5 range)
    - reference_type: "Full-Reference" or "No-Reference"
    - prior_answer: Previous iteration's answer or "None"

    Args:
        executor_output: Executor evidence containing distortion_analysis and quality_scores
        reference_path: Path to reference image (if FR mode)
        iteration_count: Current iteration count (0-indexed)
        replan_history: List of replan reasons from previous iterations
        previous_result: Previous SummarizerOutput (if available)

    Returns:
        Dict with keys: distortion_analysis, tool_scores, reference_type, prior_answer
    """
    # Format distortion analysis
    if executor_output and executor_output.distortion_analysis:
        distortion_json = {}
        for obj_name, analyses in executor_output.distortion_analysis.items():
            distortion_json[obj_name] = [
                {
                    "type": analysis.type,
                    "severity": analysis.severity,
                    "explanation": analysis.explanation
                }
                for analysis in analyses
            ]
        distortion_text = json.dumps(distortion_json, indent=2)
    else:
        distortion_text = "No distortion analysis available"

    # Format tool scores
    if executor_output and executor_output.quality_scores:
        tool_json = {}
        for obj_name, scores in executor_output.quality_scores.items():
            tool_json[obj_name] = {
                distortion: {"tool": tool_name, "score": float(score)}
                for distortion, (tool_name, score) in scores.items()
            }
        tool_text = json.dumps(tool_json, indent=2)
    else:
        tool_text = "No tool scores available"

    # Determine reference type
    reference_type = "Full-Reference" if reference_path else "No-Reference"

    # Format prior answer for replanning iterations
    if iteration_count > 0 and replan_history:
        # Get the most recent replan reason
        latest_reason = replan_history[-1] if replan_history else "Unknown reason"

        if previous_result:
            prior_answer = (
                f"Previous answer: {previous_result.final_answer}\n"
                f"Previous reasoning: {previous_result.quality_reasoning}\n"
                f"Previous iteration failed because: {latest_reason}"
            )
        else:
            prior_answer = f"Previous iteration failed because: {latest_reason}"
    else:
        prior_answer = "This is the first iteration"

    return {
        "distortion_analysis": distortion_text,
        "tool_scores": tool_text,
        "reference_type": reference_type,
        "prior_answer": prior_answer
    }


# ==================== Replanning Decision Logic ====================

def check_evidence_sufficiency(
    executor_output: Optional[ExecutorOutput],
    query_scope: Any,  # Can be str "Global" or List[str]
    max_iterations: int,
    current_iteration: int
) -> Tuple[bool, str]:
    """
    Determine if evidence is sufficient or if replanning is needed.

    This function only assesses evidence quality, NOT iteration limits.
    Iteration limit checking is handled by decide_next_node() in the graph.

    Args:
        executor_output: Executor evidence
        query_scope: Query scope from Planner (Global or list of objects)
        max_iterations: Maximum replanning iterations allowed (unused, kept for compatibility)
        current_iteration: Current iteration count (unused, kept for compatibility)

    Returns:
        Tuple of (need_replan, reason)
    """
    # Note: We don't check iteration limits here - that's the graph's job
    # This function only reports whether evidence is sufficient

    if not executor_output:
        return True, "No Executor evidence available"

    # Determine required objects
    if isinstance(query_scope, str) and query_scope == "Global":
        required_objects = {"Global"}
    elif isinstance(query_scope, list):
        required_objects = set(query_scope)
    else:
        required_objects = {str(query_scope)} if query_scope else {"Global"}

    # Check distortion analysis coverage
    if executor_output.distortion_analysis:
        covered_objects = set(executor_output.distortion_analysis.keys())
        missing_objects = required_objects - covered_objects
        if missing_objects:
            reason = f"Missing distortion analysis for {missing_objects}"
            logger.warning(f"Evidence gap detected: {reason}")
            return True, reason

    # Check tool scores availability
    if not executor_output.quality_scores or len(executor_output.quality_scores) == 0:
        reason = "No tool scores available"
        logger.warning(f"Evidence gap detected: {reason}")
        return True, reason

    # Check for contradictory evidence (severe distortion but high scores)
    if executor_output.distortion_analysis and executor_output.quality_scores:
        for obj_name, analyses in executor_output.distortion_analysis.items():
            for analysis in analyses:
                if analysis.severity in ["severe", "extreme"]:
                    # Check if corresponding scores are high
                    if obj_name in executor_output.quality_scores:
                        scores = executor_output.quality_scores[obj_name]
                        if analysis.type in scores:
                            _, score = scores[analysis.type]
                            if score > 4.0:  # High score despite severe distortion
                                reason = f"Contradictory evidence: {analysis.severity} {analysis.type} but high score ({score:.2f})"
                                logger.warning(f"Contradiction detected: {reason}")
                                # Note: We don't replan for contradictions in Phase 4, just log it
                                # return True, reason

    # Evidence appears sufficient
    return False, ""


# ==================== Summarizer Node ====================

def summarizer_node(
    state: AgenticIQAState,
    config: Optional[RunnableConfig] = None,
    max_retries: int = 3
) -> Dict[str, Any]:
    """
    Summarizer agent node - synthesizes evidence and produces final answer.

    Workflow:
    1. Check prerequisites (plan, executor_evidence)
    2. Load VLM client and images
    3. Select prompt mode (explanation/QA vs scoring)
    4. Format evidence and render prompt
    5. Call VLM with retry logic
    6. Parse and validate output
    7. Check evidence sufficiency for replanning
    8. Return state update with summarizer_result

    Args:
        state: Current AgenticIQA state
        config: LangGraph configuration
        max_retries: Maximum retry attempts for VLM calls

    Returns:
        State update with summarizer_result and iteration tracking
    """
    logger.info("=" * 60)
    logger.info("Summarizer Node Started")
    logger.info("=" * 60)

    try:
        # Check for upstream errors
        if "error" in state:
            logger.error(f"Skipping Summarizer due to upstream error: {state['error']}")
            return {"error": state["error"]}

        # Check prerequisites
        if "plan" not in state:
            error_msg = "No plan found in state, Summarizer cannot proceed"
            logger.error(error_msg)
            return {"error": error_msg}

        if "executor_evidence" not in state:
            error_msg = "No executor_evidence found in state, Summarizer cannot proceed"
            logger.error(error_msg)
            return {"error": error_msg}

        plan = state["plan"]
        executor_output = state.get("executor_evidence")
        query = state["query"]
        image_path = state["image_path"]
        reference_path = state.get("reference_path")

        logger.info(f"Task type: {plan.task_type}")
        logger.info(f"Query: {query}")
        logger.info(f"Image: {image_path}")

        # Load configuration
        try:
            config_data = load_model_backends()
            if not hasattr(config_data, 'summarizer'):
                raise ValueError("summarizer section missing in model_backends.yaml")

            summarizer_config = config_data.summarizer
            backend_name = summarizer_config.backend
            temperature = getattr(summarizer_config, 'temperature', 0.0)
            max_tokens = getattr(summarizer_config, 'max_tokens', 512)

            logger.info(f"VLM backend: {backend_name}, temperature: {temperature}")

        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Load images
        try:
            test_image = load_image(image_path)
            images = [test_image]

            if reference_path:
                ref_image = load_image(reference_path)
                images.append(ref_image)
                logger.info("Loaded test and reference images")
            else:
                logger.info("Loaded test image only")

        except Exception as e:
            error_msg = f"Failed to load images: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Create VLM client
        try:
            vlm_client = create_vlm_client(backend_name)
            logger.info(f"Created VLM client: {vlm_client.backend_name}")
        except Exception as e:
            error_msg = f"Failed to create VLM client: {e}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Detect query type using VLM for better intent recognition
        query_type_detected = detect_query_type(query, vlm_client=vlm_client)
        logger.info(f"Detected query type: {query_type_detected.value}")

        # Extract all tool scores for fusion (used in scoring mode)
        all_scores = []
        if executor_output and executor_output.quality_scores:
            for obj_scores in executor_output.quality_scores.values():
                for tool_name, score in obj_scores.values():
                    all_scores.append(score)
        tool_mean = sum(all_scores) / len(all_scores) if all_scores else 3.0

        # Get iteration context for prior answer
        iteration_count = state.get("iteration_count", 0)
        replan_history = state.get("replan_history", [])
        previous_result = state.get("summarizer_result")  # May be None on first iteration

        # Format evidence using unified function (per paper specification)
        evidence = format_evidence_unified(
            executor_output=executor_output,
            reference_path=reference_path,
            iteration_count=iteration_count,
            replan_history=replan_history,
            previous_result=previous_result
        )

        logger.info(f"Using unified prompt template (paper specification)")
        logger.info(f"Reference type: {evidence['reference_type']}")
        if iteration_count > 0:
            logger.info(f"Replanning iteration {iteration_count}, prior answer included")

        # Render unified prompt template
        prompt = PAPER_UNIFIED_PROMPT_TEMPLATE.format(
            query=query,
            distortion_analysis=evidence["distortion_analysis"],
            tool_scores=evidence["tool_scores"],
            reference_type=evidence["reference_type"],
            prior_answer=evidence["prior_answer"]
        )

        logger.debug(f"Prompt length: {len(prompt)} characters")

        # Call VLM with retry logic
        vlm_response = None
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"VLM call attempt {attempt}/{max_retries}")
                vlm_response = vlm_client.generate(
                    prompt=prompt,
                    images=images,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                logger.debug(f"VLM response: {vlm_response[:500]}")

                # Parse JSON with automatic extraction (unified format: quality_reasoning + final_answer)
                response_data = parse_json_response(vlm_response)

                # Validate required fields from unified prompt
                if "final_answer" not in response_data or "quality_reasoning" not in response_data:
                    raise ValueError("Missing required fields (final_answer, quality_reasoning) in VLM response")

                vlm_final_answer = response_data["final_answer"]
                quality_reasoning = response_data["quality_reasoning"]

                # Interpret final_answer based on detected query type
                if query_type_detected == QueryType.MCQ:
                    # MCQ MODE: Extract letter from final_answer
                    final_answer = str(vlm_final_answer).strip()
                    # Extract first letter if answer contains more text
                    if final_answer and final_answer[0].upper() in "ABCDE":
                        final_answer = final_answer[0].upper()
                    quality_score = None
                    logger.info(f"MCQ output: final_answer={final_answer} (letter)")

                elif query_type_detected == QueryType.SCORING:
                    # SCORING MODE: Apply fusion with tool scores
                    logger.info("Processing SCORING query with unified prompt")

                    fusion = ScoreFusion(eta=1.0)

                    # Try to extract numerical score from VLM's final_answer
                    vlm_score = None
                    try:
                        # Check if final_answer is already a number
                        vlm_score = float(vlm_final_answer)
                    except (ValueError, TypeError):
                        # Try to extract from quality level text
                        vlm_answer_str = str(vlm_final_answer).lower()
                        level_map = {"excellent": 5, "good": 4, "fair": 3, "poor": 2, "bad": 1}
                        for level_name, level_score in level_map.items():
                            if level_name in vlm_answer_str:
                                vlm_score = float(level_score)
                                break

                    # Construct VLM probability distribution (softened around detected level)
                    if vlm_score is not None:
                        # Use softened distribution centered on VLM's indicated level
                        vlm_level = max(1, min(5, round(vlm_score)))
                        vlm_probs = {}
                        for c in [1, 2, 3, 4, 5]:
                            # Gaussian-like softening around the selected level
                            vlm_probs[c] = max(0.01, 1.0 - 0.3 * abs(c - vlm_level))
                        # Normalize
                        total = sum(vlm_probs.values())
                        vlm_probs = {c: p / total for c, p in vlm_probs.items()}
                    else:
                        # Default to uniform distribution
                        vlm_probs = {c: 0.2 for c in [1, 2, 3, 4, 5]}

                    # Apply fusion formula with tool scores
                    if all_scores:
                        fused_score = fusion.fuse(all_scores, vlm_probs)
                        logger.info(f"Fused score: {fused_score:.3f} (tools: {all_scores}, vlm_score: {vlm_score})")
                    else:
                        # No tool scores: use VLM assessment directly
                        fused_score = vlm_score if vlm_score else 3.0
                        logger.warning(f"No tool scores, using VLM score: {fused_score:.3f}")

                    # Clip to [1.0, 5.0] range
                    fused_score = max(1.0, min(5.0, fused_score))

                    final_answer = fused_score
                    quality_score = fused_score
                    logger.info(f"SCORING output: final_answer={final_answer:.3f} (numerical)")

                else:
                    # EXPLANATION MODE: Use VLM's direct answer
                    logger.info("Processing EXPLANATION query with unified prompt")

                    # final_answer is VLM's direct response to the query
                    final_answer = str(vlm_final_answer)

                    # Compute quality_score from tool mean for reference
                    if all_scores:
                        quality_score = sum(all_scores) / len(all_scores)
                        quality_score = max(1.0, min(5.0, quality_score))
                    else:
                        quality_score = None

                    logger.info(f"EXPLANATION output: final_answer={final_answer[:50]}..., quality_score={quality_score}")

                # Check evidence sufficiency (iteration_count already retrieved above)
                max_replan_iterations = state.get("max_replan_iterations", 2)

                need_replan, replan_reason = check_evidence_sufficiency(
                    executor_output=executor_output,
                    query_scope=plan.required_object_names,
                    max_iterations=max_replan_iterations,
                    current_iteration=iteration_count
                )

                # Create SummarizerOutput
                summarizer_output = SummarizerOutput(
                    final_answer=final_answer,
                    quality_score=quality_score,
                    quality_reasoning=quality_reasoning,
                    need_replan=need_replan,
                    replan_reason=replan_reason if need_replan else None
                )

                logger.info(f"Summarizer output created: final_answer={final_answer}, quality_score={quality_score}, need_replan={need_replan}")
                if need_replan:
                    logger.info(f"Replan reason: {replan_reason}")

                # Note: iteration_count is incremented by the planner when it re-runs,
                # not here. We only report whether replanning is needed.

                # Update replan history only if replanning is needed
                # (replan_history was already retrieved earlier for evidence formatting)
                updated_replan_history = list(replan_history)  # Make a copy
                if need_replan:
                    history_entry = f"[Iteration {iteration_count}] {replan_reason}"
                    updated_replan_history.append(history_entry)
                    # Limit history size
                    if len(updated_replan_history) > 10:
                        logger.warning("Replan history exceeds 10 entries, trimming oldest")
                        updated_replan_history = updated_replan_history[-10:]

                logger.info("Summarizer completed successfully")

                return {
                    "summarizer_result": summarizer_output,
                    "replan_history": updated_replan_history
                }

            except json.JSONDecodeError as e:
                logger.warning(f"Attempt {attempt} failed: Invalid JSON - {e}")
                if attempt < max_retries:
                    prompt += "\n\nReturn ONLY valid JSON."
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed with invalid JSON")

            except ValueError as e:
                logger.warning(f"Attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed")

            except Exception as e:
                logger.error(f"Attempt {attempt} failed with unexpected error: {e}")
                if attempt < max_retries:
                    continue
                else:
                    logger.error(f"All {max_retries} attempts failed")

        # All retries failed - return fallback output
        logger.error("All VLM retries failed, returning fallback output")

        # Determine fallback answer type based on detected query type
        if query_type_detected == QueryType.SCORING:
            # For SCORING queries, return neutral numerical score
            fallback_answer = 3.0
            fallback_quality_score = 3.0
        else:
            # For MCQ/EXPLANATION queries, return letter grade
            fallback_answer = "C"  # Fair (neutral grade)
            fallback_quality_score = 3.0

        fallback_output = SummarizerOutput(
            final_answer=fallback_answer,
            quality_score=fallback_quality_score,
            quality_reasoning="VLM output parsing failed after all retries",
            need_replan=False  # Don't trigger replan loop on VLM failure
        )

        return {
            "summarizer_result": fallback_output,
            "iteration_count": state.get("iteration_count", 0)
        }

    except Exception as e:
        error_msg = f"Summarizer node failed: {e}"
        logger.error(error_msg, exc_info=True)
        return {"error": error_msg}
