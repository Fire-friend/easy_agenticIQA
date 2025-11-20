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

# Exact prompt from docs/04_module_summarizer.md for explanation/QA mode
EXPLANATION_PROMPT_TEMPLATE = """System:
You are a visual quality assessment assistant. Your task is to select the most appropriate answer to the user's
question. You are given:
- Distortion analysis (severity and visual impact of listed distortions)
- Tool response (overall quality scores from IQA models)
- Image content
Decision process
1. First, understand what kind of visual information is needed to answer the user's question.
2. Check if the provided distortion analysis or tool response already contains the required information.
3. If the provided information is sufficient, use it to answer.
4. If the information is unclear or insufficient, analyze the image directly to determine the best answer.
Return a valid JSON object in the following format:
{{
  "final_answer": "<one of the above letters>",
  "quality_reasoning": "<brief explanation, based on either distortion analysis, tool response, or direct visual observation>"
}}

User:
User's query: {query}

Distortion analysis:
{distortion_analysis}

Tool responses:
{tool_responses}

The image: <image>"""

# Updated prompt for scoring mode - requests probability distributions
SCORING_WITH_FUSION_PROMPT_TEMPLATE = """System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion analysis), assess the image quality and provide your confidence for each quality level.

Tool scores (1-5 scale, higher is better): {tool_scores}
Tool score mean: {tool_mean:.2f}

Distortion analysis:
{distortion_analysis}

Your task:
1. Analyze the image quality based on the evidence
2. For EACH quality level, provide your log-probability (confidence):
   - Level 5 (Excellent): no visible distortions
   - Level 4 (Good): minor distortions, minimal impact
   - Level 3 (Fair): moderate distortions, noticeable impact
   - Level 2 (Poor): severe distortions, significant impact
   - Level 1 (Bad): extreme distortions, unusable quality

Return valid JSON:
{{
  "quality_probs": {{
    "1": <log_prob_1>,
    "2": <log_prob_2>,
    "3": <log_prob_3>,
    "4": <log_prob_4>,
    "5": <log_prob_5>
  }},
  "quality_reasoning": "<concise justification referencing distortions and tool scores>"
}}

User:
User's query: {query}

The image: <image>"""

# Legacy scoring prompt (for backward compatibility with MCQ queries)
SCORING_PROMPT_TEMPLATE = """System:
You are a visual quality assessment assistant. Given the question and the analysis (tool scores, distortion
analysis). Your task is to assess the image quality.
You must select one single answer from the following:
A. Excellent
B. Good
C. Fair
D. Poor
E. Bad
Return the JSON:
{{
  "final_answer": "<one letter>",
  "quality_reasoning": "<concise justification referencing distortions or tool scores>"
}}

User:
User's query: {query}

Distortion analysis:
{distortion_analysis}

Tool scores:
{tool_scores}

The image: <image>"""


# ==================== Evidence Formatting ====================

def format_evidence_for_explanation(executor_output: Optional[ExecutorOutput]) -> Tuple[str, str]:
    """
    Format Executor evidence for explanation/QA mode prompt.

    Args:
        executor_output: Executor evidence

    Returns:
        Tuple of (distortion_analysis_json, tool_responses_json)
    """
    if not executor_output:
        return "No evidence available", "No tool responses available"

    # Format distortion analysis
    if executor_output.distortion_analysis:
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

    # Format tool responses
    if executor_output.quality_scores:
        tool_json = {}
        for obj_name, scores in executor_output.quality_scores.items():
            tool_json[obj_name] = {
                distortion: {"tool": tool_name, "score": score}
                for distortion, (tool_name, score) in scores.items()
            }
        tool_text = json.dumps(tool_json, indent=2)
    else:
        tool_text = "No tool responses available"

    return distortion_text, tool_text


def format_evidence_for_scoring(executor_output: Optional[ExecutorOutput]) -> Tuple[str, str]:
    """
    Format Executor evidence for scoring mode prompt.

    Args:
        executor_output: Executor evidence

    Returns:
        Tuple of (distortion_analysis_json, tool_scores_json)
    """
    if not executor_output:
        return "No evidence available", "No tool scores available"

    # Format distortion analysis (same as explanation mode)
    if executor_output.distortion_analysis:
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

    # Format tool scores (simpler format for scoring mode)
    if executor_output.quality_scores:
        tool_json = {}
        for obj_name, scores in executor_output.quality_scores.items():
            tool_json[obj_name] = {
                distortion: {"tool": tool_name, "score": float(score)}
                for distortion, (tool_name, score) in scores.items()
            }
        tool_text = json.dumps(tool_json, indent=2)
    else:
        tool_text = "No tool scores available"

    return distortion_text, tool_text


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

        logger.info(f"Query type: {plan.query_type}")
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

        # Extract all tool scores for fusion (used in multiple modes)
        all_scores = []
        if executor_output and executor_output.quality_scores:
            for obj_scores in executor_output.quality_scores.values():
                for tool_name, score in obj_scores.values():
                    all_scores.append(score)
        tool_mean = sum(all_scores) / len(all_scores) if all_scores else 3.0

        # Select prompt mode and format evidence
        if plan.query_type == "IQA" and query_type_detected == QueryType.MCQ:
            # MCQ MODE: Categorical selection, final_answer is letter
            logger.info("Using MCQ mode (categorical selection)")
            distortion_text, tool_text = format_evidence_for_explanation(executor_output)

            prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                query=query,
                distortion_analysis=distortion_text,
                tool_responses=tool_text
            )

            apply_fusion = False
            tool_scores_for_fusion = []

        elif plan.query_type == "IQA" and query_type_detected == QueryType.SCORING:
            # SCORING MODE: Request probability distributions and apply fusion
            # final_answer is numerical score
            logger.info("Using SCORING mode with fusion")
            distortion_text, tool_text = format_evidence_for_scoring(executor_output)

            logger.info(f"Tool scores: {all_scores}, mean: {tool_mean:.2f}")

            prompt = SCORING_WITH_FUSION_PROMPT_TEMPLATE.format(
                query=query,
                distortion_analysis=distortion_text,
                tool_scores=tool_text,
                tool_mean=tool_mean
            )

            # Flag to apply fusion after VLM response
            apply_fusion = True
            tool_scores_for_fusion = all_scores

        else:
            # EXPLANATION MODE: Answer question in quality_reasoning, letter grade in final_answer
            # final_answer is letter grade (A-E), quality_reasoning contains the actual answer
            logger.info("Using EXPLANATION mode (answer in quality_reasoning, letter grade in final_answer)")
            distortion_text, tool_text = format_evidence_for_explanation(executor_output)

            prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                query=query,
                distortion_analysis=distortion_text,
                tool_responses=tool_text
            )

            # For EXPLANATION queries, we use letter grade as final_answer
            apply_fusion = False
            tool_scores_for_fusion = all_scores

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

                # Parse JSON with automatic extraction
                response_data = parse_json_response(vlm_response)

                # Determine output format based on query type and apply fusion if needed
                if query_type_detected == QueryType.MCQ:
                    # MCQ MODE: final_answer is letter, quality_score is None
                    if "final_answer" not in response_data or "quality_reasoning" not in response_data:
                        raise ValueError("Missing required fields in VLM response")

                    final_answer = response_data["final_answer"]
                    quality_score = None
                    quality_reasoning = response_data["quality_reasoning"]
                    logger.info(f"MCQ output: final_answer={final_answer} (letter)")

                elif query_type_detected == QueryType.SCORING:
                    # SCORING MODE: Apply fusion, final_answer is numerical score
                    logger.info("Applying score fusion for SCORING query")

                    if "quality_probs" not in response_data and "quality_reasoning" not in response_data:
                        raise ValueError("Missing quality_probs and quality_reasoning in VLM response")

                    # Create ScoreFusion instance
                    fusion = ScoreFusion(eta=1.0)

                    # Extract VLM probabilities
                    vlm_probs = fusion.extract_vlm_probs(response_data)

                    # Apply fusion formula
                    if tool_scores_for_fusion:
                        fused_score = fusion.fuse(tool_scores_for_fusion, vlm_probs)
                        logger.info(f"Fused score: {fused_score:.3f}")
                    else:
                        # No tool scores: use VLM probs only
                        fused_score = sum(c * vlm_probs[c] for c in [1, 2, 3, 4, 5])
                        logger.warning(f"No tool scores, using VLM probs only: {fused_score:.3f}")

                    # Clip to [1.0, 5.0] range
                    fused_score = max(1.0, min(5.0, fused_score))

                    # Set final_answer and quality_score
                    final_answer = fused_score
                    quality_score = fused_score
                    quality_reasoning = response_data.get("quality_reasoning", "Score computed from fusion")
                    logger.info(f"SCORING output: final_answer={final_answer:.3f} (numerical)")

                else:
                    # EXPLANATION MODE: VLM answer goes to quality_reasoning, final_answer is letter grade
                    logger.info("Processing EXPLANATION query: mapping tool mean to letter grade for final_answer")

                    if "final_answer" not in response_data or "quality_reasoning" not in response_data:
                        raise ValueError("Missing required fields in VLM response")

                    # VLM's answer is incorporated into quality_reasoning
                    vlm_answer = response_data["final_answer"]
                    vlm_reasoning = response_data["quality_reasoning"]

                    # Combine VLM's answer with reasoning
                    quality_reasoning = f"{vlm_answer}. {vlm_reasoning}"

                    # Compute tool mean and map to letter grade
                    fusion = ScoreFusion(eta=1.0)

                    if tool_scores_for_fusion:
                        tool_score = sum(tool_scores_for_fusion) / len(tool_scores_for_fusion)
                        tool_score = max(1.0, min(5.0, tool_score))  # Clip to [1.0, 5.0]
                        logger.info(f"Tool mean for EXPLANATION: {tool_score:.3f}")
                    else:
                        # No tool scores: use neutral score
                        tool_score = 3.0
                        logger.warning("No tool scores, using neutral score 3.0")

                    # Map numerical score to quality level and letter grade
                    quality_level = fusion.map_to_level(tool_score)
                    letter_grade = fusion.map_to_letter(quality_level)

                    # final_answer is letter grade (paper requirement for non-scoring queries)
                    final_answer = letter_grade
                    quality_score = tool_score
                    logger.info(f"EXPLANATION output: final_answer={final_answer} (letter), quality_score={quality_score:.3f}, answer in quality_reasoning")

                # Check evidence sufficiency
                iteration_count = state.get("iteration_count", 0)
                max_replan_iterations = state.get("max_replan_iterations", 2)

                need_replan, replan_reason = check_evidence_sufficiency(
                    executor_output=executor_output,
                    query_scope=plan.query_scope,
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
                replan_history = state.get("replan_history", [])
                if need_replan:
                    history_entry = f"[Iteration {iteration_count}] {replan_reason}"
                    replan_history.append(history_entry)
                    # Limit history size
                    if len(replan_history) > 10:
                        logger.warning("Replan history exceeds 10 entries, trimming oldest")
                        replan_history = replan_history[-10:]

                logger.info("Summarizer completed successfully")

                return {
                    "summarizer_result": summarizer_output,
                    "replan_history": replan_history
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
