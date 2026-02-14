"""
Confidence scorer for ICD code search results.

Evaluates hybrid search results using LLM-based medical coding expertise
and assigns final confidence scores to each candidate ICD code.
"""
import logging
from typing import Literal

from core.prompts.confidence_scoring_prompt import (
    ConfidenceScoringResult,
    SYSTEM_PROMPT_FOR_CONFIDENCE_SCORING,
    get_confidence_scoring_prompt,
    CONFIDENCE_SCORING_CONFIG,
)
from utils.groq_llms import call_groq_structured

logger = logging.getLogger(__name__)


def score_search_results(
    user_query: str,
    search_results: list[dict],
    query_type: Literal["diagnosis", "procedure"],
    top_k: int = 5,
) -> ConfidenceScoringResult:
    """
    Score search results using LLM-based confidence evaluation.

    Takes reranked hybrid search results and applies medical coding expertise
    to assign final confidence scores to each ICD code candidate.

    This function evaluates all candidates provided in search_results (typically 10-15
    from hybrid search + reranking) and lets the LLM independently score them. The
    evaluated codes are then sorted by the LLM's relevance_score (not retrieval score)
    to ensure the best matches rise to the top based on clinical reasoning.

    Args:
        user_query: The user's original medical query text.
        search_results: List of result dicts from hybrid_search, each containing
            code_dotted, long_description, category_title, and score.
        query_type: Type of query ("diagnosis" or "procedure").
        top_k: Number of candidate results to evaluate (not the final return count).

    Returns:
        ConfidenceScoringResult with evaluated codes sorted by LLM relevance_score.

    Raises:
        RuntimeError: If the LLM call fails.
    """
    results_to_evaluate = search_results[:top_k]

    logger.info(
        "Starting confidence scoring: query='%s', evaluating %d results",
        user_query[:50],
        len(results_to_evaluate),
    )

    try:
        prompt = get_confidence_scoring_prompt(
            user_query=user_query,
            search_results=results_to_evaluate,
            query_type=query_type,
        )

        result = call_groq_structured(
            prompt=prompt,
            response_model=ConfidenceScoringResult,
            system_prompt=SYSTEM_PROMPT_FOR_CONFIDENCE_SCORING,
            temperature=CONFIDENCE_SCORING_CONFIG["temperature"],
            max_tokens=CONFIDENCE_SCORING_CONFIG["max_tokens"],
            model=CONFIDENCE_SCORING_CONFIG["model"],
            strict=CONFIDENCE_SCORING_CONFIG["strict"],
        )

        result.evaluated_codes.sort(key=lambda x: x.relevance_score, reverse=True)

        logger.info(
            "Confidence scoring completed: best_code=%s, overall_confidence=%s",
            result.best_code,
            result.overall_confidence,
        )

        return result

    except Exception as e:
        logger.error("Confidence scoring failed for query '%s': %s", user_query[:50], e)
        raise RuntimeError(f"Confidence scoring failed: {e}") from e
