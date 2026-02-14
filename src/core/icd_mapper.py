"""
Main ICD code mapper orchestrator.

Coordinates the full pipeline:
  query -> type detection (if auto) -> hybrid search + reranking -> confidence scoring -> results

Each step is timed and latencies are returned alongside the results.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from core.text_type_detector import detect_text_type
from core.confidence_scorer import score_search_results
from rag.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)

COLLECTION_NAME = "icd_names"
DEFAULT_RERANK_CANDIDATES = 20


@dataclass
class ICDMapperResult:
    """Result container for the ICD mapping pipeline."""

    query_text: str
    query_type: str
    top_k: int
    results: list[dict[str, Any]]
    latencies: dict[str, float] = field(default_factory=dict)


def _ms_since(start: float) -> float:
    """Return elapsed milliseconds since *start*."""
    return (time.perf_counter() - start) * 1000


def map_icd_codes(
    query_text: str,
    query_type: Literal["auto", "diagnosis", "procedure"] = "auto",
    top_k: int = 5,
) -> ICDMapperResult:
    """
    Map medical text to ICD-10 codes with confidence scores.

    Pipeline steps:
        1. Type detection  (only when query_type == "auto")
        2. Hybrid search   (dense + sparse + RRF + reranker)
        3. Confidence score (LLM-based evaluation of top candidates)
        4. Format results   (merge LLM scores with Qdrant payloads)

    Args:
        query_text: User's medical query (condition or procedure description).
        query_type: "auto" to detect automatically, or "diagnosis" / "procedure".
        top_k: Number of final results to return.

    Returns:
        ICDMapperResult with ranked results and per-step latencies.

    Raises:
        ValueError: If inputs are invalid.
        RuntimeError: If any pipeline step fails.
    """
    if not query_text or not query_text.strip():
        raise ValueError("query_text cannot be empty")

    pipeline_start = time.perf_counter()
    latencies: dict[str, float] = {}


    detected_type: str = query_type
    if query_type == "auto":
        t0 = time.perf_counter()
        try:
            detection = detect_text_type(query_text)
            detected_type = detection.text_type
            logger.info(
                "Type detection: %s (confidence=%s, latency=%.1fms)",
                detected_type,
                detection.confidence_level,
                _ms_since(t0),
            )
        except Exception as e:
            logger.error("Type detection failed, defaulting to diagnosis: %s", e)
            detected_type = "diagnosis"
        latencies["type_detection_ms"] = _ms_since(t0)


    t0 = time.perf_counter()
    try:
        search_results = hybrid_search(
            query=query_text,
            collection_name=COLLECTION_NAME,
            code_type=detected_type,
            top_k=DEFAULT_RERANK_CANDIDATES,
            use_reranker=True,
        )
        logger.info(
            "Hybrid search: %d results (latency=%.1fms)",
            len(search_results),
            _ms_since(t0),
        )
    except Exception as e:
        latencies["hybrid_search_ms"] = _ms_since(t0)
        raise RuntimeError(f"Hybrid search failed: {e}") from e
    latencies["hybrid_search_ms"] = _ms_since(t0)

    if not search_results:
        latencies["total_ms"] = _ms_since(pipeline_start)
        return ICDMapperResult(
            query_text=query_text,
            query_type=detected_type,
            top_k=top_k,
            results=[],
            latencies=latencies,
        )


    scoring_pool_size = min(max(top_k * 10, 30), len(search_results))

    scoring_input = [
        {
            "code": r.payload.get("code"),
            "code_dotted": r.payload.get("code_dotted"),
            "long_description": r.payload.get("long_description"),
            "category_title": r.payload.get("category_title"),
            "score": r.score,
        }
        for r in search_results[:scoring_pool_size]
    ]

    t0 = time.perf_counter()
    try:
        confidence_result = score_search_results(
            user_query=query_text,
            search_results=scoring_input,
            query_type=detected_type,
            top_k=scoring_pool_size,
        )
        logger.info(
            "Confidence scoring: best=%s (latency=%.1fms)",
            confidence_result.best_code,
            _ms_since(t0),
        )
    except Exception as e:
        latencies["confidence_scoring_ms"] = _ms_since(t0)
        raise RuntimeError(f"Confidence scoring failed: {e}") from e
    latencies["confidence_scoring_ms"] = _ms_since(t0)

    payload_lookup = {
        r.payload.get("code_dotted"): r.payload for r in search_results
    }

    final_results: list[dict[str, Any]] = []
    for eval_code in confidence_result.evaluated_codes:
        original = payload_lookup.get(eval_code.code)
        if original is None:
            continue

        code_raw = original.get("code", "")
        final_results.append(
            {
                "code": code_raw,
                "code_dotted": original.get("code_dotted", ""),
                "long_description": original.get("long_description", ""),
                "short_description": original.get("long_description", ""),
                "category_code": code_raw[:3] if len(code_raw) >= 3 else code_raw,
                "category_title": original.get("category_title", ""),
                "score": eval_code.relevance_score,
                "confidence": eval_code.confidence,
            }
        )
        if len(final_results) >= top_k:
            break

    latencies["total_ms"] = _ms_since(pipeline_start)

    logger.info(
        "Pipeline complete: %d results, total_latency=%.1fms",
        len(final_results),
        latencies["total_ms"],
    )

    return ICDMapperResult(
        query_text=query_text,
        query_type=detected_type,
        top_k=top_k,
        results=final_results,
        latencies=latencies,
    )
