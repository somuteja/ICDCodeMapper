"""Hybrid search implementation using dense and sparse vector search with RRF fusion."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from rag.embeddings.embedding_manager import embedding_manager
from rag.qdrant_client import get_qdrant_client
from rag.retrieval.constants import (
    CODE_TYPE_DIAGNOSIS,
    CODE_TYPE_PROCEDURE,
    DEFAULT_DENSE_VECTOR_NAME,
    DEFAULT_PREFETCH_DENSE,
    DEFAULT_PREFETCH_SPARSE,
    DEFAULT_RERANK_CANDIDATES,
    DEFAULT_SPARSE_VECTOR_NAME,
    DEFAULT_TOP_K,
    FILTER_KEY_QUERY_TYPE,
    FUSION_METHOD,
)

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Single search result with score and metadata."""

    id: str
    score: float
    payload: Dict[str, Any]


def _format_query_for_reranking(query: str) -> str:
    """Format query for reranking with special tags.

    Args:
        query: The original user query.

    Returns:
        Formatted query string with tags.
    """
    return f"This is the user query: <query>{query}</query>"


def _format_documents_for_reranking(results: List[SearchResult]) -> List[str]:
    """Convert SearchResult payloads to document strings for reranking.

    Args:
        results: List of SearchResult objects.

    Returns:
        List of formatted document strings.
    """
    documents = []
    for result in results:
        payload = result.payload
        doc_parts = []

        if "code" in payload:
            doc_parts.append(f"Code: {payload['code']}")
        if "description" in payload:
            doc_parts.append(f"Description: {payload['description']}")
        if "long_description" in payload:
            doc_parts.append(f"Details: {payload['long_description']}")
        if "category" in payload:
            doc_parts.append(f"Category: {payload['category']}")

    
        if not doc_parts:
            doc_parts.append(str(payload))

        documents.append(" | ".join(doc_parts))

    return documents


def _apply_reranking(
    query: str, results: List[SearchResult], top_k: int
) -> List[SearchResult]:
    """Apply reranking to search results.

    Args:
        query: The original user query.
        results: List of SearchResult objects to rerank.
        top_k: Number of top results to return after reranking.

    Returns:
        List of reranked SearchResult objects with updated scores.
    """
    if not results:
        return results

    formatted_query = _format_query_for_reranking(query)
    documents = _format_documents_for_reranking(results)

    try:
        scores = list(embedding_manager.rerank_model.rerank(formatted_query, documents))
    except Exception as exc:
        logger.error("Reranking failed: %s", exc)
        return results[:top_k]

    result_score_pairs = list(zip(results, scores))
    sorted_pairs = sorted(result_score_pairs, key=lambda x: x[1], reverse=True)

    reranked_results = []
    for result, new_score in sorted_pairs[:top_k]:
        reranked_result = SearchResult(
            id=result.id,
            score=new_score,
            payload=result.payload,
        )
        reranked_results.append(reranked_result)

    logger.info(
        f"Reranking completed: {len(results)} candidates -> {len(reranked_results)} results"
    )

    return reranked_results


def hybrid_search(
    query: str,
    collection_name: str,
    dense_vector_name: str = DEFAULT_DENSE_VECTOR_NAME,
    sparse_vector_name: str = DEFAULT_SPARSE_VECTOR_NAME,
    *,
    top_k: int = DEFAULT_TOP_K,
    top_k_dense: int = DEFAULT_PREFETCH_DENSE,
    top_k_sparse: int = DEFAULT_PREFETCH_SPARSE,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    code_type: Optional[Literal["diagnosis", "procedure"]] = None,
    filter_key: Optional[str] = None,
    filter_value: Optional[Any] = None,
    use_reranker: bool = True,
) -> List[SearchResult]:
    """Perform hybrid search using Qdrant's built-in RRF fusion.

    Args:
        query: The search query text.
        collection_name: Name of the Qdrant collection to search.
        dense_vector_name: Name of the dense vector field in the collection.
            Defaults to "dense_vector".
        sparse_vector_name: Name of the sparse vector field in the collection.
            Defaults to "sparse_vector".
        top_k: Number of final results to return.
        top_k_dense: Number of results to prefetch from dense search.
        top_k_sparse: Number of results to prefetch from sparse search.
        rerank_candidates: Number of candidates after fusion to send to reranker.
            Only used when use_reranker=True. Should be larger than top_k.
        code_type: Optional filter for ICD code type. If specified, filters results
            to only "diagnosis" or "procedure" codes. This is a convenience parameter
            that sets filter_key="code_type" and filter_value to the specified type.
        filter_key: Optional payload field name for custom metadata filtering.
            If code_type is specified, this parameter is ignored.
        filter_value: Optional value to match for the filter_key field.
            If code_type is specified, this parameter is ignored.
        use_reranker: Whether to apply reranking to the results. When True (default),
            results are reranked using a cross-encoder model for improved relevance.
            The query is formatted with special tags for reranking.

    Returns:
        List of SearchResult objects sorted by score in descending order.
        If use_reranker is True, scores represent reranking scores; otherwise,
        they represent RRF fusion scores.

    Raises:
        ValueError: If query is empty or invalid code_type is provided.
        RuntimeError: If embedding generation or search fails.

    Examples:
        >>> # Search for diagnosis codes only
        >>> results = hybrid_search(
        ...     query="diabetes mellitus",
        ...     collection_name="icd_codes",
        ...     code_type="diagnosis"
        ... )

        >>> # Search for procedure codes only
        >>> results = hybrid_search(
        ...     query="knee replacement",
        ...     collection_name="icd_codes",
        ...     code_type="procedure"
        ... )

        >>> # Search without filtering
        >>> results = hybrid_search(
        ...     query="heart disease",
        ...     collection_name="icd_codes"
        ... )

        >>> # Custom metadata filtering
        >>> results = hybrid_search(
        ...     query="infection",
        ...     collection_name="icd_codes",
        ...     filter_key="category",
        ...     filter_value="infectious_disease"
        ... )

        >>> # Search without reranking
        >>> results = hybrid_search(
        ...     query="heart disease",
        ...     collection_name="icd_codes",
        ...     use_reranker=False
        ... )
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if code_type is not None and code_type not in ("diagnosis", "procedure"):
        raise ValueError(
            f"Invalid code_type '{code_type}'. Must be 'diagnosis' or 'procedure'."
        )

    query = query.strip()

    try:
        dense_vector = list(embedding_manager.get_dense_embeddings([query])[0])
    except Exception as exc:
        logger.error("Dense embedding failed: %s", exc)
        raise RuntimeError(f"Dense embedding failed: {exc}") from exc

    try:
        sparse_emb = embedding_manager.get_sparse_embeddings([query])[0]
        sparse_vector = SparseVector(
            indices=sparse_emb.indices.tolist(),
            values=sparse_emb.values.tolist(),
        )
    except Exception as exc:
        logger.error("Sparse embedding failed: %s", exc)
        raise RuntimeError(f"Sparse embedding failed: {exc}") from exc

    query_filter = None
    if code_type is not None:
        filter_value_to_use = (
            CODE_TYPE_DIAGNOSIS if code_type == "diagnosis" else CODE_TYPE_PROCEDURE
        )
        query_filter = Filter(
            must=[
                FieldCondition(
                    key=FILTER_KEY_QUERY_TYPE,
                    match=MatchValue(value=filter_value_to_use),
                )
            ]
        )
        logger.debug(f"Applying code_type filter: {code_type}")
    elif filter_key is not None and filter_value is not None:
        query_filter = Filter(
            must=[FieldCondition(key=filter_key, match=MatchValue(value=filter_value))]
        )
        logger.debug(f"Applying custom filter: {filter_key}={filter_value}")

    client = get_qdrant_client()

    fusion_limit = rerank_candidates if use_reranker else top_k

    try:
        results = client.query_points(
            collection_name=collection_name,
            prefetch=[
                Prefetch(
                    query=dense_vector,
                    using=dense_vector_name,
                    limit=top_k_dense,
                    filter=query_filter,
                ),
                Prefetch(
                    query=sparse_vector,
                    using=sparse_vector_name,
                    limit=top_k_sparse,
                    filter=query_filter,
                ),
            ],
            query=FusionQuery(fusion=FUSION_METHOD),
            limit=fusion_limit,
            with_payload=True,
        )
    except Exception as exc:
        logger.error("Hybrid search failed: %s", exc)
        raise RuntimeError(f"Hybrid search failed: {exc}") from exc

    logger.info(
        f"Hybrid search completed: query='{query[:50]}...', "
        f"results={len(results.points)}, filter={code_type or 'none'}"
    )

    search_results = [
        SearchResult(
            id=str(point.id),
            score=point.score,
            payload=point.payload or {},
        )
        for point in results.points
    ]

    if use_reranker:
        return _apply_reranking(query, search_results, top_k)

    return search_results
