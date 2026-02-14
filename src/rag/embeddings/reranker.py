import logging
from rag.embeddings.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)


def rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """
    Rerank documents based on relevance to user query.

    Args:
        query: User query to compare against
        documents: List of documents to rerank
        top_k: Number of top documents to return (default: 5)

    Returns:
        List of top_k documents sorted by relevance (highest score first)
    """
    scores = list(embedding_manager.rerank_model.rerank(query, documents))

    doc_score_pairs = list(zip(documents, scores))
    sorted_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)

    logger.info(f"Reranking {len(documents)} documents for query: '{query}'")
    for i, (doc, score) in enumerate(sorted_pairs[:top_k], 1):
        logger.info(f"  Rank {i}: Score={score:.4f}, Doc preview: {doc[:100]}...")

    reranked_documents = [doc for doc, _ in sorted_pairs[:top_k]]
    return reranked_documents
