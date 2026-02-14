import logging
from fastembed.sparse.sparse_embedding_base import SparseEmbedding
from rag.embeddings.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)


def generate_bm25_embedding(text: str) -> SparseEmbedding:
    """
    Generate a BM25 sparse embedding for a single string.

    Args:
        text: The input string to generate embedding for.

    Returns:
        A sparse embedding object from fastembed.
    """
    logger.info(f"Generating BM25 sparse embedding for single text (length: {len(text)} chars)")
    embeddings = list(embedding_manager.sparse_model.embed([text]))
    return embeddings[0]


def generate_bm25_embeddings(texts: list[str]) -> list[SparseEmbedding]:
    """
    Generate BM25 sparse embeddings for a list of strings.

    Args:
        texts: List of input strings to generate embeddings for.

    Returns:
        A list of sparse embedding objects from fastembed.
    """
    logger.info(f"Generating BM25 sparse embeddings for {len(texts)} texts")
    return list(embedding_manager.sparse_model.embed(texts))
