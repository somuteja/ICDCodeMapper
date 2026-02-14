import logging
from rag.embeddings.embedding_manager import embedding_manager

logger = logging.getLogger(__name__)


def generate_dense_embedding(text: str) -> list[float]:
    """
    Generate a dense embedding for a single string.

    Args:
        text: The input string to generate embedding for.

    Returns:
        A list of floats containing the dense embedding vector.
    """
    logger.info(f"Generating dense embedding for single text (length: {len(text)} chars)")
    embeddings = list(embedding_manager.dense_model.embed([text]))
    return embeddings[0].tolist()


def generate_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generate dense embeddings for a list of strings.

    Args:
        texts: List of input strings to generate embeddings for.

    Returns:
        A list of lists of floats containing the dense embedding vectors.
    """
    logger.info(f"Generating dense embeddings for {len(texts)} texts")
    return [emb.tolist() for emb in embedding_manager.dense_model.embed(texts)]
