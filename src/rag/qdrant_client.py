import os
from typing import Optional

from qdrant_client import QdrantClient


_client: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Return a singleton QdrantClient instance.

    Reads from environment variables:
        - QDRANT_URL (required)
        - QDRANT_API_KEY (required)

    Raises:
        RuntimeError: If required environment variables are missing.
    """
    global _client
    if _client is None:
        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")
        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set")
        _client = QdrantClient(url=url, api_key=api_key)
    return _client
