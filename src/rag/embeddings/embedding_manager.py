import logging
from threading import Lock

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from rag.embeddings.constants import DENSE_MODEL, RERANK_MODEL, SPARSE_MODEL

logger = logging.getLogger(__name__)


class EmbeddingManager:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(f"Initializing embedding models...")
        self._dense_model = TextEmbedding(model_name=DENSE_MODEL)
        logger.info(f"Dense model loaded: {DENSE_MODEL}")
        self._sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)
        logger.info(f"Sparse model loaded: {SPARSE_MODEL}")
        self._rerank_model = TextCrossEncoder(model_name=RERANK_MODEL)
        logger.info(f"Rerank model loaded: {RERANK_MODEL}")
        self._initialized = True

    @property
    def dense_model(self) -> TextEmbedding:
        return self._dense_model

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        return self._sparse_model

    @property
    def rerank_model(self) -> TextCrossEncoder:
        return self._rerank_model

    def get_dense_embeddings(self, texts: list[str]) -> list[list[float]]:
        return list(self._dense_model.embed(texts))

    def get_sparse_embeddings(self, texts: list[str]):
        return list(self._sparse_model.embed(texts))


embedding_manager = EmbeddingManager()
