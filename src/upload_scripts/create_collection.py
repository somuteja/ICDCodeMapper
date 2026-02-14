"""
Script to create the Qdrant collection for ICD names with hybrid search support.
"""
import logging
import os
from pathlib import Path
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_icd_names_collection():
    """
    Create a Qdrant collection named 'icd_names' with:
    - Dense vector: 1024 dimensions (BAAI/bge-large-en-v1.5)
    - Sparse vector: BM25 for keyword matching
    """

    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

    client = QdrantClient(url=url, api_key=api_key)

    collection_name = "icd_names"

    collections = client.get_collections().collections
    if any(col.name == collection_name for col in collections):
        logger.warning(f"Collection '{collection_name}' already exists.")
        response = input("Do you want to recreate it? This will delete all existing data. (yes/no): ")
        if response.lower() != "yes":
            logger.info("Aborting collection creation.")
            return
        logger.info(f"Deleting existing collection '{collection_name}'...")
        client.delete_collection(collection_name)


    logger.info(f"Creating collection '{collection_name}'...")
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(
                size=1024,  
                distance=Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse_bm25": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False  
                )
            )
        }
    )

    logger.info(f"Collection '{collection_name}' created successfully!")
    logger.info(f"Dense vector 'dense': 1024 dimensions, COSINE distance")
    logger.info(f"Sparse vector 'sparse_bm25': BM25 indexing")

    # Create payload index for query_type field to enable filtering
    logger.info("Creating payload index for 'query_type' field...")
    client.create_payload_index(
        collection_name=collection_name,
        field_name="query_type",
        field_schema=PayloadSchemaType.KEYWORD
    )
    logger.info("✓ Payload index created for 'query_type' field")

    collection_info = client.get_collection(collection_name)
    logger.info(f"\nCollection info:")
    logger.info(f"  Points count: {collection_info.points_count}")
    logger.info(f"  Vectors: {collection_info.config.params.vectors}")
    logger.info(f"  Sparse vectors: {collection_info.config.params.sparse_vectors}")


if __name__ == "__main__":
    create_icd_names_collection()
