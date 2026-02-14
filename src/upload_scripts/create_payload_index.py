"""
Script to create a payload index on the 'query_type' field in an existing collection.
This allows filtering by diagnosis/procedure type in hybrid search.
"""
import logging
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_query_type_index(collection_name="icd_names"):
    """
    Create a payload index on the 'query_type' field for filtering.

    This is required to filter search results by diagnosis vs procedure codes.
    """
    url = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")

    if not url or not api_key:
        raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

    client = QdrantClient(url=url, api_key=api_key)

    collections = client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        logger.error(f"Collection '{collection_name}' does not exist!")
        logger.error("Please create the collection first using create_collection.py")
        return

    logger.info(f"Creating payload index for collection '{collection_name}'...")

    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="query_type",
            field_schema=PayloadSchemaType.KEYWORD
        )
        logger.info("✓ Payload index created successfully for 'query_type' field")
        logger.info("You can now filter by diagnosis/procedure in hybrid search!")

    except Exception as e:
        if "already exists" in str(e).lower():
            logger.info("✓ Index already exists for 'query_type' field")
        else:
            logger.error(f"Failed to create index: {e}")
            raise

    
    collection_info = client.get_collection(collection_name)
    logger.info(f"\nCollection '{collection_name}' status:")
    logger.info(f"  Points count: {collection_info.points_count}")
    logger.info(f"  Payload indexes: {collection_info.payload_schema}")


if __name__ == "__main__":
    create_query_type_index()
