"""
Script to upload ICD-10 codes to Qdrant collection with hybrid search.
Uploads first 1000 records in batches of 10 with detailed logging.
Includes robust error handling, retry logic, and failure tracking.
"""
import csv
import json
import logging
import os
import time
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from fastembed import TextEmbedding, SparseTextEmbedding

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ICDUploader:
    def __init__(self, collection_name="icd_names"):
        """Initialize the uploader with Qdrant client and embedding models"""
        self.collection_name = collection_name

        url = os.environ.get("QDRANT_URL")
        api_key = os.environ.get("QDRANT_API_KEY")

        if not url or not api_key:
            raise RuntimeError("QDRANT_URL and QDRANT_API_KEY must be set in .env file")

        logger.info("Connecting to Qdrant...")
        self.client = QdrantClient(url=url, api_key=api_key)
        logger.info("✓ Connected to Qdrant successfully")

        logger.info("Loading dense embedding model (BAAI/bge-large-en-v1.5)...")
        self.dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        logger.info("✓ Dense model loaded")

        logger.info("Loading sparse embedding model (BM25)...")
        self.sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")
        logger.info("✓ Sparse model loaded")

    def read_preprocessed_data(self, csv_file, limit=1000, offset=0):
        """Read preprocessed CSV data"""
        logger.info(f"Reading data from {csv_file} (offset: {offset}, limit: {limit})...")

        data = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < offset:
                    continue
                if i >= offset + limit:
                    break
                data.append(row)

        logger.info(f"✓ Loaded {len(data)} records (rows {offset}-{offset + len(data) - 1})")
        return data

    def create_embeddings(self, texts):
        """Generate dense and sparse embeddings for texts"""
        # Dense embeddings
        dense_embeddings = list(self.dense_model.embed(texts))

        # Sparse embeddings
        sparse_embeddings = list(self.sparse_model.embed(texts))

        return dense_embeddings, sparse_embeddings

    def upload_batch(self, batch_data, batch_num, start_id, max_retries=3):
        """Upload a single batch to Qdrant with retry logic"""
        batch_start_time = time.time()

        for attempt in range(max_retries):
            try:
                # Extract texts for embedding
                texts = [record['embedded_text'] for record in batch_data]

                # Generate embeddings
                logger.info(f"  Generating embeddings for batch {batch_num}...")
                dense_embeddings, sparse_embeddings = self.create_embeddings(texts)

                # Create points
                points = []
                for i, record in enumerate(batch_data):
                    point_id = start_id + i

                    # Convert sparse embedding to required format
                    sparse_vector = sparse_embeddings[i]

                    point = PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_embeddings[i].tolist() if hasattr(dense_embeddings[i], 'tolist') else list(dense_embeddings[i]),
                            "sparse_bm25": {
                                "indices": sparse_vector.indices.tolist(),
                                "values": sparse_vector.values.tolist()
                            }
                        },
                        payload={
                            "code": record['code'],
                            "code_dotted": record['code_dotted'],
                            "long_description": record['long_description'],
                            "category_title": record['category_title'],
                            "embedded_text": record['embedded_text'],
                            "system": record['system'],
                            "query_type": record['query_type']
                        }
                    )
                    points.append(point)

                logger.info(f"  Uploading batch {batch_num} to Qdrant...")
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

                batch_time = time.time() - batch_start_time
                logger.info(f"  ✓ Batch {batch_num} uploaded successfully ({batch_time:.2f}s)")

                return batch_time, None  # Success, no error

            except Exception as e:
                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(f"  ⚠ Batch {batch_num} failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    logger.warning(f"  Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"  ✗ Batch {batch_num} failed after {max_retries} attempts: {str(e)}")
                    return None, str(e)  # Failed after all retries

        return None, "Unknown error"  # Should never reach here

    def upload_data(self, data, batch_size=10, start_id=0):
        """Upload data in batches with progress tracking and error handling"""
        total_records = len(data)
        total_batches = (total_records + batch_size - 1) // batch_size

        logger.info(f"\n{'='*60}")
        logger.info(f"Starting upload process")
        logger.info(f"Total records: {total_records}")
        logger.info(f"Starting ID: {start_id}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"{'='*60}\n")

        start_time = time.time()
        total_batch_time = 0
        successful_batches = 0
        failed_batches = []

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, total_records)
            batch_data = data[start_idx:end_idx]

            logger.info(f"Batch {batch_num + 1}/{total_batches} (records {start_idx + 1}-{end_idx}):")

            batch_time, error = self.upload_batch(batch_data, batch_num + 1, start_id + start_idx)

            if error is None:
                # Success
                total_batch_time += batch_time
                successful_batches += 1
            else:
                # Failed - track it and continue
                failed_batches.append({
                    'batch_num': batch_num + 1,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'start_id': start_id + start_idx,
                    'error': error,
                    'records': len(batch_data)
                })

            progress = ((batch_num + 1) / total_batches) * 100
            logger.info(f"  Progress: {progress:.1f}% complete (Success: {successful_batches}, Failed: {len(failed_batches)})\n")

        total_time = time.time() - start_time

        logger.info(f"\n{'='*60}")
        logger.info(f"Upload process completed!")
        logger.info(f"{'='*60}")
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Successful batches: {successful_batches}")
        logger.info(f"Failed batches: {len(failed_batches)}")
        logger.info(f"Success rate: {(successful_batches/total_batches)*100:.1f}%")
        logger.info(f"Total time: {total_time:.2f}s")
        if successful_batches > 0:
            logger.info(f"Average time per successful batch: {total_batch_time/successful_batches:.2f}s")
        logger.info(f"{'='*60}\n")

        # Log failed batches if any
        if failed_batches:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"FAILED BATCHES SUMMARY ({len(failed_batches)} batches)")
            logger.warning(f"{'='*60}")
            for fb in failed_batches:
                logger.warning(f"Batch {fb['batch_num']}: Records {fb['start_idx'] + 1}-{fb['end_idx']} (IDs {fb['start_id']}-{fb['start_id'] + fb['records'] - 1})")
                logger.warning(f"  Error: {fb['error']}")
            logger.warning(f"{'='*60}\n")

            # Save failed batch info to file for retry
            self.save_failed_batches(failed_batches, start_id)

        # Verify upload
        self.verify_upload()

        return successful_batches, failed_batches

    def save_failed_batches(self, failed_batches, start_id):
        """Save failed batch information to a JSON file for retry"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"failed_batches_{timestamp}.json"

        failed_info = {
            'timestamp': timestamp,
            'start_id': start_id,
            'failed_batches': failed_batches
        }

        with open(filename, 'w') as f:
            json.dump(failed_info, f, indent=2)

        logger.info(f"Failed batch information saved to: {filename}")
        logger.info(f"You can use this file to retry only the failed batches")

    def verify_upload(self):
        """Verify the upload by checking collection stats"""
        logger.info("Verifying upload...")
        try:
            collection_info = self.client.get_collection(self.collection_name)
            logger.info(f"✓ Collection '{self.collection_name}' now has {collection_info.points_count} points")
        except Exception as e:
            logger.error(f"✗ Failed to verify upload: {str(e)}")


def main():
    """Main function to run the upload process"""
    csv_file = "icd_10_pcs_preprocessed.csv"

    OFFSET = 0
    LIMIT = 79115
    BATCH_SIZE = 10
    START_ID = 80000  # Diagnosis codes occupy IDs 0-71703; start procedures at 80000

    if not os.path.exists(csv_file):
        logger.error(f"File not found: {csv_file}")
        logger.error("Please run preprocess_icd10_pcs.py first to generate the preprocessed data")
        return 1

    try:
        uploader = ICDUploader()

        data = uploader.read_preprocessed_data(csv_file, limit=LIMIT, offset=OFFSET)

        successful_batches, failed_batches = uploader.upload_data(data, batch_size=BATCH_SIZE, start_id=START_ID)

        if len(failed_batches) == 0:
            logger.info("✓ All batches uploaded successfully!")
            return 0
        elif successful_batches > 0:
            logger.warning(f"⚠ Partial success: {successful_batches} succeeded, {len(failed_batches)} failed")
            return 2  
        else:
            logger.error("✗ All batches failed!")
            return 3  

    except KeyboardInterrupt:
        logger.warning("\n\nUpload interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Upload failed with unexpected error: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
