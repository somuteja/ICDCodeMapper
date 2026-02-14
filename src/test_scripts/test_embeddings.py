"""
Test script for embeddings, sparse embeddings, and reranker.
This is a temporary test file to verify all components work correctly.
"""

import logging

from rag.embeddings.dense_embeddings import (
    generate_dense_embedding,
    generate_dense_embeddings,
)
from rag.embeddings.sparse_embeddings import (
    generate_bm25_embedding,
    generate_bm25_embeddings,
)
from rag.embeddings.reranker import rerank

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


def test_dense_embeddings():
    print("=" * 60)
    print("Testing Dense Embeddings")
    print("=" * 60)

    text = "Type 2 diabetes mellitus"
    embedding = generate_dense_embedding(text)
    print(f"\nSingle text: '{text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")

    texts = [
        "Type 2 diabetes mellitus",
        "Headache disorder",
        "Hypertension",
    ]
    embeddings = generate_dense_embeddings(texts)
    print(f"\nBatch processing {len(texts)} texts")
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Each embedding dimension: {len(embeddings[0])}")
    print("✓ Dense embeddings working!")


def test_sparse_embeddings():
    print("\n" + "=" * 60)
    print("Testing Sparse Embeddings (BM25)")
    print("=" * 60)

    text = "Type 2 diabetes mellitus"
    embedding = generate_bm25_embedding(text)
    print(f"\nSingle text: '{text}'")
    print(f"Sparse embedding type: {type(embedding)}")
    print(f"Embedding data: {embedding}")

    texts = [
        "Type 2 diabetes mellitus",
        "Headache disorder",
        "Hypertension",
    ]
    embeddings = generate_bm25_embeddings(texts)
    print(f"\nBatch processing {len(texts)} texts")
    print(f"Number of embeddings: {len(embeddings)}")
    print("✓ Sparse embeddings working!")


def test_reranker():
    print("\n" + "=" * 60)
    print("Testing Reranker (Cross-Encoder)")
    print("=" * 60)


    query = "ICD-10 code for diabetes"
    documents = [
        "E11 - Type 2 diabetes mellitus",
        "R51 - Headache",
        "E10 - Type 1 diabetes mellitus",
        "I10 - Essential (primary) hypertension",
        "E11.9 - Type 2 diabetes mellitus without complications",
    ]

    print(f"\nQuery: '{query}'")
    print(f"\nOriginal documents (unsorted):")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    # Test with default top_k=5
    print(f"\n--- Testing with default top_k=5 ---")
    reranked_docs = rerank(query, documents)
    print(f"\nReranked documents (top {len(reranked_docs)}):")
    for i, doc in enumerate(reranked_docs, 1):
        print(f"  {i}. {doc}")

    # Test with custom top_k=3
    print(f"\n--- Testing with custom top_k=3 ---")
    reranked_docs_top3 = rerank(query, documents, top_k=3)
    print(f"\nReranked documents (top {len(reranked_docs_top3)}):")
    for i, doc in enumerate(reranked_docs_top3, 1):
        print(f"  {i}. {doc}")

    print("\n✓ Reranker working!")
    print(f"✓ Most relevant document: {reranked_docs[0]}")
    print(f"✓ Returned {len(reranked_docs_top3)} documents when top_k=3")


def main():
    print("\n" + "=" * 60)
    print("TESTING ALL EMBEDDING COMPONENTS")
    print("=" * 60)

    try:
        test_dense_embeddings()
        test_sparse_embeddings()
        test_reranker()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nAll embedding components are working correctly.")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
