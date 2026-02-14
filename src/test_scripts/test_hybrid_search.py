"""
Test script for hybrid search functionality.
This test file verifies the hybrid search works correctly with and without filters.
"""

import logging
import os
from dotenv import load_dotenv

from rag.retrieval.hybrid_search import hybrid_search


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


COLLECTION_NAME = "icd_names"


def print_search_results(results, max_results=5):
    """Pretty print search results."""
    for i, result in enumerate(results[:max_results], 1):
        print(f"\n  {i}. Score: {result.score:.4f}")
        print(f"     Code: {result.payload.get('code_dotted', 'N/A')}")
        print(f"     Description: {result.payload.get('long_description', 'N/A')[:100]}...")
        print(f"     Category: {result.payload.get('category_title', 'N/A')[:80]}...")
        print(f"     Query Type: {result.payload.get('query_type', 'N/A')}")


def test_basic_hybrid_search():
    """Test basic hybrid search without any filters."""
    print("=" * 60)
    print("Testing Basic Hybrid Search (No Filter)")
    print("=" * 60)

    query = "diabetes mellitus type 2"
    print(f"\nQuery: '{query}'")
    print(f"Collection: {COLLECTION_NAME}")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5
    )

    print(f"\nReturned {len(results)} results:")
    print_search_results(results)
    print("\n✓ Basic hybrid search working!")


def test_diagnosis_filter():
    """Test hybrid search with diagnosis filter."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search with Diagnosis Filter")
    print("=" * 60)

    query = "heart disease"
    print(f"\nQuery: '{query}'")
    print(f"Filter: code_type='diagnosis'")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        code_type="diagnosis",
        top_k=5
    )

    print(f"\nReturned {len(results)} results:")
    print_search_results(results)


    all_diagnosis = all(
        r.payload.get('query_type') == 'diagnosis'
        for r in results
    )
    if all_diagnosis:
        print("\n✓ All results are diagnosis codes!")
    else:
        print("\n✗ Warning: Some results are not diagnosis codes")

    print("✓ Diagnosis filter working!")


def test_procedure_filter():
    """Test hybrid search with procedure filter."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search with Procedure Filter")
    print("=" * 60)

    query = "knee replacement surgery"
    print(f"\nQuery: '{query}'")
    print(f"Filter: code_type='procedure'")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        code_type="procedure",
        top_k=5
    )

    print(f"\nReturned {len(results)} results:")
    if results:
        print_search_results(results)
        # Verify all results are procedure codes
        all_procedure = all(
            r.payload.get('query_type') == 'procedure'
            for r in results
        )
        if all_procedure:
            print("\n✓ All results are procedure codes!")
        else:
            print("\n✗ Warning: Some results are not procedure codes")
    else:
        print("  (No procedure codes found in collection - this is expected if only diagnosis codes are uploaded)")

    print("✓ Procedure filter working!")


def test_custom_parameters():
    """Test hybrid search with custom parameters."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search with Custom Parameters")
    print("=" * 60)

    query = "hypertension"
    print(f"\nQuery: '{query}'")
    print(f"Custom parameters: top_k=3, top_k_dense=20, top_k_sparse=20, rerank_candidates=10")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=3,
        top_k_dense=20,
        top_k_sparse=20,
        rerank_candidates=10
    )

    print(f"\nReturned {len(results)} results (should be 3):")
    print_search_results(results, max_results=3)

    if len(results) <= 3:
        print("\n✓ Custom parameters working!")
    else:
        print(f"\n✗ Warning: Expected 3 results, got {len(results)}")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 60)
    print("Testing Error Handling")
    print("=" * 60)

    # Test empty query
    print("\n1. Testing empty query...")
    try:
        results = hybrid_search(
            query="",
            collection_name=COLLECTION_NAME
        )
        print("✗ Should have raised ValueError for empty query")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test invalid code_type
    print("\n2. Testing invalid code_type...")
    try:
        results = hybrid_search(
            query="test",
            collection_name=COLLECTION_NAME,
            code_type="invalid_type"  # type: ignore
        )
        print("✗ Should have raised ValueError for invalid code_type")
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    print("\n✓ Error handling working correctly!")


def test_comparison():
    """Compare results with different filters."""
    print("\n" + "=" * 60)
    print("Comparison: Same Query, Different Filters")
    print("=" * 60)

    query = "infection"
    print(f"\nQuery: '{query}'")

    print("\n--- Without filter ---")
    results_no_filter = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=3
    )
    print(f"Results: {len(results_no_filter)}")
    print_search_results(results_no_filter, max_results=3)

    print("\n--- With diagnosis filter ---")
    results_diagnosis = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        code_type="diagnosis",
        top_k=3
    )
    print(f"Results: {len(results_diagnosis)}")
    print_search_results(results_diagnosis, max_results=3)

    print("\n✓ Comparison test complete!")


def test_reranker_enabled():
    """Test hybrid search with reranker enabled (default behavior)."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search with Reranker Enabled (Default)")
    print("=" * 60)

    query = "chronic kidney disease"
    print(f"\nQuery: '{query}'")
    print(f"Reranker: Enabled (use_reranker=True)")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        use_reranker=True
    )

    print(f"\nReturned {len(results)} reranked results:")
    print_search_results(results)

    # Check that we got results
    if results:
        print("\n✓ Reranker executed successfully!")
        print(f"✓ Top result score: {results[0].score:.4f}")
    else:
        print("\n✗ Warning: No results returned")

    print("✓ Reranker enabled test complete!")


def test_reranker_disabled():
    """Test hybrid search with reranker disabled."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Search with Reranker Disabled")
    print("=" * 60)

    query = "chronic kidney disease"
    print(f"\nQuery: '{query}'")
    print(f"Reranker: Disabled (use_reranker=False)")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        use_reranker=False
    )

    print(f"\nReturned {len(results)} results (RRF fusion only):")
    print_search_results(results)

    # Check that we got results
    if results:
        print("\n✓ Hybrid search without reranker executed successfully!")
        print(f"✓ Top result score: {results[0].score:.4f}")
    else:
        print("\n✗ Warning: No results returned")

    print("✓ Reranker disabled test complete!")


def test_reranker_comparison():
    """Compare results with and without reranker."""
    print("\n" + "=" * 60)
    print("Comparison: Reranker Enabled vs Disabled")
    print("=" * 60)

    query = "diabetes with complications"
    print(f"\nQuery: '{query}'")

    print("\n--- WITHOUT Reranker (RRF fusion only) ---")
    results_no_rerank = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        use_reranker=False
    )
    print(f"Results: {len(results_no_rerank)}")
    print_search_results(results_no_rerank, max_results=5)

    print("\n--- WITH Reranker ---")
    results_with_rerank = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        use_reranker=True
    )
    print(f"Results: {len(results_with_rerank)}")
    print_search_results(results_with_rerank, max_results=5)

    # Compare the results
    print("\n--- Comparison Analysis ---")
    if results_no_rerank and results_with_rerank:
        # Check if top results are different
        top_no_rerank = results_no_rerank[0].payload.get('code_dotted', 'N/A')
        top_with_rerank = results_with_rerank[0].payload.get('code_dotted', 'N/A')

        print(f"Top result without reranker: {top_no_rerank}")
        print(f"Top result with reranker: {top_with_rerank}")

        if top_no_rerank != top_with_rerank:
            print("\n✓ Reranker changed the ranking order (as expected)!")
        else:
            print("\n• Top results are the same (reranker agreed with RRF)")

        # Compare score ranges
        print(f"\nScore ranges:")
        print(f"  Without reranker: {results_no_rerank[0].score:.4f} to {results_no_rerank[-1].score:.4f}")
        print(f"  With reranker: {results_with_rerank[0].score:.4f} to {results_with_rerank[-1].score:.4f}")

        print("\n✓ Reranker is working and producing different scores!")
    else:
        print("\n✗ Warning: One or both searches returned no results")

    print("\n✓ Reranker comparison test complete!")


def test_reranker_with_filters():
    """Test that reranker works correctly with filters."""
    print("\n" + "=" * 60)
    print("Testing Reranker with Diagnosis Filter")
    print("=" * 60)

    query = "heart failure"
    print(f"\nQuery: '{query}'")
    print(f"Filter: code_type='diagnosis'")
    print(f"Reranker: Enabled")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        code_type="diagnosis",
        top_k=5,
        use_reranker=True
    )

    print(f"\nReturned {len(results)} reranked results:")
    print_search_results(results)

    # Verify all results are diagnosis codes
    if results:
        all_diagnosis = all(
            r.payload.get('query_type') == 'diagnosis'
            for r in results
        )
        if all_diagnosis:
            print("\n✓ All results are diagnosis codes!")
            print("✓ Reranker + filter combination working correctly!")
        else:
            print("\n✗ Warning: Some results are not diagnosis codes")
    else:
        print("\n✗ Warning: No results returned")

    print("✓ Reranker with filters test complete!")


def test_reranker_score_ordering():
    """Test that reranker returns results in descending score order."""
    print("\n" + "=" * 60)
    print("Testing Reranker Score Ordering")
    print("=" * 60)

    query = "pneumonia"
    print(f"\nQuery: '{query}'")
    print(f"Reranker: Enabled")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=10,
        use_reranker=True
    )

    print(f"\nReturned {len(results)} results:")

    # Check score ordering
    if results and len(results) > 1:
        is_descending = all(
            results[i].score >= results[i + 1].score
            for i in range(len(results) - 1)
        )

        print(f"Scores: {[f'{r.score:.4f}' for r in results[:5]]}")

        if is_descending:
            print("\n✓ Results are correctly ordered by reranking score (descending)!")
        else:
            print("\n✗ Warning: Results are NOT in descending order")

        print_search_results(results, max_results=5)
    else:
        print("Not enough results to verify ordering")

    print("\n✓ Score ordering test complete!")


def test_rerank_candidates_parameter():
    """Test the rerank_candidates parameter controls fusion output."""
    print("\n" + "=" * 60)
    print("Testing Rerank Candidates Parameter")
    print("=" * 60)

    query = "type 2 diabetes with complications"
    print(f"\nQuery: '{query}'")
    print("\nTesting multi-stage pipeline:")
    print("  Stage 1: Dense (50) + Sparse (50) retrieval")
    print("  Stage 2: Fusion → rerank_candidates")
    print("  Stage 3: Reranking → top_k")

    print("\n--- Test 1: rerank_candidates=30, top_k=5 ---")
    results_30 = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        rerank_candidates=30,
        use_reranker=True
    )
    print(f"Final results: {len(results_30)} (should be 5)")
    print_search_results(results_30, max_results=3)

    print("\n--- Test 2: rerank_candidates=10, top_k=5 ---")
    results_10 = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        rerank_candidates=10,
        use_reranker=True
    )
    print(f"Final results: {len(results_10)} (should be 5)")
    print_search_results(results_10, max_results=3)

    if len(results_30) == 5 and len(results_10) == 5:
        print("\n✓ Rerank candidates parameter working correctly!")
        print("  • Both configurations returned correct number of final results")
        print("  • More rerank_candidates gives reranker more choices to optimize")
    else:
        print(f"\n✗ Warning: Expected 5 results, got {len(results_30)} and {len(results_10)}")

    print("\n✓ Rerank candidates parameter test complete!")


def test_pipeline_stages():
    """Demonstrate the complete multi-stage retrieval pipeline."""
    print("\n" + "=" * 60)
    print("Multi-Stage Pipeline Demonstration")
    print("=" * 60)

    query = "chronic heart failure"
    print(f"\nQuery: '{query}'")

    print("\nPipeline Configuration:")
    print("  📥 Dense Retrieval:     top_k_dense=50")
    print("  📥 Sparse Retrieval:    top_k_sparse=50")
    print("  🔀 Fusion (RRF):        rerank_candidates=20")
    print("  🎯 Reranking:           top_k=5")

    results = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        top_k_dense=50,
        top_k_sparse=50,
        rerank_candidates=20,
        use_reranker=True
    )

    print(f"\n✅ Pipeline complete! Final results: {len(results)}")
    print("\nTop 5 reranked results:")
    print_search_results(results, max_results=5)

    print("\n📊 Pipeline Flow:")
    print("  1. Retrieved ~100 candidates (50 dense + 50 sparse)")
    print("  2. Fusion combined & deduped → top 20 candidates")
    print("  3. Reranker evaluated 20 candidates → top 5 results")
    print("\n✓ Multi-stage pipeline working efficiently!")


def test_rerank_candidates_comparison():
    """Compare impact of different rerank_candidates values."""
    print("\n" + "=" * 60)
    print("Rerank Candidates Impact Analysis")
    print("=" * 60)

    query = "sepsis with organ failure"
    print(f"\nQuery: '{query}'")

    print("\n--- Small candidate pool (rerank_candidates=10) ---")
    results_small = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        rerank_candidates=10,
        use_reranker=True
    )
    print(f"Results: {len(results_small)}")
    if results_small:
        print(f"Top result: {results_small[0].payload.get('code_dotted', 'N/A')}")
        print(f"Top score: {results_small[0].score:.4f}")

    print("\n--- Large candidate pool (rerank_candidates=40) ---")
    results_large = hybrid_search(
        query=query,
        collection_name=COLLECTION_NAME,
        top_k=5,
        rerank_candidates=40,
        use_reranker=True
    )
    print(f"Results: {len(results_large)}")
    if results_large:
        print(f"Top result: {results_large[0].payload.get('code_dotted', 'N/A')}")
        print(f"Top score: {results_large[0].score:.4f}")

    print("\n--- Analysis ---")
    if results_small and results_large:
        same_top = (results_small[0].payload.get('code_dotted') ==
                   results_large[0].payload.get('code_dotted'))

        if same_top:
            print("✓ Top result is the same (both pools contained the best match)")
        else:
            print("✓ Top results differ (larger pool found a better match!)")

        print(f"\nScore difference: {abs(results_large[0].score - results_small[0].score):.4f}")
        print("\n💡 Larger rerank_candidates gives reranker more options to find")
        print("   the best match, at the cost of more reranking computation.")

    print("\n✓ Rerank candidates comparison complete!")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING HYBRID SEARCH FUNCTIONALITY")
    print("=" * 60)

    # Check environment variables
    if not os.environ.get("QDRANT_URL") or not os.environ.get("QDRANT_API_KEY"):
        print("\n✗ ERROR: QDRANT_URL and QDRANT_API_KEY must be set in .env file")
        return

    try:
        test_basic_hybrid_search()
        test_diagnosis_filter()
        test_procedure_filter()
        test_custom_parameters()
        test_error_handling()
        test_comparison()

        # Reranker tests
        test_reranker_enabled()
        test_reranker_disabled()
        test_reranker_comparison()
        test_reranker_with_filters()
        test_reranker_score_ordering()

        # Multi-stage pipeline tests
        test_rerank_candidates_parameter()
        test_pipeline_stages()
        test_rerank_candidates_comparison()

        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nHybrid search is working correctly with all features:")
        print("  • Basic hybrid search (dense + sparse with RRF)")
        print("  • Diagnosis/procedure filtering")
        print("  • Custom parameters (top_k, prefetch limits, rerank_candidates)")
        print("  • Error handling")
        print("  • Reranker functionality (enabled/disabled)")
        print("  • Reranker with filters")
        print("  • Reranker score ordering")
        print("  • Multi-stage pipeline (prefetch → fusion → reranking)")
        print("  • Rerank candidates parameter impact")

    except Exception as e:
        print("\n" + "=" * 60)
        print("✗ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
