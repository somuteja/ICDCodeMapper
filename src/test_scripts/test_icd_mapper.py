"""
Test script for ICD code mapper orchestrator.

This demonstrates the complete pipeline from medical text to ICD-10 code mappings:
  - Query type detection (auto/diagnosis/procedure)
  - Hybrid search (dense + sparse + RRF + reranking)
  - Confidence scoring (LLM-based evaluation)
  - Result formatting with latency tracking
"""
import sys
from pathlib import Path


src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from core.icd_mapper import map_icd_codes, ICDMapperResult
from dotenv import load_dotenv
import json


load_dotenv()


def print_separator(title: str = ""):
    """Print a visual separator with optional title."""
    if title:
        print(f"\n{'=' * 80}")
        print(f"{title.center(80)}")
        print(f"{'=' * 80}")
    else:
        print(f"{'=' * 80}")


def print_result(result: ICDMapperResult):
    """Pretty print an ICD mapper result."""
    print(f"\n📝 QUERY: {result.query_text}")
    print(f"🏷️  DETECTED TYPE: {result.query_type.upper()}")
    print(f"🎯 REQUESTED TOP K: {result.top_k}")

    print(f"\n⏱️  LATENCIES:")
    for key, value in result.latencies.items():
        print(f"  • {key.replace('_', ' ').title()}: {value:.1f}ms")

    if not result.results:
        print("\n⚠️  NO RESULTS FOUND")
        return

    print(f"\n🔍 TOP {len(result.results)} RESULTS:")
    print(f"{'-' * 80}")

    for i, code_result in enumerate(result.results, 1):
        print(f"\n[{i}] {code_result['code_dotted']}")
        print(f"    Long Description: {code_result['long_description'][:80]}...")
        print(f"    Category: [{code_result['category_code']}] {code_result['category_title']}")
        print(f"    Relevance Score: {code_result['score']:.4f}")
        print(f"    Confidence: {code_result['confidence']}")


def test_auto_mode_diagnosis():
    """Test auto mode with diagnosis-type queries."""
    print_separator("TEST: AUTO MODE - DIAGNOSIS QUERIES")

    queries = [
        "Type 2 diabetes mellitus",
        "Acute myocardial infarction",
        "Chronic obstructive pulmonary disease",
        "Essential hypertension",
        "Pneumonia"
    ]

    for query in queries:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="auto",
                top_k=10
            )
            print_result(result)
            print_separator()
        except Exception as e:
            print(f"\n❌ ERROR for query '{query}': {e}")
            print_separator()


def test_auto_mode_procedure():
    """Test auto mode with procedure-type queries."""
    print_separator("TEST: AUTO MODE - PROCEDURE QUERIES")

    queries = [
        "Coronary artery bypass graft surgery",
        "Appendectomy",
        "Total knee replacement",
        "Cataract surgery",
        "Colonoscopy"
    ]

    for query in queries:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="auto",
                top_k=10
            )
            print_result(result)
            print_separator()
        except Exception as e:
            print(f"\n❌ ERROR for query '{query}': {e}")
            print_separator()


def test_explicit_diagnosis_mode():
    """Test with explicit diagnosis mode (no type detection)."""
    print_separator("TEST: EXPLICIT DIAGNOSIS MODE")

    queries = [
        "Diabetes",
        "Heart failure",
        "Asthma"
    ]

    for query in queries:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="diagnosis",
                top_k=10
            )
            print_result(result)

            if "type_detection_ms" in result.latencies:
                print("\n⚠️  WARNING: Type detection latency present in explicit mode!")
            else:
                print("\n✓ Confirmed: No type detection in explicit mode")

            print_separator()
        except Exception as e:
            print(f"\n❌ ERROR for query '{query}': {e}")
            print_separator()


def test_explicit_procedure_mode():
    """Test with explicit procedure mode (no type detection)."""
    print_separator("TEST: EXPLICIT PROCEDURE MODE")

    queries = [
        "Surgery on knee",
        "Blood transfusion",
        "X-ray of chest"
    ]

    for query in queries:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="procedure",
                top_k=10
            )
            print_result(result)

            if "type_detection_ms" in result.latencies:
                print("\n⚠️  WARNING: Type detection latency present in explicit mode!")
            else:
                print("\n✓ Confirmed: No type detection in explicit mode")

            print_separator()
        except Exception as e:
            print(f"\n❌ ERROR for query '{query}': {e}")
            print_separator()


def test_different_top_k_values():
    """Test with different top_k values."""
    print_separator("TEST: DIFFERENT TOP_K VALUES")

    query = "Chronic kidney disease"
    top_k_values = [1, 3, 5, 10]

    for k in top_k_values:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="auto",
                top_k=k
            )
            print(f"\n📊 TOP_K = {k}")
            print(f"   Results returned: {len(result.results)}")
            print(f"   Total latency: {result.latencies.get('total_ms', 0):.1f}ms")

            if result.results:
                print(f"   Best match: {result.results[0]['code_dotted']}")
                print(f"   Best score: {result.results[0]['score']:.4f}")
                print(f"   Best confidence: {result.results[0]['confidence']}")
        except Exception as e:
            print(f"\n❌ ERROR for top_k={k}: {e}")

    print_separator()


def test_edge_cases():
    """Test edge cases and error handling."""
    print_separator("TEST: EDGE CASES AND ERROR HANDLING")

    test_cases = [
        ("", "Empty string"),
        ("   ", "Whitespace only"),
        ("a", "Single character"),
        ("This is a very long medical query that describes a complex condition with multiple symptoms including fever, cough, shortness of breath, fatigue, and other various symptoms that might be relevant to the diagnosis", "Very long query"),
        ("COVID-19", "Abbreviation"),
        ("Type II DM", "Medical abbreviations")
    ]

    for query, description in test_cases:
        print(f"\n🧪 Testing: {description}")
        print(f"   Query: '{query}'")
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="auto",
                top_k=10
            )
            print(f"   ✓ Success: {len(result.results)} results")
            if result.results:
                print(f"   Top match: {result.results[0]['code_dotted']}")
        except ValueError as e:
            print(f"   ✓ Expected ValueError: {e}")
        except Exception as e:
            print(f"   ❌ Unexpected error: {type(e).__name__}: {e}")

    print_separator()


def test_result_structure():
    """Test that result structure matches expected schema."""
    print_separator("TEST: RESULT STRUCTURE VALIDATION")

    query = "Acute bronchitis"

    try:
        result = map_icd_codes(
            query_text=query,
            query_type="auto",
            top_k=3
        )

        print(f"\n✓ Testing ICDMapperResult structure...")

        required_attrs = ['query_text', 'query_type', 'top_k', 'results', 'latencies']
        for attr in required_attrs:
            if hasattr(result, attr):
                print(f"   ✓ Has attribute: {attr}")
            else:
                print(f"   ✗ Missing attribute: {attr}")

        print(f"\n✓ Testing latencies dictionary...")
        expected_latencies = ['hybrid_search_ms', 'confidence_scoring_ms', 'total_ms']
        for key in expected_latencies:
            if key in result.latencies:
                print(f"   ✓ Has latency: {key}")
            else:
                print(f"   ✗ Missing latency: {key}")


        if result.results:
            print(f"\n✓ Testing result item structure...")
            required_fields = [
                'code', 'code_dotted', 'long_description', 'short_description',
                'category_code', 'category_title', 'score', 'confidence'
            ]
            for field in required_fields:
                if field in result.results[0]:
                    print(f"   ✓ Has field: {field}")
                else:
                    print(f"   ✗ Missing field: {field}")

        print(f"\n✓ STRUCTURE VALIDATION COMPLETE")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")

    print_separator()


def test_performance_benchmark():
    """Benchmark pipeline performance."""
    print_separator("TEST: PERFORMANCE BENCHMARK")

    queries = [
        "Hypertension",
        "Diabetes mellitus type 2",
        "Acute appendicitis"
    ]

    total_times = []

    for query in queries:
        try:
            result = map_icd_codes(
                query_text=query,
                query_type="auto",
                top_k=10
            )
            total_times.append(result.latencies.get('total_ms', 0))

            print(f"\n📊 Query: {query}")
            print(f"   Type Detection: {result.latencies.get('type_detection_ms', 0):.1f}ms")
            print(f"   Hybrid Search: {result.latencies.get('hybrid_search_ms', 0):.1f}ms")
            print(f"   Confidence Scoring: {result.latencies.get('confidence_scoring_ms', 0):.1f}ms")
            print(f"   Total: {result.latencies.get('total_ms', 0):.1f}ms")
        except Exception as e:
            print(f"\n❌ ERROR for query '{query}': {e}")

    if total_times:
        avg_time = sum(total_times) / len(total_times)
        min_time = min(total_times)
        max_time = max(total_times)

        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Average Total Time: {avg_time:.1f}ms")
        print(f"   Fastest: {min_time:.1f}ms")
        print(f"   Slowest: {max_time:.1f}ms")

    print_separator()


def interactive_test():
    """Interactive mode for testing custom queries."""
    print_separator("INTERACTIVE TESTING MODE")
    print("Enter medical text to map to ICD codes (or 'quit' to exit)")
    print("Commands:")
    print("  - Type your query and press Enter")
    print("  - Type 'mode:auto', 'mode:diagnosis', or 'mode:procedure' to change mode")
    print("  - Type 'top:N' to change top_k value")
    print("  - Type 'quit' or 'exit' to exit")
    print_separator()

    current_mode = "auto"
    current_top_k = 5

    while True:
        text = input(f"\n[mode={current_mode}, top_k={current_top_k}] Enter query: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break

        if text.startswith('mode:'):
            mode = text.split(':')[1].strip()
            if mode in ['auto', 'diagnosis', 'procedure']:
                current_mode = mode
                print(f"✓ Mode changed to: {current_mode}")
            else:
                print(f"✗ Invalid mode. Use: auto, diagnosis, or procedure")
            continue

        if text.startswith('top:'):
            try:
                k = int(text.split(':')[1].strip())
                if k > 0:
                    current_top_k = k
                    print(f"✓ Top K changed to: {current_top_k}")
                else:
                    print(f"✗ Top K must be positive")
            except ValueError:
                print(f"✗ Invalid number for top_k")
            continue

        if not text:
            print("Please enter some text.")
            continue

        try:
            result = map_icd_codes(
                query_text=text,
                query_type=current_mode,
                top_k=current_top_k
            )
            print_result(result)
        except Exception as e:
            print(f"\n❌ ERROR: {e}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("ICD CODE MAPPER TEST SUITE".center(80))
    print("=" * 80)
    print("\nThis test suite validates the complete ICD mapping pipeline:")
    print("  ✓ Type detection (auto mode)")
    print("  ✓ Hybrid search (dense + sparse + RRF + reranking)")
    print("  ✓ Confidence scoring (LLM-based)")
    print("  ✓ Result formatting and latency tracking")
    print("=" * 80)

    try:
        # Run automated tests
        test_auto_mode_diagnosis()
        test_auto_mode_procedure()
        test_explicit_diagnosis_mode()
        test_explicit_procedure_mode()
        test_different_top_k_values()
        test_edge_cases()
        test_result_structure()
        test_performance_benchmark()

        print("\n" + "=" * 80)
        print("✅ ALL AUTOMATED TESTS COMPLETED!".center(80))
        print("=" * 80)

        # Optional: Run interactive mode
        response = input("\nRun interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            interactive_test()

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all API keys are set (GROQ_API_KEY, etc.)")
        print("2. Verify Qdrant is running and accessible")
        print("3. Check that collection 'icd_names' exists and is populated")
        print("4. Verify all dependencies are installed")


if __name__ == "__main__":
    main()
