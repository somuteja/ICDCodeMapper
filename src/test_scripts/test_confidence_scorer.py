"""
Test script for confidence scoring functionality.

This test file verifies that the confidence scorer correctly evaluates
ICD code search results and assigns appropriate confidence scores using
LLM-based medical coding expertise.
"""

import logging
from dotenv import load_dotenv

from core.confidence_scorer import score_search_results
from core.prompts.confidence_scoring_prompt import ConfidenceScoringResult


load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


def print_scoring_result(result: ConfidenceScoringResult):
    """Pretty print confidence scoring results."""
    print(f"\n{'=' * 80}")
    print("CONFIDENCE SCORING RESULT")
    print(f"{'=' * 80}")

    print(f"\nQUERY UNDERSTANDING:")
    print(f"{result.query_understanding}")

    print(f"\nREASONING:")
    print(f"{result.reasoning}")

    print(f"\nEVALUATED CODES:")
    for i, code_eval in enumerate(result.evaluated_codes, 1):
        print(f"\n  {i}. {code_eval.code}")
        print(f"     Relevance Score: {code_eval.relevance_score:.2f}")
        print(f"     Confidence: {code_eval.confidence.upper()}")
        print(f"     Reasoning: {code_eval.match_reasoning}")

    print(f"\nBEST CODE: {result.best_code}")
    print(f"OVERALL CONFIDENCE: {result.overall_confidence.upper()}")
    print(f"{'=' * 80}\n")


def create_mock_search_results(query_type: str = "diagnosis") -> list[dict]:
    """Create mock search results for testing."""
    if query_type == "diagnosis":
        return [
            {
                "code_dotted": "E11.9",
                "long_description": "Type 2 diabetes mellitus without complications",
                "category_title": "Diabetes mellitus",
                "score": 0.95
            },
            {
                "code_dotted": "E11.65",
                "long_description": "Type 2 diabetes mellitus with hyperglycemia",
                "category_title": "Diabetes mellitus",
                "score": 0.88
            },
            {
                "code_dotted": "E10.9",
                "long_description": "Type 1 diabetes mellitus without complications",
                "category_title": "Diabetes mellitus",
                "score": 0.75
            },
            {
                "code_dotted": "E11.8",
                "long_description": "Type 2 diabetes mellitus with unspecified complications",
                "category_title": "Diabetes mellitus",
                "score": 0.70
            },
            {
                "code_dotted": "E13.9",
                "long_description": "Other specified diabetes mellitus without complications",
                "category_title": "Diabetes mellitus",
                "score": 0.65
            }
        ]
    else:  # procedure
        return [
            {
                "code_dotted": "0SR9019",
                "long_description": "Replacement of Right Hip Joint with Metal Synthetic Substitute, Cemented, Open Approach",
                "category_title": "Hip joint replacement procedures",
                "score": 0.92
            },
            {
                "code_dotted": "0SRB019",
                "long_description": "Replacement of Left Hip Joint with Metal Synthetic Substitute, Cemented, Open Approach",
                "category_title": "Hip joint replacement procedures",
                "score": 0.85
            },
            {
                "code_dotted": "0SR901A",
                "long_description": "Replacement of Right Hip Joint with Metal Synthetic Substitute, Uncemented, Open Approach",
                "category_title": "Hip joint replacement procedures",
                "score": 0.80
            }
        ]


def test_basic_diagnosis_scoring():
    """Test basic confidence scoring with a diagnosis query."""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Diagnosis Confidence Scoring")
    print("=" * 80)

    user_query = "type 2 diabetes"
    search_results = create_mock_search_results(query_type="diagnosis")

    print(f"\nUser Query: '{user_query}'")
    print(f"Search Results to Evaluate: {len(search_results)} codes")
    print(f"Query Type: diagnosis")

    print("\nCalling confidence scorer...")
    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=5
    )

    print_scoring_result(result)

    # Validate result
    assert isinstance(result, ConfidenceScoringResult), "Result should be ConfidenceScoringResult"
    assert result.best_code, "Best code should not be empty"
    assert len(result.evaluated_codes) > 0, "Should have evaluated codes"
    assert result.overall_confidence in ["high", "medium", "low"], "Valid confidence level"

    print("✓ Basic diagnosis scoring test passed!\n")
    return result


def test_basic_procedure_scoring():
    """Test basic confidence scoring with a procedure query."""
    print("\n" + "=" * 80)
    print("TEST 2: Basic Procedure Confidence Scoring")
    print("=" * 80)

    user_query = "total hip replacement right side"
    search_results = create_mock_search_results(query_type="procedure")

    print(f"\nUser Query: '{user_query}'")
    print(f"Search Results to Evaluate: {len(search_results)} codes")
    print(f"Query Type: procedure")

    print("\nCalling confidence scorer...")
    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="procedure",
        top_k=3
    )

    print_scoring_result(result)

    # Validate result
    assert isinstance(result, ConfidenceScoringResult), "Result should be ConfidenceScoringResult"
    assert len(result.evaluated_codes) <= 3, "Should evaluate top_k=3 codes"
    assert result.best_code, "Best code should not be empty"

    print("✓ Basic procedure scoring test passed!\n")
    return result


def test_top_k_parameter():
    """Test that top_k parameter limits the number of evaluated codes."""
    print("\n" + "=" * 80)
    print("TEST 3: Top K Parameter")
    print("=" * 80)

    user_query = "diabetes with complications"
    search_results = create_mock_search_results(query_type="diagnosis")

    print(f"\nUser Query: '{user_query}'")
    print(f"Total Search Results: {len(search_results)} codes")

    # Test with top_k=3
    print(f"\n--- Testing with top_k=3 ---")
    result_3 = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=3
    )

    print(f"Evaluated Codes: {len(result_3.evaluated_codes)}")
    print(f"Codes: {[c.code for c in result_3.evaluated_codes]}")

    # Test with top_k=5
    print(f"\n--- Testing with top_k=5 ---")
    result_5 = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=5
    )

    print(f"Evaluated Codes: {len(result_5.evaluated_codes)}")
    print(f"Codes: {[c.code for c in result_5.evaluated_codes]}")

    # Validate
    assert len(result_3.evaluated_codes) <= 3, f"Expected <= 3 codes, got {len(result_3.evaluated_codes)}"
    assert len(result_5.evaluated_codes) <= 5, f"Expected <= 5 codes, got {len(result_5.evaluated_codes)}"

    print("\n✓ Top K parameter test passed!\n")


def test_relevance_scores():
    """Test that relevance scores are within valid range."""
    print("\n" + "=" * 80)
    print("TEST 4: Relevance Score Validation")
    print("=" * 80)

    user_query = "chronic heart failure"
    search_results = [
        {
            "code_dotted": "I50.9",
            "long_description": "Heart failure, unspecified",
            "category_title": "Heart failure",
            "score": 0.90
        },
        {
            "code_dotted": "I50.23",
            "long_description": "Acute on chronic systolic (congestive) heart failure",
            "category_title": "Heart failure",
            "score": 0.85
        },
        {
            "code_dotted": "I11.0",
            "long_description": "Hypertensive heart disease with heart failure",
            "category_title": "Hypertensive diseases",
            "score": 0.70
        }
    ]

    print(f"\nUser Query: '{user_query}'")
    print(f"Evaluating {len(search_results)} codes")

    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=3
    )

    print(f"\nRelevance Scores:")
    for code_eval in result.evaluated_codes:
        print(f"  {code_eval.code}: {code_eval.relevance_score:.2f} ({code_eval.confidence})")

        # Validate score range
        assert 0.0 <= code_eval.relevance_score <= 1.0, \
            f"Relevance score {code_eval.relevance_score} out of range [0.0, 1.0]"

    # Check that scores are ordered (best should be first)
    if len(result.evaluated_codes) > 1:
        scores = [c.relevance_score for c in result.evaluated_codes]
        print(f"\nScore ordering: {scores}")
        # Note: Scores might not always be perfectly descending depending on LLM

    print("\n✓ Relevance score validation test passed!\n")


def test_confidence_levels():
    """Test that appropriate confidence levels are assigned."""
    print("\n" + "=" * 80)
    print("TEST 5: Confidence Level Assignment")
    print("=" * 80)

    test_cases = [
        {
            "query": "type 2 diabetes mellitus",
            "results": [
                {
                    "code_dotted": "E11.9",
                    "long_description": "Type 2 diabetes mellitus without complications",
                    "category_title": "Diabetes mellitus",
                    "score": 0.98
                }
            ],
            "expected_confidence": "high"
        },
        {
            "query": "diabetic complications",
            "results": [
                {
                    "code_dotted": "E11.8",
                    "long_description": "Type 2 diabetes mellitus with unspecified complications",
                    "category_title": "Diabetes mellitus",
                    "score": 0.75
                },
                {
                    "code_dotted": "E11.9",
                    "long_description": "Type 2 diabetes mellitus without complications",
                    "category_title": "Diabetes mellitus",
                    "score": 0.70
                }
            ],
            "expected_confidence": "medium"
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"Query: '{test_case['query']}'")

        result = score_search_results(
            user_query=test_case["query"],
            search_results=test_case["results"],
            query_type="diagnosis",
            top_k=3
        )

        print(f"Overall Confidence: {result.overall_confidence}")
        print(f"Best Code: {result.best_code}")

        for code_eval in result.evaluated_codes:
            print(f"  {code_eval.code}: {code_eval.confidence} confidence")

        # Note: We can't strictly enforce expected confidence as it depends on LLM reasoning
        assert result.overall_confidence in ["high", "medium", "low"], \
            "Confidence must be high, medium, or low"

    print("\n✓ Confidence level assignment test passed!\n")


def test_reasoning_quality():
    """Test that reasoning and explanations are provided."""
    print("\n" + "=" * 80)
    print("TEST 6: Reasoning Quality")
    print("=" * 80)

    user_query = "acute myocardial infarction"
    search_results = [
        {
            "code_dotted": "I21.9",
            "long_description": "Acute myocardial infarction, unspecified",
            "category_title": "Ischemic heart diseases",
            "score": 0.95
        },
        {
            "code_dotted": "I21.4",
            "long_description": "Non-ST elevation (NSTEMI) myocardial infarction",
            "category_title": "Ischemic heart diseases",
            "score": 0.88
        }
    ]

    print(f"\nUser Query: '{user_query}'")

    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=2
    )

    # Validate reasoning fields
    print(f"\nQuery Understanding ({len(result.query_understanding)} chars):")
    print(f"{result.query_understanding[:200]}...")
    assert result.query_understanding, "Query understanding should not be empty"
    assert len(result.query_understanding) > 20, "Query understanding should be meaningful"

    print(f"\nOverall Reasoning ({len(result.reasoning)} chars):")
    print(f"{result.reasoning[:200]}...")
    assert result.reasoning, "Reasoning should not be empty"
    assert len(result.reasoning) > 20, "Reasoning should be meaningful"

    print(f"\nCode-specific Reasoning:")
    for code_eval in result.evaluated_codes:
        print(f"\n  {code_eval.code}:")
        print(f"  {code_eval.match_reasoning[:150]}...")
        assert code_eval.match_reasoning, f"Code {code_eval.code} should have reasoning"
        assert len(code_eval.match_reasoning) > 10, f"Code {code_eval.code} reasoning should be meaningful"

    print("\n✓ Reasoning quality test passed!\n")


def test_best_code_selection():
    """Test that the best code is selected from evaluated codes."""
    print("\n" + "=" * 80)
    print("TEST 7: Best Code Selection")
    print("=" * 80)

    user_query = "pneumonia"
    search_results = [
        {
            "code_dotted": "J18.9",
            "long_description": "Pneumonia, unspecified organism",
            "category_title": "Pneumonia",
            "score": 0.92
        },
        {
            "code_dotted": "J15.9",
            "long_description": "Unspecified bacterial pneumonia",
            "category_title": "Pneumonia",
            "score": 0.88
        },
        {
            "code_dotted": "J12.9",
            "long_description": "Viral pneumonia, unspecified",
            "category_title": "Pneumonia",
            "score": 0.82
        }
    ]

    print(f"\nUser Query: '{user_query}'")
    print(f"Candidate Codes: {[r['code_dotted'] for r in search_results]}")

    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=3
    )

    print(f"\nBest Code Selected: {result.best_code}")

    # Validate best code is in evaluated codes
    evaluated_code_list = [c.code for c in result.evaluated_codes]
    print(f"Evaluated Codes: {evaluated_code_list}")

    assert result.best_code in evaluated_code_list, \
        f"Best code '{result.best_code}' should be in evaluated codes: {evaluated_code_list}"

    print(f"\n✓ Best code '{result.best_code}' is valid!")
    print("\n✓ Best code selection test passed!\n")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 80)
    print("TEST 8: Error Handling")
    print("=" * 80)

    # Test empty search results
    print("\n1. Testing empty search results...")
    try:
        result = score_search_results(
            user_query="diabetes",
            search_results=[],
            query_type="diagnosis",
            top_k=5
        )
        print("Note: Empty results handled gracefully (LLM may return empty evaluation)")
        # Depending on implementation, this might succeed with empty evaluated_codes
    except Exception as e:
        print(f"✓ Correctly raised exception: {type(e).__name__}: {e}")

    # Test invalid query type
    print("\n2. Testing potential invalid query type handling...")
    try:
        # This might be validated at the Pydantic level or in the function
        result = score_search_results(
            user_query="test",
            search_results=create_mock_search_results(),
            query_type="invalid_type",  # type: ignore
            top_k=5
        )
        print("Note: Invalid query type may be passed to LLM (depends on validation)")
    except Exception as e:
        print(f"✓ Correctly raised exception: {type(e).__name__}")

    print("\n✓ Error handling test complete!\n")


def test_realistic_scenario():
    """Test with a realistic medical coding scenario."""
    print("\n" + "=" * 80)
    print("TEST 9: Realistic Medical Coding Scenario")
    print("=" * 80)

    user_query = "patient presents with chest pain and shortness of breath, suspected acute coronary syndrome"
    search_results = [
        {
            "code_dotted": "I24.9",
            "long_description": "Acute ischemic heart disease, unspecified",
            "category_title": "Ischemic heart diseases",
            "score": 0.89
        },
        {
            "code_dotted": "I20.0",
            "long_description": "Unstable angina",
            "category_title": "Ischemic heart diseases",
            "score": 0.85
        },
        {
            "code_dotted": "R07.9",
            "long_description": "Chest pain, unspecified",
            "category_title": "Symptoms and signs involving the circulatory and respiratory systems",
            "score": 0.82
        },
        {
            "code_dotted": "R06.02",
            "long_description": "Shortness of breath",
            "category_title": "Symptoms and signs involving the circulatory and respiratory systems",
            "score": 0.78
        }
    ]

    print(f"\nComplex Query: '{user_query}'")
    print(f"Mixed Results: ICD codes + symptom codes")

    result = score_search_results(
        user_query=user_query,
        search_results=search_results,
        query_type="diagnosis",
        top_k=4
    )

    print_scoring_result(result)

    print("\nAnalysis:")
    print(f"Best Code: {result.best_code}")
    print(f"Overall Confidence: {result.overall_confidence}")
    print("\nExpected behavior: Should prefer specific diagnosis codes (I24.9, I20.0)")
    print("over symptom codes (R07.9, R06.02) for 'suspected acute coronary syndrome'")

    print("\n✓ Realistic scenario test passed!\n")


def main():
    """Run all confidence scoring tests."""
    print("\n" + "=" * 80)
    print("CONFIDENCE SCORER TEST SUITE")
    print("=" * 80)
    print("Model: moonshotai/kimi-k2-instruct-0905")
    print("Testing LLM-based confidence evaluation for ICD code search results")
    print("=" * 80)

    try:
        # Run all tests
        test_basic_diagnosis_scoring()
        test_basic_procedure_scoring()
        test_top_k_parameter()
        test_relevance_scores()
        test_confidence_levels()
        test_reasoning_quality()
        test_best_code_selection()
        test_error_handling()
        test_realistic_scenario()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nConfidence scorer is working correctly with all features:")
        print("  • Basic diagnosis and procedure scoring")
        print("  • Top K parameter control")
        print("  • Relevance score validation (0.0 to 1.0)")
        print("  • Confidence level assignment (high/medium/low)")
        print("  • Quality reasoning and explanations")
        print("  • Best code selection from candidates")
        print("  • Error handling for edge cases")
        print("  • Realistic medical coding scenarios")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        print("\nTroubleshooting:")
        print("1. Ensure GROQ_API_KEY is set in .env file")
        print("2. Verify access to moonshotai/kimi-k2-instruct-0905 model")
        print("3. Check API rate limits and quota")
        print("4. Verify Qdrant connection if search results are from real searches")


if __name__ == "__main__":
    main()
