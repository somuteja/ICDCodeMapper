"""
Test script for medical text type detection (diagnosis vs procedure).

This demonstrates how to use the detection prompt with Groq API to automatically
classify medical text when the user selects "auto" mode.
"""
from core.prompts.auto_detection_prompt import (
    MedicalTextTypeDetection,
    EXAMPLE_DIAGNOSES,
    EXAMPLE_PROCEDURES
)
from core.text_type_detector import detect_text_type


def test_single_detection(medical_text: str, expected_type: str = None):
    """Test detection on a single medical text."""
    print(f"\n{'=' * 80}")
    print(f"INPUT TEXT: {medical_text}")
    print(f"{'=' * 80}")

    result = detect_text_type(medical_text)

    print(f"\n🧠 REASONING:")
    print(f"{result.reasoning}")

    print(f"\n🔍 KEY INDICATORS:")
    for indicator in result.key_indicators:
        print(f"  - {indicator}")

    print(f"\n✅ CLASSIFICATION: {result.text_type.upper()}")
    print(f"📊 CONFIDENCE: {result.confidence_level.upper()}")

    if result.alternative_interpretation:
        print(f"\n💭 ALTERNATIVE INTERPRETATION:")
        print(f"{result.alternative_interpretation}")

    if expected_type:
        is_correct = result.text_type == expected_type
        print(f"\n{'✓' if is_correct else '✗'} Expected: {expected_type}, Got: {result.text_type}")

    print(f"\n{'=' * 80}")

    return result


def test_example_diagnoses():
    """Test detection on example diagnosis texts."""
    print("\n" + "=" * 80)
    print("TESTING EXAMPLE DIAGNOSES")
    print("=" * 80)

    for i, text in enumerate(EXAMPLE_DIAGNOSES[:3], 1):  # Test first 3
        print(f"\n[{i}/{len(EXAMPLE_DIAGNOSES[:3])}]")
        test_single_detection(text, expected_type="diagnosis")


def test_example_procedures():
    """Test detection on example procedure texts."""
    print("\n" + "=" * 80)
    print("TESTING EXAMPLE PROCEDURES")
    print("=" * 80)

    for i, text in enumerate(EXAMPLE_PROCEDURES[:3], 1):  # Test first 3
        print(f"\n[{i}/{len(EXAMPLE_PROCEDURES[:3])}]")
        test_single_detection(text, expected_type="procedure")


def test_ambiguous_cases():
    """Test detection on potentially ambiguous cases."""
    print("\n" + "=" * 80)
    print("TESTING AMBIGUOUS CASES")
    print("=" * 80)

    ambiguous_texts = [
        ("Cardiac arrest", "diagnosis"),  # Condition, not procedure
        ("Cardiac catheterization", "procedure"),  # Procedure despite "cardiac"
        ("Insulin therapy", "procedure"),  # Treatment/intervention
        ("Insulin-dependent diabetes", "diagnosis"),  # Condition
        ("Pain management", "procedure"),  # Treatment approach
        ("Chronic pain syndrome", "diagnosis")  # Condition
    ]

    for text, expected in ambiguous_texts:
        test_single_detection(text, expected_type=expected)


def test_real_world_scenarios():
    """Test with real-world patient scenario descriptions."""
    print("\n" + "=" * 80)
    print("TESTING REAL-WORLD SCENARIOS")
    print("=" * 80)

    scenarios = [
        ("Patient has been diagnosed with acute appendicitis", "diagnosis"),
        ("Underwent laparoscopic appendectomy yesterday", "procedure"),
        ("Follow-up CT scan of abdomen and pelvis", "procedure"),
        ("Postoperative wound infection", "diagnosis"),
        ("Removal of surgical staples", "procedure"),
        ("Acute bronchitis with wheezing", "diagnosis")
    ]

    for text, expected in scenarios:
        test_single_detection(text, expected_type=expected)


def interactive_test():
    """Interactive mode for testing custom inputs."""
    print("\n" + "=" * 80)
    print("INTERACTIVE TESTING MODE")
    print("=" * 80)
    print("Enter medical text to classify (or 'quit' to exit)")
    print("=" * 80)

    while True:
        text = input("\nEnter medical text: ").strip()

        if text.lower() in ['quit', 'exit', 'q']:
            print("Exiting interactive mode...")
            break

        if not text:
            print("Please enter some text.")
            continue

        test_single_detection(text)


def main():
    """Run all tests."""
    print("MEDICAL TEXT TYPE DETECTION TEST SUITE")
    print("Model: moonshotai/kimi-k2-instruct-0905")
    print("Temperature: 0.0 (Deterministic)")

    try:
        # Run automated tests
        test_example_diagnoses()
        test_example_procedures()
        test_ambiguous_cases()
        test_real_world_scenarios()

        print("\n" + "=" * 80)
        print("✅ ALL AUTOMATED TESTS COMPLETED!")
        print("=" * 80)

        # Optional: Run interactive mode
        response = input("\nRun interactive mode? (y/n): ").strip().lower()
        if response == 'y':
            interactive_test()

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure GROQ_API_KEY is set in environment variables")
        print("2. Verify access to moonshotai/kimi-k2-instruct-0905 model")
        print("3. Check API rate limits and quota")


if __name__ == "__main__":
    main()
