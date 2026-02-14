"""
Test script for Gemini API client.
"""
from utils.gemini_llms import call_gemini


def test_basic_call():
    """Test basic Gemini API call."""
    print("Testing basic Gemini API call...")
    response = call_gemini(
        prompt="Say 'Hello, World!' and nothing else.",
        temperature=0.3,
        max_tokens=50
    )
    print(f"Response: {response}")
    print()


def test_with_system_prompt():
    """Test Gemini API call with system prompt."""
    print("Testing with system prompt...")
    response = call_gemini(
        prompt="What is ICD-10?",
        system_prompt="You are a medical coding expert. Keep answers concise.",
        temperature=0.5,
        max_tokens=200
    )
    print(f"Response: {response}")
    print()


def test_different_temperatures():
    """Test with different temperature settings."""
    print("Testing different temperatures...")

    prompt = "Complete this sentence: The weather today is"

    print("Low temperature (0.1):")
    response_low = call_gemini(prompt=prompt, temperature=0.1, max_tokens=30)
    print(f"Response: {response_low}")
    print()

    print("High temperature (0.9):")
    response_high = call_gemini(prompt=prompt, temperature=0.9, max_tokens=30)
    print(f"Response: {response_high}")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Gemini API Client Test")
    print("=" * 60)
    print()

    try:
        test_basic_call()
        test_with_system_prompt()
        test_different_temperatures()

        print("=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"Error during testing: {e}")
        print("\nMake sure GEMINI_API_KEY environment variable is set.")
