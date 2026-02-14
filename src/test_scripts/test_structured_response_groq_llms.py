"""
Test script for Groq structured JSON completion.
"""
from pydantic import BaseModel
from typing import List, Optional
from utils.groq_llms import call_groq_structured



class ProductReview(BaseModel):
    """Model for product review extraction."""
    product_name: str
    rating: float
    summary: str
    pros: List[str]
    cons: List[str]


class ICDCode(BaseModel):
    """Model for ICD-10 code extraction."""
    code: str
    description: str
    category: str


class PersonInfo(BaseModel):
    """Model for person information extraction."""
    name: str
    age: int
    occupation: str
    location: Optional[str] = None


def test_pydantic_model_strict():
    """Test structured output with Pydantic model in strict mode."""
    print("Testing Pydantic model with strict mode...")

    result = call_groq_structured(
        prompt="I bought the UltraSound Pro Headphones last week and they're incredible! The sound quality is amazing and the battery lasts forever. Only downside is they're a bit heavy. I'd give them 4.5 out of 5 stars.",
        response_model=ProductReview,
        system_prompt="Extract product review information from the text.",
        temperature=0.3,
        model="openai/gpt-oss-20b",
        strict=True
    )

    print(f"Product Name: {result.product_name}")
    print(f"Rating: {result.rating}")
    print(f"Summary: {result.summary}")
    print(f"Pros: {result.pros}")
    print(f"Cons: {result.cons}")
    print(f"Type: {type(result)}")
    print()


def test_icd_code_extraction():
    """Test ICD-10 code extraction with structured output."""
    print("Testing ICD-10 code extraction...")

    result = call_groq_structured(
        prompt="Patient diagnosed with type 2 diabetes mellitus without complications.",
        response_model=ICDCode,
        system_prompt="Extract ICD-10 code information. Use E11.9 for Type 2 diabetes mellitus without complications.",
        temperature=0.1,
        model="openai/gpt-oss-20b",
        strict=True
    )

    print(f"ICD Code: {result.code}")
    print(f"Description: {result.description}")
    print(f"Category: {result.category}")
    print()


def test_person_info_with_optional():
    """Test structured output with optional fields."""
    print("Testing person info with optional fields...")

    result = call_groq_structured(
        prompt="John Smith is a 35-year-old software engineer.",
        response_model=PersonInfo,
        system_prompt="Extract person information from the text.",
        temperature=0.2,
        model="openai/gpt-oss-20b",
        strict=True
    )

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print(f"Location: {result.location}")
    print()


def test_dict_schema_strict():
    """Test structured output with raw dict schema in strict mode."""
    print("Testing raw dict schema with strict mode...")

    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "author": {"type": "string"},
            "year": {"type": "number"},
            "genre": {"type": "string"}
        },
        "required": ["title", "author", "year", "genre"],
        "additionalProperties": False
    }

    result = call_groq_structured(
        prompt="Tell me about the book '1984' by George Orwell published in 1949. It's a dystopian fiction novel.",
        response_model=schema,
        system_prompt="Extract book information.",
        temperature=0.2,
        model="openai/gpt-oss-20b",
        strict=True
    )

    print(f"Title: {result['title']}")
    print(f"Author: {result['author']}")
    print(f"Year: {result['year']}")
    print(f"Genre: {result['genre']}")
    print(f"Type: {type(result)}")
    print()


def test_best_effort_mode():
    """Test structured output with best-effort mode (strict=False)."""
    print("Testing best-effort mode (strict=False)...")

    result = call_groq_structured(
        prompt="Sarah Johnson is 28 years old and works as a data scientist in San Francisco.",
        response_model=PersonInfo,
        system_prompt="Extract person information from the text.",
        temperature=0.3,
        model="llama-3.3-70b-versatile",  # Using a model that works better with best-effort
        strict=False
    )

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Occupation: {result.occupation}")
    print(f"Location: {result.location}")
    print()


def test_multiple_extractions():
    """Test multiple structured extractions in sequence."""
    print("Testing multiple extractions...")

    prompts = [
        "Mike Chen is 42 and is a chef in New York.",
        "Emma Watson, age 30, teacher from London.",
        "David Kim, 25-year-old musician."
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"  Extraction {i}:")
        result = call_groq_structured(
            prompt=prompt,
            response_model=PersonInfo,
            system_prompt="Extract person information.",
            temperature=0.2,
            model="openai/gpt-oss-20b",
            strict=True
        )
        print(f"    Name: {result.name}, Age: {result.age}, Job: {result.occupation}")

    print()


def test_complex_nested_structure():
    """Test with a more complex nested structure."""
    print("Testing complex nested structure...")

    class Address(BaseModel):
        street: str
        city: str
        country: str

    class Person(BaseModel):
        name: str
        age: int
        address: Address

    result = call_groq_structured(
        prompt="Alice Brown is 29 years old and lives at 123 Main Street in Seattle, USA.",
        response_model=Person,
        system_prompt="Extract person and address information.",
        temperature=0.2,
        model="openai/gpt-oss-20b",
        strict=True
    )

    print(f"Name: {result.name}")
    print(f"Age: {result.age}")
    print(f"Address: {result.address.street}, {result.address.city}, {result.address.country}")
    print()


if __name__ == "__main__":
    print("=" * 70)
    print("Groq Structured JSON Completion Test")
    print("=" * 70)
    print()

    try:
        # Run all tests
        test_pydantic_model_strict()
        test_icd_code_extraction()
        test_person_info_with_optional()
        test_dict_schema_strict()
        test_best_effort_mode()
        test_multiple_extractions()
        test_complex_nested_structure()

        print("=" * 70)
        print("All tests completed successfully!")
        print("=" * 70)
        print("\nNote: Tests use actual Groq API calls and require GROQ_API_KEY")
        print("Strict mode requires gpt-oss-20b or gpt-oss-120b models")

    except Exception as e:
        print(f"\nError during testing: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure GROQ_API_KEY environment variable is set")
        print("2. Verify you have access to gpt-oss-20b model for strict mode")
        print("3. Check your API rate limits and quota")
