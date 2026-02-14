"""
Simple Gemini API client for generating content.
"""
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


def call_gemini(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str = "gemini-3-flash-preview"
) -> str:
    """
    Call Gemini model with the specified parameters.

    Args:
        prompt: User prompt/question to send to the model
        system_prompt: System instruction for the model (optional)
        temperature: Controls randomness (0.0 to 1.0, default 0.7)
        max_tokens: Maximum tokens in response (default 1024)
        model: Gemini model to use (default gemini-3-flash-preview)

    Returns:
        Generated text response from the model
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )

    if system_prompt:
        config.system_instruction = system_prompt

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    return response.text
