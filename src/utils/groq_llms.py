"""
Simple Groq API client for generating content.
"""
import os
import json
from typing import Any, Dict, Optional, Type, Union
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel


load_dotenv()


def call_groq(
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str = "llama-3.3-70b-versatile"
) -> str:
    """
    Call Groq model with the specified parameters.

    Args:
        prompt: User prompt/question to send to the model
        system_prompt: System instruction for the model (optional)
        temperature: Controls randomness (0.0 to 1.0, default 0.7)
        max_tokens: Maximum tokens in response (default 1024)
        model: Groq model to use (default llama-3.3-70b-versatile)

    Returns:
        Generated text response from the model
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)

    messages = []

    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })

    messages.append({
        "role": "user",
        "content": prompt
    })

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    return chat_completion.choices[0].message.content


def call_groq_structured(
    prompt: str,
    response_model: Union[Type[BaseModel], Dict[str, Any]],
    system_prompt: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str = "openai/gpt-oss-20b",
    strict: bool = True
) -> Union[BaseModel, Dict[str, Any]]:
    """
    Call Groq model with structured JSON output using response schema.

    This function uses Groq's Structured Outputs feature to ensure responses
    conform to a specified JSON schema. In strict mode (strict=True), the output
    is guaranteed to match the schema. In best-effort mode (strict=False), the
    model attempts to match the schema but may occasionally fail.

    Args:
        prompt: User prompt/question to send to the model
        response_model: Either a Pydantic BaseModel class or a dict containing JSON schema.
                       If Pydantic model is provided, it will be converted to JSON schema.
        system_prompt: System instruction for the model (optional)
        temperature: Controls randomness (0.0 to 1.0, default 0.7)
        max_tokens: Maximum tokens in response (default 1024)
        model: Groq model to use (default openai/gpt-oss-20b for strict mode support)
               Note: Strict mode only works with gpt-oss-20b and gpt-oss-120b models
        strict: If True, uses constrained decoding for guaranteed schema compliance.
               If False, uses best-effort mode (default True)

    Returns:
        If response_model is a Pydantic model: Returns an instance of that model
        If response_model is a dict schema: Returns parsed JSON dict

    Raises:
        ValueError: If GROQ_API_KEY is not found in environment variables
        ValidationError: If using best-effort mode and response doesn't match schema

    Example:
        >>> from pydantic import BaseModel
        >>>
        >>> class Review(BaseModel):
        ...     product_name: str
        ...     rating: float
        ...     summary: str
        >>>
        >>> result = call_groq_structured(
        ...     prompt="I love these headphones! 5 stars!",
        ...     response_model=Review,
        ...     system_prompt="Extract product review information"
        ... )
        >>> print(result.product_name, result.rating)
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")

    client = Groq(api_key=api_key)


    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": system_prompt
        })
    messages.append({
        "role": "user",
        "content": prompt
    })

    is_pydantic = isinstance(response_model, type) and issubclass(response_model, BaseModel)

    if is_pydantic:
        schema = response_model.model_json_schema()
        schema_name = response_model.__name__.lower()

        if strict:
            if "properties" in schema:
                schema["required"] = list(schema["properties"].keys())
            schema["additionalProperties"] = False
    else:
        schema = response_model
        schema_name = schema.get("title", "response_schema").lower()

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_completion_tokens=max_tokens,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": strict,
                "schema": schema
            }
        }
    )

    response_content = chat_completion.choices[0].message.content
    parsed_json = json.loads(response_content)

    if is_pydantic:
        return response_model(**parsed_json)
    else:
        return parsed_json
