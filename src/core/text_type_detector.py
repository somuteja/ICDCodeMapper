"""
Medical text type detector module.

Provides a simple interface for detecting whether medical text describes
a diagnosis or procedure using LLM-based classification.
"""
import logging
from core.prompts.auto_detection_prompt import (
    MedicalTextTypeDetection,
    SYSTEM_PROMPT_FOR_TYPE_DETECTION,
    get_detection_prompt,
    get_fallback_detection,
    DETECTION_CONFIG
)
from utils.groq_llms import call_groq_structured

logger = logging.getLogger(__name__)


def detect_text_type(medical_text: str) -> MedicalTextTypeDetection:
    """
    Detect whether medical text describes a diagnosis or procedure.

    Uses Groq API with Moonshot Kimi K2 model and chain of thought reasoning
    for transparent and accurate classification.

    Args:
        medical_text: The medical text to classify

    Returns:
        MedicalTextTypeDetection object containing:
            - reasoning: Step-by-step thought process
            - key_indicators: Words/phrases that support the classification
            - text_type: Either "diagnosis" or "procedure"
            - confidence_level: "high", "medium", or "low"
            - alternative_interpretation: Explanation if confidence is not high

    Example:
        >>> result = detect_text_type("Type 2 diabetes mellitus")
        >>> print(result.text_type)
        'diagnosis'
        >>> print(result.confidence_level)
        'high'
    """
    try:
        logger.info(f"Starting text type detection for: '{medical_text[:50]}...'")

        prompt = get_detection_prompt(medical_text)

        result = call_groq_structured(
            prompt=prompt,
            response_model=MedicalTextTypeDetection,
            system_prompt=SYSTEM_PROMPT_FOR_TYPE_DETECTION,
            temperature=DETECTION_CONFIG["temperature"],
            max_tokens=DETECTION_CONFIG["max_tokens"],
            model=DETECTION_CONFIG["model"],
            strict=DETECTION_CONFIG["strict"]
        )

        logger.info(f"Successfully detected text type: {result.text_type} (confidence: {result.confidence_level})")
        return result
    except Exception as e:
        logger.error(f"Failed to detect text type for '{medical_text[:50]}...': {str(e)}")
        logger.warning("Returning fallback detection response due to error")
        return get_fallback_detection(medical_text, str(e))
