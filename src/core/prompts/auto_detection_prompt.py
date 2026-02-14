"""
Prompt and parameters for detection of medical text
"""
from pydantic import BaseModel, Field
from typing import Literal


class MedicalTextTypeDetection(BaseModel):
    """
    Pydantic model for detecting whether medical text describes a diagnosis or procedure.

    This model uses chain of thought reasoning to accurately classify medical text
    and provides transparency in the decision-making process.
    """

    reasoning: str
    key_indicators: list[str]
    text_type: Literal["diagnosis", "procedure"]
    confidence_level: Literal["high", "medium", "low"]
    alternative_interpretation: str


SYSTEM_PROMPT_FOR_TYPE_DETECTION = """You are a medical coding expert specializing in ICD-10 classification.

Your task is to analyze medical text and determine whether it describes a DIAGNOSIS (disease, condition, disorder) or a PROCEDURE (medical intervention, surgery, examination, treatment).

You must think step-by-step and provide clear reasoning for your classification."""


def get_detection_prompt(medical_text: str) -> str:
    """
    Generate the detection prompt with chain of thought instructions.

    Args:
        medical_text: The medical text to classify

    Returns:
        Formatted prompt string with instructions and the text to analyze
    """
    return f"""Analyze the following medical text and determine if it describes a DIAGNOSIS or a PROCEDURE.

Follow this reasoning process:

1. **Extract Key Terms**: Identify the main medical terms and keywords in the text
2. **Analyze Context**: Understand how these terms are used - are they describing:
   - A condition/disease/disorder that a patient HAS? → Likely DIAGNOSIS
   - An action/intervention/treatment being PERFORMED? → Likely PROCEDURE
3. **Check Clinical Intent**: What is the primary purpose?
   - Identifying what's wrong with the patient? → DIAGNOSIS
   - Describing what will be done to the patient? → PROCEDURE
4. **Consider Edge Cases**: Some terms can be ambiguous (e.g., "cardiac arrest" is a diagnosis, but "cardiac catheterization" is a procedure)
5. **Make Final Decision**: Based on the weight of evidence

**Medical Text to Analyze:**
"{medical_text}"

**Instructions:**
- Provide detailed step-by-step reasoning
- List specific key indicators from the text that support your decision
- Classify as either "diagnosis" or "procedure"
- Assess your confidence level (high/medium/low)
- If confidence is not high, explain the alternative interpretation

Think carefully and provide your analysis in the structured format."""


DETECTION_CONFIG = {
    "model": "moonshotai/kimi-k2-instruct-0905",
    "temperature": 0.0, 
    "max_tokens": 1024,
    "strict": True
}


EXAMPLE_DIAGNOSES = [
    "Type 2 diabetes mellitus without complications",
    "Acute myocardial infarction",
    "Essential hypertension",
    "Major depressive disorder, recurrent episode",
    "Chronic obstructive pulmonary disease",
    "Migraine headache",
    "Pneumonia due to COVID-19"
]

EXAMPLE_PROCEDURES = [
    "Appendectomy",
    "Total knee replacement surgery",
    "Coronary artery bypass graft",
    "MRI scan of the brain",
    "Cataract extraction with lens implant",
    "Physical therapy for lower back pain",
    "Colonoscopy with biopsy"
]


def get_fallback_detection(medical_text: str, error_message: str = "") -> MedicalTextTypeDetection:
    """
    Generate a fallback detection response when automatic detection fails.

    This provides a safe default that indicates the system couldn't confidently
    classify the text, defaulting to "diagnosis" as it's more common in medical coding.

    Args:
        medical_text: The medical text that failed to classify
        error_message: Optional error message describing what went wrong

    Returns:
        MedicalTextTypeDetection object with low confidence and fallback classification
    """
    return MedicalTextTypeDetection(
        reasoning=f"Automatic text type detection failed{': ' + error_message if error_message else ''}. "
                  f"Defaulting to 'diagnosis' as it is the most common case in ICD-10 coding. "
                  f"Manual review recommended for text: '{medical_text[:100]}{'...' if len(medical_text) > 100 else ''}'",
        key_indicators=["Unable to extract indicators due to detection failure"],
        text_type="diagnosis",
        confidence_level="low",
        alternative_interpretation="This text could not be automatically classified. "
                                   "It may be a procedure, an ambiguous term, non-medical text, "
                                   "or require additional context for accurate classification. "
                                   "Please verify the classification manually."
    )
