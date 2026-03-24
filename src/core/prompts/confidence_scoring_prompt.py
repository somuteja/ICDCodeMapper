"""
Prompt, Pydantic models, and configuration for ICD code confidence scoring.

This module provides the structured output models and prompt templates used
by the confidence scorer to evaluate hybrid search results and assign
final confidence scores to each ICD code candidate.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ICDCodeEvaluation(BaseModel):
    """Individual ICD code evaluation with reasoning and confidence score."""

    model_config = ConfigDict(extra='forbid')

    code: str = Field(description="The ICD code being evaluated (dotted format, e.g. 'A00.0')")
    relevance_score: float = Field(
        description="Relevance score from 0.0 to 1.0 where 1.0 is a perfect match",
        ge=0.0,
        le=1.0,
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level for this match"
    )
    match_reasoning: str = Field(
        description="Brief explanation of why this code does or does not match the query"
    )


class ConfidenceScoringResult(BaseModel):
    """
    Complete confidence scoring result with chain-of-thought reasoning.

    The LLM evaluates candidate ICD codes returned by hybrid search and
    reranking, then assigns each a relevance score and confidence level.
    """

    model_config = ConfigDict(extra='forbid')

    reasoning: str = Field(
        description="Step-by-step analysis of the query and how it maps to ICD coding"
    )
    query_understanding: str = Field(
        description="The LLM's interpretation of what the user is looking for"
    )
    evaluated_codes: list[ICDCodeEvaluation] = Field(
        description="List of evaluated ICD codes with scores and reasoning, ordered by relevance"
    )
    best_code: str = Field(
        description="The single most relevant ICD code from the evaluated results"
    )
    overall_confidence: Literal["high", "medium", "low"] = Field(
        description="Overall confidence in the top recommendations"
    )


SYSTEM_PROMPT_FOR_CONFIDENCE_SCORING = """You are an expert medical coder specializing in ICD-10 classification and code assignment.

Your task is to evaluate candidate ICD-10 codes returned by a retrieval system and assign precise confidence scores based on clinical relevance to the user's query.

Evaluation criteria:
1. **Semantic match** - Does the code description accurately capture the meaning of the query?
2. **Clinical accuracy** - Is this the correct ICD-10 code for the described condition?
3. **Specificity** - Is the code at the appropriate level of specificity (not too broad, not too narrow)?
4. **Terminology alignment** - Do the medical terms align between query and code description?

Scoring rubric:
- 0.90 - 1.00: Exact or near-exact match, correct specificity (HIGH confidence)
- 0.70 - 0.89: Good match with minor specificity or terminology differences (MEDIUM confidence)
- 0.50 - 0.69: Partial match, related but not the best code (LOW confidence)
- Below 0.50: Poor match, only loosely related (LOW confidence)

You must think step-by-step, provide clear reasoning for each code evaluation, and identify the single best matching code."""


def get_confidence_scoring_prompt(
    user_query: str,
    search_results: list[dict],
    query_type: str,
) -> str:
    """
    Generate the confidence scoring prompt with candidate results.

    Args:
        user_query: The user's original medical query text.
        search_results: List of dicts from hybrid search, each containing
            code_dotted, long_description, category_title, and score.
        query_type: The detected or specified query type ("diagnosis" or "procedure").

    Returns:
        Formatted prompt string for the LLM.
    """
    formatted_results = []
    for idx, result in enumerate(search_results, 1):
        formatted_results.append(
            f"{idx}. Code: {result.get('code_dotted', 'N/A')}\n"
            f"   Description: {result.get('long_description', 'N/A')}\n"
            f"   Category: {result.get('category_title', 'N/A')}"
        )

    results_text = "\n\n".join(formatted_results)

    return f"""Evaluate the following ICD-10 code candidates for the given medical query.

**User Query:** "{user_query}"
**Query Type:** {query_type}

**Candidate ICD Codes** (from hybrid retrieval + reranking):

{results_text}

**Instructions:**

1. Analyze the user's query to understand the exact medical condition or procedure described.
2. For each candidate code, evaluate semantic match, clinical accuracy, and specificity.
3. Assign a relevance score (0.0 to 1.0) and confidence level (high/medium/low) to each code.
4. Provide brief reasoning for each evaluation.
5. **Order all evaluated codes by relevance score (highest to lowest).**
6. Identify the single best matching code.

**Required JSON output fields (use these exact field names):**
- `reasoning`: step-by-step analysis of the query and ICD mapping
- `query_understanding`: your interpretation of what the user is looking for
- `evaluated_codes`: list of objects, each with `code`, `relevance_score`, `confidence`, `match_reasoning`
- `best_code`: the single most relevant ICD code
- `overall_confidence`: "high", "medium", or "low"

Evaluate all candidate codes and provide your structured analysis."""


CONFIDENCE_SCORING_CONFIG = {
    "model": "moonshotai/kimi-k2-instruct-0905",
    "temperature": 0.0,
    "max_tokens": 5000,
    "strict": True,
}
