"""
Pydantic request/response models for the ICD Code Mapper API.
"""
from pydantic import BaseModel, Field
from typing import Any, Literal


class ICDMapRequest(BaseModel):
    """Request body for POST /icd_map."""

    query_text: str = Field(
        ...,
        min_length=1,
        description="Medical text describing a condition or procedure",
        examples=["type 2 diabetes mellitus"],
    )
    query_type: Literal["auto", "diagnosis", "procedure"] = Field(
        default="auto",
        description="Query type: 'auto' for automatic detection, 'diagnosis', or 'procedure'",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top results to return (1-50)",
    )


class ICDCodeResult(BaseModel):
    """Single ICD code result with confidence score."""

    code: str = Field(description="ICD code without dots (e.g. 'A000')")
    code_dotted: str = Field(description="ICD code with dots (e.g. 'A00.0')")
    long_description: str = Field(description="Full description of the ICD code")
    short_description: str = Field(description="Short description of the ICD code")
    category_code: str = Field(description="Category code (e.g. 'A00')")
    category_title: str = Field(description="Category title")
    score: float = Field(description="Confidence score from 0.0 to 1.0", ge=0.0, le=1.0)
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level for this match"
    )


class ICDMapResponse(BaseModel):
    """Response body for POST /icd_map."""

    query_text: str = Field(description="Original query text")
    query_type: str = Field(description="Detected or specified query type")
    top_k: int = Field(description="Number of results requested")
    results: list[ICDCodeResult] = Field(description="Ranked ICD code results")
    latencies: dict[str, float] | None = Field(
        default=None,
        description="Latency breakdown in milliseconds for each pipeline step",
    )


class HealthCheckResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "healthy"
    service: str = "ICD Code Mapper API"
    version: str = "1.0.0"
