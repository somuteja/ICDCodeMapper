"""Data models representing API request and response shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryType = Literal["auto", "diagnosis", "procedure"]


@dataclass
class ICDCodeResult:
    code: str
    code_dotted: str
    long_description: str
    short_description: str
    category_code: str
    category_title: str
    score: float
    confidence: Literal["high", "medium", "low"]


@dataclass
class Latencies:
    type_detection_ms: float | None
    hybrid_search_ms: float
    confidence_scoring_ms: float
    total_ms: float


@dataclass
class ICDMapResponse:
    query_text: str
    query_type: str
    top_k: int
    results: list[ICDCodeResult]
    latencies: Latencies


@dataclass
class HealthStatus:
    online: bool
    status: str = ""
    service: str = ""
    version: str = ""
    error: str = ""
