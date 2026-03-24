"""HTTP client for the ICD Code Mapper backend API."""

from __future__ import annotations

import requests

from models import HealthStatus, ICDCodeResult, ICDMapResponse, Latencies, QueryType


class ICDMapperClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> HealthStatus:
        """Check whether the backend is reachable and healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            data = resp.json()
            return HealthStatus(
                online=True,
                status=data.get("status", ""),
                service=data.get("service", ""),
                version=data.get("version", ""),
            )
        except requests.ConnectionError:
            return HealthStatus(online=False, error="Cannot connect to the API server.")
        except requests.Timeout:
            return HealthStatus(online=False, error="Health check timed out.")
        except Exception as exc:  # noqa: BLE001
            return HealthStatus(online=False, error=str(exc))

    def map_icd(
        self,
        query_text: str,
        query_type: QueryType = "auto",
        top_k: int = 5,
    ) -> ICDMapResponse:
        """
        Call POST /icd_map and return a parsed ICDMapResponse.

        Raises:
            requests.HTTPError: on 4xx/5xx responses.
            requests.ConnectionError: when the server is unreachable.
            requests.Timeout: when the request exceeds self.timeout seconds.
        """
        payload = {"query_text": query_text, "query_type": query_type, "top_k": top_k}
        resp = requests.post(
            f"{self.base_url}/icd_map",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json())

    @staticmethod
    def _parse_response(data: dict) -> ICDMapResponse:
        results = [
            ICDCodeResult(
                code=r["code"],
                code_dotted=r["code_dotted"],
                long_description=r["long_description"],
                short_description=r["short_description"],
                category_code=r["category_code"],
                category_title=r["category_title"],
                score=float(r["score"]),
                confidence=r["confidence"],
            )
            for r in data.get("results", [])
        ]

        raw_lat = data.get("latencies", {})
        latencies = Latencies(
            type_detection_ms=raw_lat.get("type_detection_ms"),
            hybrid_search_ms=float(raw_lat.get("hybrid_search_ms", 0)),
            confidence_scoring_ms=float(raw_lat.get("confidence_scoring_ms", 0)),
            total_ms=float(raw_lat.get("total_ms", 0)),
        )

        return ICDMapResponse(
            query_text=data["query_text"],
            query_type=data["query_type"],
            top_k=data["top_k"],
            results=results,
            latencies=latencies,
        )
