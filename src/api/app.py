"""
FastAPI application for the ICD Code Mapper.

Endpoints:
    POST /icd_map  - Map medical text to ICD-10 codes
    GET  /health   - Health check
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    HealthCheckResponse,
    ICDCodeResult,
    ICDMapRequest,
    ICDMapResponse,
)
from core.icd_mapper import map_icd_codes

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load heavyweight singletons at startup."""
    logger.info("Starting ICD Code Mapper API...")

    from rag.embeddings.embedding_manager import embedding_manager
    logger.info("✓ Embedding manager initialized (dense + sparse + reranker)")

    
    _ = embedding_manager.dense_model
    _ = embedding_manager.sparse_model
    _ = embedding_manager.rerank_model
    logger.info("✓ All embedding models warmed up")

    from rag.qdrant_client import get_qdrant_client
    _ = get_qdrant_client()
    logger.info("✓ Qdrant client connected")

    try:
        from utils.groq_llms import call_groq
        _ = call_groq(
            prompt="test",
            system_prompt="Reply with 'ok'",
            max_tokens=10,
            temperature=0.0
        )
        logger.info("✓ Groq API validated")
    except Exception as e:
        logger.warning(f"Groq API validation failed: {e}")

    logger.info("ICD Code Mapper API ready")
    yield
    logger.info("Shutting down ICD Code Mapper API.")


app = FastAPI(
    title="ICD Code Mapper API",
    description="Map medical text to ICD-10 codes using hybrid search and LLM-based confidence scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    return HealthCheckResponse()


@app.post(
    "/icd_map",
    response_model=ICDMapResponse,
    summary="Map medical text to ICD-10 codes",
)
async def icd_map(request: ICDMapRequest) -> ICDMapResponse:
    """
    Map medical query text to relevant ICD-10 codes with confidence scores.

    Pipeline: type detection (if auto) -> hybrid search -> reranking -> LLM confidence scoring.
    """
    try:
        logger.info(
            "Request: query='%s', type=%s, top_k=%d",
            request.query_text[:50],
            request.query_type,
            request.top_k,
        )

        result = map_icd_codes(
            query_text=request.query_text,
            query_type=request.query_type,
            top_k=request.top_k,
        )

        response = ICDMapResponse(
            query_text=result.query_text,
            query_type=result.query_type,
            top_k=result.top_k,
            results=[ICDCodeResult(**r) for r in result.results],
            latencies=result.latencies,
        )

        logger.info(
            "Response: %d results, total_latency=%.1fms",
            len(response.results),
            result.latencies.get("total_ms", 0),
        )
        return response

    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except RuntimeError as e:
        logger.error("Pipeline error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        )
