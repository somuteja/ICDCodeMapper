# ICD Code Mapper

Map medical text to ICD-10 codes using hybrid search and LLM-based confidence scoring.

## Architecture

```
User Query
    |
    v
[1] Type Detection (auto mode only)
    |   LLM classifies query as "diagnosis" or "procedure"
    |   Model: Kimi-K2 via Groq | Structured JSON output
    v
[2] Hybrid Search + Reranking
    |   Dense: BGE-large-en-v1.5
    |   Sparse: BM25
    |   Fusion: Reciprocal Rank Fusion (RRF)
    |   Reranker: ms-marco-MiniLM-L-6-v2
    |   Vector DB: Qdrant (collection: icd_names)
    v
[3] Confidence Scoring (LLM)
    |   Evaluates each candidate code with medical coding expertise
    |   Assigns 0.0-1.0 score + high/medium/low confidence
    |   Model: Kimi-K2 via Groq | Structured JSON output
    v
[4] Ranked Results with Confidence Scores
```

## Project Structure

```
src/
    api/
        __init__.py
        app.py              # FastAPI application (POST /icd_map, GET /health)
        models.py           # Request/response Pydantic models
    core/
        __init__.py
        confidence_scorer.py    # LLM-based confidence scoring
        icd_mapper.py           # Main pipeline orchestrator
        text_type_detector.py   # Auto detection of diagnosis vs procedure
        prompts/
            __init__.py
            auto_detection_prompt.py        # Type detection prompt + config
            confidence_scoring_prompt.py    # Confidence scoring prompt + config
    rag/
        __init__.py
        qdrant_client.py        # Qdrant singleton client
        embeddings/
            __init__.py
            constants.py        # Model names
            dense_embeddings.py
            sparse_embeddings.py
            embedding_manager.py    # Singleton embedding manager
            reranker.py
        retrieval/
            __init__.py
            constants.py        # Search defaults
            hybrid_search.py    # Hybrid search implementation
    utils/
        __init__.py
        gemini_llms.py      # Gemini API client
        groq_llms.py        # Groq API client (structured JSON output)
    upload_scripts/         # Data preprocessing and Qdrant upload
    test_scripts/           # Test scripts
    main.py                 # Application entry point
    requirements.txt
    .env                    # Environment variables (not committed)
```

## Setup

### Prerequisites

- Python 3.11+
- Qdrant Cloud account (or local Qdrant instance)
- Groq API key

### Environment Variables

Create a `.env` file in `src/`:

```env
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
GROQ_API_KEY=your-groq-api-key
```

### Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

## Running the API

```bash
cd src
python main.py
```

The API starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## API

### POST /icd_map

Map medical text to ICD-10 codes.

**Request:**

```json
{
  "query_text": "type 2 diabetes",
  "query_type": "auto",
  "top_k": 5
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query_text` | string | required | Medical text to map |
| `query_type` | string | `"auto"` | `"auto"`, `"diagnosis"`, or `"procedure"` |
| `top_k` | int | `5` | Number of results (1-50) |

**Response:**

```json
{
  "query_text": "type 2 diabetes",
  "query_type": "diagnosis",
  "top_k": 5,
  "results": [
    {
      "code": "E119",
      "code_dotted": "E11.9",
      "long_description": "Type 2 diabetes mellitus without complications",
      "short_description": "Type 2 diabetes mellitus without complications",
      "category_code": "E11",
      "category_title": "Type 2 diabetes mellitus",
      "score": 0.95,
      "confidence": "high"
    }
  ],
  "latencies": {
    "type_detection_ms": 230.5,
    "hybrid_search_ms": 450.2,
    "confidence_scoring_ms": 680.1,
    "total_ms": 1360.8
  }
}
```

### GET /health

Health check endpoint.

```json
{
  "status": "healthy",
  "service": "ICD Code Mapper API",
  "version": "1.0.0"
}
```

## Dataset

- Source: [ICD-10 CSV](https://raw.githubusercontent.com/k4m1113/ICD-10-CSV/master/codes.csv)
- 71,704 diagnosis codes
- Stored in Qdrant with dense (BGE-large) and sparse (BM25) vectors

## Models Used

| Purpose | Model | Provider |
|---------|-------|----------|
| Dense embeddings | BAAI/bge-large-en-v1.5 | FastEmbed |
| Sparse embeddings | Qdrant/bm25 | FastEmbed |
| Reranking | Xenova/ms-marco-MiniLM-L-6-v2 | FastEmbed |
| Type detection | moonshotai/kimi-k2-instruct-0905 | Groq |
| Confidence scoring | moonshotai/kimi-k2-instruct-0905 | Groq |
