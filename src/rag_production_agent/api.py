"""FastAPI app: HTTP endpoints for the UI plus the Inngest serve endpoint."""

import hashlib
import logging
import re

import inngest
import inngest.fast_api
from fastapi import FastAPI, HTTPException, Request, UploadFile, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client.http.exceptions import ResponseHandlingException

from rag_production_agent import __version__, pipeline
from rag_production_agent.config import get_settings
from rag_production_agent.models import Answer
from rag_production_agent.store import get_store
from rag_production_agent.workflows import FUNCTIONS, INGEST_EVENT, inngest_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Production Agent", version=__version__)
inngest.fast_api.serve(app, inngest_client, FUNCTIONS)


@app.exception_handler(ResponseHandlingException)
async def vector_store_unreachable(request: Request, exc: ResponseHandlingException) -> JSONResponse:
    """Turn a raw connection error into a clear 503 instead of a bare 500."""
    url = get_settings().qdrant_url
    logger.error("Vector store unreachable at %s: %s", url, exc)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": f"Vector store unreachable at {url}. Is Qdrant running?"},
    )


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)


class IngestResponse(BaseModel):
    source_id: str
    event_id: str
    sha256: str


@app.get("/health")
def health() -> dict:
    """Liveness: the process is up."""
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready() -> dict:
    """Readiness: dependencies are reachable."""
    try:
        reachable = get_store().ping()
    except ResponseHandlingException:
        reachable = False
    if not reachable:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "vector store unreachable")
    return {"status": "ready"}


@app.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(file: UploadFile) -> IngestResponse:
    """Save the PDF and hand ingestion to Inngest (it can take minutes on the free tier)."""
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "only PDF files are supported")

    data = await file.read()
    sha256 = hashlib.sha256(data).hexdigest()
    filename = re.sub(r"[^\w.\- ]", "_", file.filename or "upload.pdf")  # no path traversal

    upload_dir = get_settings().upload_dir
    upload_dir.mkdir(parents=True, exist_ok=True)
    path = upload_dir / filename
    path.write_bytes(data)

    # Same file → same event ID → Inngest drops the duplicate.
    ids = await inngest_client.send(
        inngest.Event(
            name=INGEST_EVENT,
            id=f"ingest-{sha256}",
            data={"pdf_path": str(path.resolve()), "source_id": filename},
        )
    )
    return IngestResponse(source_id=filename, event_id=ids[0], sha256=sha256)


@app.post("/query")
def query(req: QueryRequest) -> Answer:
    """Answer a question synchronously (retrieve → generate)."""
    return pipeline.answer(req.question.strip(), req.top_k)


def run() -> None:
    import uvicorn

    uvicorn.run("rag_production_agent.api:app", host="0.0.0.0", port=8000)
