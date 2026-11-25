"""The RAG core: plain functions with no web or orchestration code.

The API, the Inngest workflows, the tests and the eval harness all call these.
Dependencies (embedder, store, LLM) can be passed in; by default they come from settings.
"""

import uuid
from pathlib import Path

from rag_production_agent.config import get_settings
from rag_production_agent.ingestion.loader import load_and_chunk_pdf
from rag_production_agent.models import Answer, ChunkedDocument, Generation, RetrievalResult
from rag_production_agent.prompts import NO_CONTEXT_ANSWER, PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from rag_production_agent.providers import LLM, Embedder, get_embedder, get_llm
from rag_production_agent.providers.ratelimit import make_token_batches
from rag_production_agent.store import VectorStore, get_store

# --- Ingestion ------------------------------------------------------------------


def chunk_document(path: str | Path, source_id: str | None = None) -> ChunkedDocument:
    s = get_settings()
    chunks = load_and_chunk_pdf(path, s.chunk_size, s.chunk_overlap)
    return ChunkedDocument(source_id=source_id or Path(path).name, chunks=chunks)


def ingestion_batches(chunks: list[str]) -> list[list[str]]:
    """Split chunks into batches that each fit inside ~one minute of embedding quota."""
    return make_token_batches(chunks, get_settings().embed_tpm_limit)


def chunk_ids(source_id: str, start: int, count: int) -> list[str]:
    """Deterministic IDs: re-ingesting a document overwrites its points instead of duplicating them."""
    return [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{start + i}")) for i in range(count)]


def index_chunks(
    chunks: list[str],
    source_id: str,
    offset: int = 0,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> int:
    """Embed chunks and upsert them. `offset` is the index of chunks[0] within the document."""
    if not chunks:
        return 0
    embedder = embedder or get_embedder()
    store = store or get_store()
    vectors = embedder.embed(chunks)
    ids = chunk_ids(source_id, offset, len(chunks))
    payloads = [{"source": source_id, "text": text, "chunk_index": offset + i} for i, text in enumerate(chunks)]
    store.upsert(ids, vectors, payloads)
    return len(chunks)


def ingest_pdf(
    path: str | Path,
    source_id: str | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> int:
    """Synchronous end-to-end ingestion (scripts, eval, tests). The app uses the Inngest workflow."""
    doc = chunk_document(path, source_id)
    return index_chunks(doc.chunks, doc.source_id, embedder=embedder, store=store)


# --- Query ----------------------------------------------------------------------


def retrieve(
    question: str,
    top_k: int | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
) -> RetrievalResult:
    embedder = embedder or get_embedder()
    store = store or get_store()
    query_vector = embedder.embed([question])[0]
    return RetrievalResult(chunks=store.search(query_vector, top_k or get_settings().top_k))


def generate(question: str, contexts: list[str], *, llm: LLM | None = None) -> Generation:
    if not contexts:
        # Nothing retrieved: don't spend tokens asking the LLM to guess.
        return Generation(text=NO_CONTEXT_ANSWER, model="none")
    llm = llm or get_llm()
    return llm.generate(SYSTEM_PROMPT, build_user_prompt(question, contexts))


def answer(
    question: str,
    top_k: int | None = None,
    *,
    embedder: Embedder | None = None,
    store: VectorStore | None = None,
    llm: LLM | None = None,
) -> Answer:
    retrieved = retrieve(question, top_k, embedder=embedder, store=store)
    generation = generate(question, retrieved.contexts, llm=llm)
    return Answer(
        answer=generation.text,
        sources=retrieved.sources,
        num_contexts=len(retrieved.chunks),
        prompt_version=PROMPT_VERSION,
    )
