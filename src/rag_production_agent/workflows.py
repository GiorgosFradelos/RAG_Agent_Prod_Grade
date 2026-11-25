"""Inngest functions: durable, retried, observable wrappers around pipeline.py."""

import logging

import inngest

from rag_production_agent import pipeline
from rag_production_agent.config import get_settings
from rag_production_agent.models import Answer, ChunkedDocument, Generation, IngestResult, RetrievalResult

INGEST_EVENT = "rag/ingest_pdf"
QUERY_EVENT = "rag/query_pdf_ai"

inngest_client = inngest.Inngest(
    app_id=get_settings().inngest_app_id,
    logger=logging.getLogger("uvicorn"),
    is_production=not get_settings().inngest_dev,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(fn_id="RAG: Ingest PDF", trigger=inngest.TriggerEvent(event=INGEST_EVENT))
async def ingest_pdf(ctx: inngest.Context) -> dict:
    pdf_path = ctx.event.data["pdf_path"]
    source_id = ctx.event.data.get("source_id", pdf_path)

    doc = await ctx.step.run(
        "load-and-chunk",
        lambda: pipeline.chunk_document(pdf_path, source_id),
        output_type=ChunkedDocument,
    )

    # One step per batch: finished batches are memoized, and a failure
    # (e.g. exhausted quota) only retries that batch.
    total, offset = 0, 0
    for n, batch in enumerate(pipeline.ingestion_batches(doc.chunks)):
        result = await ctx.step.run(
            f"embed-and-upsert-{n}",
            lambda b=batch, o=offset: IngestResult(ingested=pipeline.index_chunks(b, doc.source_id, o)),
            output_type=IngestResult,
        )
        total += result.ingested
        offset += len(batch)
    return IngestResult(ingested=total).model_dump()


@inngest_client.create_function(fn_id="RAG: Query PDF", trigger=inngest.TriggerEvent(event=QUERY_EVENT))
async def query_pdf(ctx: inngest.Context) -> dict:
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", get_settings().top_k))

    retrieved = await ctx.step.run(
        "embed-and-search",
        lambda: pipeline.retrieve(question, top_k),
        output_type=RetrievalResult,
    )
    generation = await ctx.step.run(
        "llm-answer",
        lambda: pipeline.generate(question, retrieved.contexts),
        output_type=Generation,
    )
    return Answer(
        answer=generation.text,
        sources=retrieved.sources,
        num_contexts=len(retrieved.chunks),
        prompt_version=pipeline.PROMPT_VERSION,
    ).model_dump()


FUNCTIONS = [ingest_pdf, query_pdf]
