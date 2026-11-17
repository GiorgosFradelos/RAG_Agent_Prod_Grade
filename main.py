import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai

from llama_index.core import Settings
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage, MessageRole

from dotenv import load_dotenv
import uuid
import os
import datetime

from data_loader import load_and_chunk_pdf, embed_texts, make_token_batches
from vector_db import QdrantStorage
from custom_types import *


load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer()
)




@inngest_client.create_function(
    fn_id="RAG: Inngest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(batch: list[str], source_id: str, offset: int) -> RAGUpsertResult:
        vecs = embed_texts(batch)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{offset + i}")) for i in range(len(batch))]
        payloads = [{"source": source_id, "text": batch[i]} for i in range(len(batch))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(batch))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)

    # Split big documents into batches that each fit in ~one minute of embedding quota.
    # Each batch is its own Inngest step: finished batches are memoized, and if one
    # fails (e.g. quota exhausted) only that batch is retried.
    total, offset = 0, 0
    for n, batch in enumerate(make_token_batches(chunks_and_src.chunks)):
        result = await ctx.step.run(
            f"embed-and-upsert-{n}",
            lambda b=batch, o=offset: _upsert(b, chunks_and_src.source_id, o),
            output_type=RAGUpsertResult,
        )
        total += result.ingested
        offset += len(batch)
    return RAGUpsertResult(ingested=total).model_dump()





@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # print("\n\n|||||| Question recieved: ", question)
    # print("Found Context: ", found, " |||||\n\n")

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    # Configure LlamaIndex to use Groq
    llm = Groq(
        model="openai/gpt-oss-120b",  # Alt: llama-3.3-70b-versatile / Or your preferred active LLM
        api_key=os.getenv('GROQ_API_KEY')
    )
    Settings.llm = llm

    # Define chat messages using LlamaIndex message structures
    messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="You answer questions using only the provided context."),
        ChatMessage(role=MessageRole.USER, content=user_content)
    ]


    # res = await ctx.step.ai.infer(
    #     "llm-answer",
    #     adapter=llm,
    #     body={
    #         "max_tokens": 1024,
    #         "temperature": 0.2,
    #         "messages": [
    #             {"role": "system", "content": "You answer questions using only the provided context."},
    #             {"role": "user", "content": user_content}
    #         ]
    #     }
    # )


    def _generate_answer():
        response = llm.complete(
            prompt=f"System: You answer questions using only the provided context.\n\nUser: {user_content}"
        )
        return str(response)

    answer = await ctx.step.run("llm-answer", _generate_answer)

    return {
        "answer": answer.strip(),
        "sources": found.sources,
        "num_contexts": len(found.contexts)
    }

app = FastAPI()



inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])