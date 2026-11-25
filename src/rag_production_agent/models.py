"""Data models shared by the pipeline, the API and the Inngest workflows."""

from pydantic import BaseModel


class ChunkedDocument(BaseModel):
    source_id: str
    chunks: list[str]


class IngestResult(BaseModel):
    ingested: int


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]

    @property
    def contexts(self) -> list[str]:
        return [c.text for c in self.chunks]

    @property
    def sources(self) -> list[str]:
        # Unique, in rank order
        return list(dict.fromkeys(c.source for c in self.chunks if c.source))


class Generation(BaseModel):
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Answer(BaseModel):
    answer: str
    sources: list[str]
    num_contexts: int
    prompt_version: str
