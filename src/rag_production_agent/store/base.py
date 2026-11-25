from typing import Protocol

from rag_production_agent.models import RetrievedChunk


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None: ...

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]: ...

    def ping(self) -> bool: ...
