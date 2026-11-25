"""Provider interfaces. The pipeline depends only on these, never on a vendor SDK."""

from typing import Protocol

from rag_production_agent.models import Generation


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class LLM(Protocol):
    def generate(self, system: str, user: str) -> Generation: ...
