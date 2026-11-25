from functools import lru_cache

from rag_production_agent.config import get_settings, secret
from rag_production_agent.store.base import VectorStore
from rag_production_agent.store.qdrant import QdrantStore

__all__ = ["QdrantStore", "VectorStore", "get_store"]


@lru_cache
def get_store() -> VectorStore:
    s = get_settings()
    return QdrantStore(
        url=s.qdrant_url,
        collection=s.qdrant_collection,
        dim=s.embed_dim,
        api_key=secret(s.qdrant_api_key),
    )
