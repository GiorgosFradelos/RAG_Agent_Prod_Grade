from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag_production_agent.models import RetrievedChunk


class QdrantStore:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "docs",
        dim: int = 3072,
        api_key: str | None = None,
    ):
        if url == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection
        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            return
        # Fail loudly if the embedding model changed but the collection wasn't recreated.
        existing = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(existing, "size", None)
        if size is not None and size != self.dim:
            raise ValueError(
                f"Collection '{self.collection}' has dimension {size}, but the embedder produces {self.dim}. "
                "Recreate the collection (or use a new QDRANT_COLLECTION) and re-ingest."
            )

    def upsert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(ids, vectors, payloads, strict=True)]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k,
        )
        chunks = []
        for point in results.points:
            payload = point.payload or {}
            if text := payload.get("text"):
                chunks.append(RetrievedChunk(text=text, source=payload.get("source", ""), score=point.score))
        return chunks

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False
