from __future__ import annotations

from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag.config import settings
from rag.types import ScoredChunk


class QdrantRepository:
    def __init__(
        self,
        collection_name: str | None = None,
        url: str | None = None,
        api_key: str | None = None,
    ):
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.client = QdrantClient(
            url=url or settings.qdrant_url,
            api_key=api_key or settings.qdrant_api_key,
        )

    def collection_exists(self) -> bool:
        collections = self.client.get_collections().collections
        return any(collection.name == self.collection_name for collection in collections)

    def ensure_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
    ) -> int:
        if not chunks:
            return 0

        self.ensure_collection(vector_size=len(embeddings[0]))
        points = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            payload = {
                "text": chunk["text"],
                **chunk["metadata"],
            }
            points.append(
                models.PointStruct(
                    id=str(uuid4()),
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(points)

    def search(self, query_vector: list[float], limit: int) -> list[ScoredChunk]:
        if not self.collection_exists():
            return []

        if hasattr(self.client, "search"):
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            results = response.points

        chunks: list[ScoredChunk] = []
        for item in results:
            payload = dict(item.payload or {})
            text = str(payload.pop("text", ""))
            chunks.append(
                ScoredChunk(
                    id=str(item.id),
                    text=text,
                    metadata=payload,
                    vector_score=float(item.score),
                )
            )
        return chunks

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return self.client.count(collection_name=self.collection_name, exact=True).count
