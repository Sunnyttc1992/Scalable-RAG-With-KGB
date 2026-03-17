from __future__ import annotations

from itertools import islice

from openai import OpenAI

from rag.config import settings


def _batched(items: list[str], batch_size: int):
    iterator = iter(items)
    while batch := list(islice(iterator, batch_size)):
        yield batch


class OpenAIEmbedder:
    def __init__(self, model: str | None = None):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to create embeddings.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_embedding_model

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in _batched(texts, batch_size):
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

