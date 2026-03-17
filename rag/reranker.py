from __future__ import annotations

import json

from openai import OpenAI

from rag.config import settings
from rag.types import ScoredChunk


class LLMReranker:
    def __init__(self, model: str | None = None):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to rerank results.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_rerank_model

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_k: int,
    ) -> list[ScoredChunk]:
        if not chunks:
            return []

        candidate_block = "\n\n".join(
            f"[{index + 1}] {chunk.text[:1400]}"
            for index, chunk in enumerate(chunks)
        )
        prompt = (
            "You are a retrieval reranker. Rank the passages by relevance to the user "
            "query. Return strict JSON with a single key named ranked_passages. "
            "Each item must include index and score, where score is 0 to 1.\n\n"
            f"Query: {query}\n\n"
            f"Passages:\n{candidate_block}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "Return only valid JSON.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
            content = response.choices[0].message.content or '{"ranked_passages":[]}'
            parsed = json.loads(content)
        except Exception:
            fallback = sorted(chunks, key=lambda chunk: chunk.vector_score, reverse=True)
            return fallback[:top_k]

        ranked: list[ScoredChunk] = []
        seen_indexes: set[int] = set()
        for item in parsed.get("ranked_passages", []):
            raw_index = item.get("index")
            if not isinstance(raw_index, int):
                continue
            index = raw_index - 1
            if index < 0 or index >= len(chunks) or index in seen_indexes:
                continue
            seen_indexes.add(index)
            chunk = chunks[index]
            chunk.rerank_score = float(item.get("score", 0.0))
            ranked.append(chunk)
            if len(ranked) >= top_k:
                return ranked

        fallback = sorted(chunks, key=lambda chunk: chunk.vector_score, reverse=True)
        return fallback[:top_k]
