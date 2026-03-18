from __future__ import annotations

import json

from openai import OpenAI

from rag.config import settings
from rag.types import ScoredChunk


class LLMReranker:
    # Create the reranker client and choose the model used for relevance scoring.
    def __init__(self, model: str | None = None):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to rerank results.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_rerank_model

    # Ask the LLM to score candidate passages and keep the highest-ranked ones.
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
            "You are a high-precision relevance ranking system for DC Water customer support.\n"
            "Your task is to evaluate and rank candidate passages based strictly on their relevance "
            "to the user query.\n\n"

            "Instructions:\n"
            "- Focus ONLY on semantic relevance to the query.\n"
            "- Prioritize passages that directly answer or support the query.\n"
            "- Penalize irrelevant, vague, or redundant passages.\n"
            "- Do NOT generate explanations or additional text.\n"
            "- Do NOT modify or rewrite passages.\n"
            "- Only return the ranking.\n\n"

            "Scoring Guidelines:\n"
            "- Score must be a float between 0 and 1.\n"
            "- 1.0 = highly relevant and directly answers the query\n"
            "- 0.7–0.9 = relevant but partially complete\n"
            "- 0.4–0.6 = loosely related\n"
            "- 0.0–0.3 = irrelevant or off-topic\n\n"

            "Output Format (STRICT JSON ONLY):\n"
            "{\n"
            '  "ranked_passages": [\n'
            '    {"index": <int>, "score": <float>},\n'
            "    ...\n"
            "  ]\n"
            "}\n\n"

            "Rules:\n"
            "- Do NOT include explanations, reasoning, or extra keys.\n"
            "- Do NOT include text outside the JSON.\n"
            "- Ensure valid JSON format.\n"
            "- Rank ALL provided passages.\n"
            "- Sort results in descending order of score.\n\n"

            f"Query:\n{query}\n\n"
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
