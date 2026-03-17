from __future__ import annotations

import re

from openai import OpenAI

from rag.config import settings
from rag.types import ScoredChunk


def _normalize_answer_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\\\((.*?)\\\)", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[(.*?)\\\]", r"\1", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


class AnswerGenerator:
    def __init__(self, model: str | None = None):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to generate answers.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_answer_model

    def answer(self, query: str, chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "I could not find any relevant context in the knowledge base."

        context = "\n\n".join(
            f"Source {index + 1}:\n{chunk.text}"
            for index, chunk in enumerate(chunks)
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer using only the provided context. Write in clear, "
                        "natural plain English for a normal chat UI. Do not use "
                        "LaTeX, Markdown math, escaped symbols, or notation like "
                        "\\(H\\). If you mention variables from a formula, write "
                        "them in simple text such as 'H = pumping head, in pounds "
                        "per square inch.' Prefer readable sentences or simple plain "
                        "text lists. If the context is not enough, say so clearly "
                        "and do not make up facts."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Question: {query}\n\nContext:\n{context}",
                },
            ],
        )
        return _normalize_answer_text(response.choices[0].message.content or "")
