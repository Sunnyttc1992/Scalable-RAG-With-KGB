from __future__ import annotations

import re
from typing import Any

from openai import OpenAI

from rag.config import settings
from rag.types import ScoredChunk


# Clean the model output so it reads well in the chat UI.
def _normalize_answer_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\\\((.*?)\\\)", r"\1", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\\\[(.*?)\\\]", r"\1", cleaned, flags=re.DOTALL)
    cleaned = cleaned.replace("**", "")
    cleaned = re.sub(r"^\s*[-*]\s+", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


class AnswerGenerator:
    # Create the answer-generation client and choose the chat model.
    def __init__(self, model: str | None = None):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to generate answers.")
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = model or settings.openai_answer_model

    # Generate a final answer from retrieved chunks and recent conversation turns.
    def answer(
        self,
        query: str,
        chunks: list[ScoredChunk],
        history: list[dict[str, Any]] | None = None,
    ) -> str:
        if not chunks:
            return "I could not find any relevant context in the knowledge base."

        conversation_history = self._format_history(history or [])
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
                        "text lists. Use the conversation history only to resolve "
                        "follow-up questions like 'that', 'it', or 'compare those'. "
                        "If the context is not enough, say so clearly and do not "
                        "make up facts."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Conversation history:\n{conversation_history}\n\n"
                        f"Question: {query}\n\n"
                        f"Context:\n{context}"
                    ),
                },
            ],
        )
        return _normalize_answer_text(response.choices[0].message.content or "")

    # Convert recent chat messages into plain text for the prompt.
    def _format_history(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "No prior conversation."

        formatted_turns: list[str] = []
        for item in history[-6:]:
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            formatted_turns.append(f"{speaker}: {content}")

        return "\n".join(formatted_turns) if formatted_turns else "No prior conversation."
