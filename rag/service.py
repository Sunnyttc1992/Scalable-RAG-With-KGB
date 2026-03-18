from __future__ import annotations

from pathlib import Path
from typing import Any

from ingestion.chunking.metadata import enrich_metadata
from ingestion.chunking.splitter import split_text
from ingestion.loaders.docx import parse_docx_bytes
from ingestion.loaders.pdf import parse_pdf_bytes
from rag.config import settings
from rag.embeddings import OpenAIEmbedder
from rag.generator import AnswerGenerator
from rag.repository import QdrantRepository
from rag.reranker import LLMReranker


class RAGService:
    def __init__(self):
        self.embedder = OpenAIEmbedder()
        self.repository = QdrantRepository()
        self.reranker = LLMReranker()
        self.generator = AnswerGenerator()

    def ingest_file(self, path: str | Path) -> int:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text, metadata = self._load_file(file_path)
        chunks = split_text(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        prepared_chunks = []
        for chunk in chunks:
            prepared_chunks.append(
                {
                    "text": chunk["text"],
                    "metadata": enrich_metadata(
                        {**metadata, **chunk["metadata"]},
                        chunk["text"],
                    ),
                }
            )

        embeddings = self.embedder.embed_texts([chunk["text"] for chunk in prepared_chunks])
        return self.repository.upsert_chunks(prepared_chunks, embeddings)

    def retrieve(self, query: str):
        query_vector = self.embedder.embed_query(query)
        retrieved = self.repository.search(
            query_vector=query_vector,
            limit=settings.retrieval_limit,
        )
        return self.reranker.rerank(
            query=query,
            chunks=retrieved,
            top_k=settings.rerank_limit,
        )

    def answer(self, query: str, history: list[dict[str, Any]] | None = None) -> dict:
        retrieval_query = self._build_retrieval_query(query, history or [])
        ranked_chunks = self.retrieve(retrieval_query)
        return {
            "answer": self.generator.answer(query, ranked_chunks, history=history),
            "matches": [
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                    "vector_score": chunk.vector_score,
                    "rerank_score": chunk.rerank_score,
                }
                for chunk in ranked_chunks
            ],
        }

    def _build_retrieval_query(
        self,
        query: str,
        history: list[dict[str, Any]],
    ) -> str:
        if not history:
            return query

        recent_turns: list[str] = []
        for item in history[-4:]:
            role = item.get("role", "user")
            content = str(item.get("content", "")).strip()
            if not content:
                continue
            prefix = "User" if role == "user" else "Assistant"
            recent_turns.append(f"{prefix}: {content}")

        if not recent_turns:
            return query

        return (
            "Use the recent conversation to interpret the latest user question.\n"
            f"Recent conversation:\n" + "\n".join(recent_turns) + f"\nLatest question: {query}"
        )

    def _load_file(self, path: Path) -> tuple[str, dict]:
        raw_bytes = path.read_bytes()
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return parse_pdf_bytes(raw_bytes, path.name)
        if suffix == ".docx":
            return parse_docx_bytes(raw_bytes, path.name)
        if suffix in {".txt", ".md"}:
            text = raw_bytes.decode("utf-8")
            return text, {"filename": path.name, "type": suffix.lstrip(".")}
        raise ValueError(f"Unsupported file type: {suffix}")
