import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    openai_rerank_model: str = os.getenv("OPENAI_RERANK_MODEL", "gpt-4.1-mini")
    openai_answer_model: str = os.getenv("OPENAI_ANSWER_MODEL", "gpt-4.1-mini")
    qdrant_url: str | None = os.getenv(
        "QDRANT_URL",
        os.getenv("QDRANT_ENDPOINT_URL", "http://localhost:6333"),
    )
    qdrant_api_key: str | None = os.getenv("QDRANT_API_KEY")
    qdrant_collection_name: str = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "rag_documents",
    )
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    retrieval_limit: int = int(os.getenv("RETRIEVAL_LIMIT", "12"))
    rerank_limit: int = int(os.getenv("RERANK_LIMIT", "5"))


settings = Settings()
