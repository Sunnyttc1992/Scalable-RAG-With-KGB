from dataclasses import dataclass
from typing import Any


@dataclass
class ScoredChunk:
    id: str
    text: str
    metadata: dict[str, Any]
    vector_score: float
    rerank_score: float | None = None

