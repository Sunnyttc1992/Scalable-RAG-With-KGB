from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import RAGService


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the RAG repository.")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()

    result = RAGService().answer(args.question)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
