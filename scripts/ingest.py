from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.service import RAGService


# Parse CLI file paths and ingest each file into the vector store.
def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest files into Qdrant.")
    parser.add_argument("paths", nargs="+", help="Files to ingest")
    args = parser.parse_args()

    service = RAGService()
    total = 0
    for path in args.paths:
        count = service.ingest_file(path)
        total += count
        print(f"Ingested {count} chunks from {path}")

    print(f"Finished ingesting {total} chunks.")


if __name__ == "__main__":
    main()
