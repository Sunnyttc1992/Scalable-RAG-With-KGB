# pipelines/ingestion/chunking/metadata.py
import hashlib
import datetime

# Add a content hash and ingest timestamp to each chunk's metadata.
def enrich_metadata(base_metadata: dict, content: str) -> dict:
    return {
        **base_metadata,
        "chunk_hash": hashlib.md5(content.encode('utf-8')).hexdigest(),
        "ingested_at": datetime.datetime.utcnow().isoformat()
    }
