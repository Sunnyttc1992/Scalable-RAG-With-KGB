# Scalable-RAG-With-KGB

A reranker-based RAG repository built on top of Qdrant. The project includes:

- file ingestion for `pdf`, `docx`, `txt`, and `md`
- chunking and metadata enrichment
- OpenAI embeddings for vector search
- Qdrant as the vector database
- an LLM reranking stage before answer generation
- CLI scripts and a small FastAPI service
- a custom Gradio interface with an NYC-inspired visual theme

## Project Structure

```text
.
├── app.py                  # FastAPI app with mounted Gradio UI
├── docker-compose.yaml     # Local Qdrant service
├── qdrant.py               # Shared Qdrant client
├── rag/
│   ├── config.py           # Environment-driven settings
│   ├── embeddings.py       # OpenAI embedding client
│   ├── repository.py       # Qdrant collection management and search
│   ├── reranker.py         # LLM-based reranking layer
│   ├── generator.py        # Final answer generation
│   └── service.py          # End-to-end ingest/query orchestration
├── ingestion/
│   ├── loaders/            # PDF and DOCX parsing
│   ├── chunking/           # Text splitting and metadata enrichment
│   └── embedding/          # Batch embedding adapter
└── scripts/
    ├── ingest.py           # CLI ingestion
    └── query.py            # CLI querying
```

## Setup

1. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the environment template and fill in your values.
   ```bash
   cp .env.example .env
   ```
4. Start Qdrant locally.
   ```bash
   docker compose up -d
   ```

## Configuration

The main environment variables are:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_RERANK_MODEL=gpt-4.1-mini
OPENAI_ANSWER_MODEL=gpt-4.1-mini
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=rag_documents
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_LIMIT=12
RERANK_LIMIT=5
```

For Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY` instead of using the local container.

## Usage

Ingest one or more files:

```bash
python scripts/ingest.py ./docs/policy.pdf ./docs/handbook.docx
```

Query from the CLI:

```bash
python scripts/query.py "What does the policy say about remote access?"
```

Run the API and Gradio interface:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

Example API calls:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"./docs/policy.pdf"}'
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize the remote access requirements."}'
```

## How It Works

1. Documents are parsed and split into chunks.
2. Each chunk is enriched with metadata and embedded with OpenAI.
3. Embeddings are stored in Qdrant.
4. A query retrieves the top vector matches from Qdrant.
5. The reranker reorders those candidates by relevance.
6. The generator answers using only the reranked context.

## License

See [LICENSE](/workspaces/Scalable-RAG-With-KGB/LICENSE).
