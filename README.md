# Water Management Handbook Assistant

This project is a retrieval-augmented question-answering app for water management reference material. It indexes handbook content into Qdrant, retrieves relevant passages with OpenAI embeddings, reranks those passages with an LLM, and produces grounded answers with visible source snippets in a Gradio interface.

The current setup is well suited for asking questions against the indexed water management handbook, such as:

- unit pumping cost formulas
- pumping head and efficiency definitions
- irrigation and water delivery guidance
- handbook-based operational or reference questions

## What The App Does

- ingests `pdf`, `docx`, `txt`, and `md` documents
- splits files into chunks and enriches metadata
- stores embeddings in Qdrant
- reranks retrieved passages before answer generation
- returns a natural-language answer plus source evidence
- serves a FastAPI API with a mounted Gradio UI

## Project Structure

```text
.
├── app.py                  # FastAPI app with mounted Gradio interface
├── docker-compose.yaml     # Local Qdrant service
├── qdrant.py               # Shared Qdrant client
├── docs/                   # Sample and indexed source documents
├── rag/
│   ├── config.py           # Environment-driven settings
│   ├── embeddings.py       # OpenAI embedding client
│   ├── repository.py       # Qdrant collection management and search
│   ├── reranker.py         # LLM reranking layer
│   ├── generator.py        # Final grounded answer generation
│   └── service.py          # End-to-end ingest/query orchestration
├── ingestion/
│   ├── loaders/            # PDF and DOCX parsing
│   ├── chunking/           # Text splitting and metadata enrichment
│   └── embedding/          # Batch embedding adapter
└── scripts/
    ├── ingest.py           # CLI ingestion
    └── query.py            # CLI querying
```

## Requirements

- Python 3.11+
- Docker or Docker Desktop
- an OpenAI API key

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

4. Start Qdrant.

```bash
docker compose up -d
```

## Configuration

Main environment variables:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_RERANK_MODEL=gpt-4.1-mini
OPENAI_ANSWER_MODEL=gpt-4.1-mini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=rag_documents
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_LIMIT=12
RERANK_LIMIT=5
```

If you use Qdrant Cloud, set `QDRANT_URL` and `QDRANT_API_KEY` to your hosted instance instead of the local container.

## Ingest Documents

To index one or more files:

```bash
.venv/bin/python scripts/ingest.py ./docs/water-management-handbook-2013.pdf
```

You can also ingest multiple files in one command:

```bash
.venv/bin/python scripts/ingest.py ./docs/file1.pdf ./docs/file2.txt
```

## Query From The CLI

```bash
.venv/bin/python scripts/query.py "What is the formula for unit pumping cost for electric-powered pumping plants?"
```

## Run The App

For local development:

```bash
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

For GitHub Codespaces:

```bash
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open the forwarded URL for port `8000` from the `PORTS` tab.

## API Endpoints

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Ingest a file:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"path":"./docs/water-management-handbook-2013.pdf"}'
```

Query the indexed handbook:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Define pumping head and give the electric pumping cost formula."}'
```

## How It Works

1. A source document is loaded and parsed.
2. The text is split into chunks with metadata.
3. OpenAI embeddings are created for each chunk.
4. Chunks are stored in Qdrant.
5. A user question is embedded and used to retrieve likely matches.
6. The reranker reorders those matches by relevance.
7. The answer generator produces a grounded response from the reranked context.

## Notes

- The Gradio interface is currently phrased for the water management handbook use case.
- Answers are normalized for a plain chat-style UI, so formula explanations read naturally instead of showing raw LaTeX-style formatting.
- If the app is running in Codespaces and the page does not open, check the forwarded port for `8000`.

## License

See [LICENSE](/workspaces/Scalable-RAG-With-KGB/LICENSE).
