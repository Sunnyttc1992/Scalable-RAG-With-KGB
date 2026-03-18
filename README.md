# Water Management Handbook Assistant

This project is a retrieval-augmented chatbot for water management reference material. It indexes handbook content into Qdrant, retrieves relevant passages with OpenAI embeddings, reranks those passages with an LLM, and produces grounded conversational answers with visible source snippets in a Gradio interface.

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
- keeps short-term conversation memory for follow-up questions
- returns a natural-language answer plus source evidence
- runs as a standard Gradio app for Hugging Face Spaces

## Project Structure

```text
.
├── app.py                  # Gradio app entrypoint for local use and Spaces
├── docker-compose.yaml     # Optional local Qdrant service
├── qdrant.py               # Shared Qdrant client helper
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
- an OpenAI API key
- a Qdrant collection, preferably Qdrant Cloud for deployment

## Hugging Face Spaces Deployment

This repo is prepared for a Gradio Space. The simplest production setup is:

- Hugging Face Spaces for the web app
- Qdrant Cloud for vector storage
- Hugging Face Space secrets for API keys and URLs

### 1. Create the Space

Create a new Hugging Face Space and choose:

- SDK: `Gradio`
- Python version: `3.11` or newer

### 2. Upload the Repository

Upload or push this repository, but do not include your real `.env` file.

### 3. Add Space Secrets

In the Space settings, add these secrets:

```text
OPENAI_API_KEY
QDRANT_URL
QDRANT_API_KEY
QDRANT_COLLECTION_NAME
```

Recommended values:

- `QDRANT_URL`: your Qdrant Cloud endpoint
- `QDRANT_COLLECTION_NAME`: the collection you already ingested, for example `Agentic-Rag`

### 4. Build and Run

Hugging Face Spaces will install `requirements.txt` and run [app.py](/workspaces/Scalable-RAG-With-KGB/app.py) as a Gradio app automatically.

## Local Setup

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

4. Set `QDRANT_URL` and `QDRANT_API_KEY`.

For deployment-like local testing, prefer Qdrant Cloud.

Optional local Qdrant:

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
QDRANT_URL=https://your-cluster-id.region.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION_NAME=water-handbook
CHUNK_SIZE=800
CHUNK_OVERLAP=120
RETRIEVAL_LIMIT=12
RERANK_LIMIT=5
```

If you use local Qdrant instead, set `QDRANT_URL=http://localhost:6333`.

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

For local development or a quick pre-deploy check:

```bash
.venv/bin/python app.py
```

Then open:

```text
http://127.0.0.1:7860
```

For Codespaces or another forwarded environment:

```bash
PORT=7860 .venv/bin/python app.py
```

Then open the forwarded URL for port `7860` from the `PORTS` tab.

## How It Works

1. A source document is loaded and parsed.
2. The text is split into chunks with metadata.
3. OpenAI embeddings are created for each chunk.
4. Chunks are stored in Qdrant.
5. A user question is embedded and used to retrieve likely matches.
6. The reranker reorders those matches by relevance.
7. The answer generator produces a grounded response from the reranked context.

## Notes

- The Gradio interface is phrased for the water management handbook use case.
- Answers are normalized for a plain chat-style UI, so formula explanations read naturally instead of showing raw LaTeX-style formatting.
- Follow-up questions use short conversation memory.
- For Spaces, use Qdrant Cloud rather than local Docker storage.

## Secrets Safety Checklist

Before pushing to GitHub or uploading to a Space:

- keep your real `.env` file out of git
- use Space Secrets instead of hardcoding keys
- make sure [`.env.example`](/workspaces/Scalable-RAG-With-KGB/.env.example) contains placeholders only
- rotate any API keys that were ever pasted into a tracked file or screenshot
- confirm your Qdrant Cloud URL is safe to publish, but keep the API key secret

## License

See [LICENSE](/workspaces/Scalable-RAG-With-KGB/LICENSE).
