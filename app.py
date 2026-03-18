from __future__ import annotations

import gradio as gr
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.service import RAGService

app = FastAPI(title="Reranker RAG with Qdrant")
service = RAGService()


class IngestRequest(BaseModel):
    path: str


class QueryRequest(BaseModel):
    question: str


def handle_ingest(path: str) -> str:
    try:
        chunk_count = service.ingest_file(path)
    except (FileNotFoundError, ValueError) as exc:
        return f"Ingest failed: {exc}"
    except Exception as exc:
        return f"Ingest failed: {exc}"

    return f"Ingested {chunk_count} chunks from {path}"


def handle_query(question: str) -> tuple[str, list[list[str]]]:
    if not question.strip():
        return "Ask a question about the indexed water management handbook.", []

    try:
        result = service.answer(question)
    except Exception as exc:
        return f"Query failed: {exc}", []

    matches = []
    for item in result["matches"]:
        source = item["metadata"].get("filename", "unknown")
        score = item.get("rerank_score")
        if score is None:
            score = item.get("vector_score")
        snippet = item["text"][:220].replace("\n", " ").strip()
        matches.append([source, f"{score:.4f}", snippet])

    return result["answer"], matches


def build_interface() -> gr.Blocks:
    ocean_css = """
    :root {
      --ocean-deep: #0f6d8c;
      --ocean-aqua: #69d8e6;
      --ocean-surf: #9beff5;
      --ocean-mist: #eefcff;
      --ocean-foam: #ffffff;
      --ocean-text: #20506a;
      --ocean-muted: #5f7e92;
      --ocean-line: rgba(68, 153, 181, 0.18);
      --ocean-shadow: 0 20px 55px rgba(61, 149, 179, 0.16);
    }

    body, .gradio-container {
      background:
        radial-gradient(circle at top left, rgba(155, 239, 245, 0.9), transparent 32%),
        radial-gradient(circle at 85% 12%, rgba(110, 214, 229, 0.45), transparent 20%),
        linear-gradient(180deg, #f6feff 0%, #ecfbff 48%, #f9fdff 100%);
      color: var(--ocean-text);
      font-family: "Manrope", "Segoe UI", sans-serif;
    }

    .app-shell {
      max-width: 1140px;
      margin: 0 auto;
      padding: 30px 20px 40px;
    }

    .topbar {
      position: relative;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.7);
      border-radius: 32px;
      padding: 28px 30px;
      background:
        linear-gradient(135deg, rgba(255, 255, 255, 0.92), rgba(219, 248, 252, 0.88)),
        linear-gradient(90deg, rgba(120, 216, 228, 0.22), rgba(255, 255, 255, 0.3));
      box-shadow: var(--ocean-shadow);
    }

    .topbar:before {
      content: "";
      position: absolute;
      inset: auto -4rem -4rem auto;
      width: 16rem;
      height: 16rem;
      border-radius: 50%;
      background: radial-gradient(circle, rgba(105, 216, 230, 0.34), rgba(105, 216, 230, 0));
    }

    .eyebrow {
      display: inline-block;
      margin-bottom: 12px;
      padding: 7px 14px;
      border-radius: 999px;
      background: rgba(105, 216, 230, 0.18);
      color: var(--ocean-deep);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }

    .topbar h1 {
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.5rem);
      line-height: 0.98;
      letter-spacing: -0.05em;
      color: #18465d;
      font-weight: 800;
    }

    .topbar p {
      max-width: 760px;
      margin: 14px 0 0;
      font-size: 1rem;
      line-height: 1.75;
      color: var(--ocean-muted);
    }

    .surface-row {
      align-items: start;
      gap: 18px;
      margin-top: 20px;
    }

    .surface {
      border: 1px solid rgba(255, 255, 255, 0.72) !important;
      border-radius: 28px !important;
      background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(238, 252, 255, 0.96)) !important;
      box-shadow: var(--ocean-shadow);
      padding: 8px !important;
    }

    .surface-accent {
      background:
        linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(226, 249, 252, 0.98)) !important;
    }

    .section-copy {
      padding: 8px 12px 4px;
    }

    .section-copy h2 {
      margin: 0;
      font-size: 1.35rem;
      letter-spacing: -0.03em;
      color: #1d5067;
      font-weight: 750;
    }

    .section-copy p {
      margin: 8px 0 0;
      font-size: 0.96rem;
      line-height: 1.65;
      color: var(--ocean-muted);
    }

    .gr-form,
    .gr-box,
    .gr-group {
      border: none !important;
      background: transparent !important;
      box-shadow: none !important;
    }

    .gr-button {
      border-radius: 18px !important;
      border: 1px solid rgba(48, 142, 171, 0.08) !important;
      background: linear-gradient(135deg, #56cfde, #8be8ee) !important;
      color: #11445b !important;
      font-weight: 800 !important;
      letter-spacing: 0.01em;
      min-height: 50px;
      box-shadow: 0 12px 24px rgba(105, 216, 230, 0.22);
    }

    .gr-button.secondary {
      background: linear-gradient(135deg, #2bb7cb, #6fdce7) !important;
      color: white !important;
    }

    label span {
      color: #537488 !important;
      font-weight: 600 !important;
    }

    textarea, input {
      background: rgba(255, 255, 255, 0.92) !important;
      color: var(--ocean-text) !important;
      border: 1px solid var(--ocean-line) !important;
      border-radius: 18px !important;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }

    textarea::placeholder, input::placeholder {
      color: #8ca8b6 !important;
    }

    .gr-textbox, .gr-dataframe {
      background: transparent !important;
    }

    .result-panel textarea {
      background: linear-gradient(180deg, rgba(245, 253, 255, 0.98), rgba(233, 248, 252, 0.98)) !important;
      min-height: 180px;
    }

    .dataframe-wrap {
      border: 1px solid var(--ocean-line) !important;
      border-radius: 22px !important;
      overflow: hidden;
      background: rgba(255, 255, 255, 0.85) !important;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }

    table {
      color: var(--ocean-text) !important;
    }

    th {
      background: rgba(124, 224, 234, 0.18) !important;
      color: #25566e !important;
    }

    td {
      background: rgba(255, 255, 255, 0.86) !important;
    }

    .footnote {
      margin-top: 12px;
      padding-left: 6px;
      color: #7190a2;
      font-size: 0.9rem;
    }

    @media (max-width: 860px) {
      .app-shell {
        padding: 18px 14px 28px;
      }

      .topbar {
        padding: 22px;
      }
    }
    """

    theme = gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="cyan",
        neutral_hue="sky",
        radius_size="lg",
        spacing_size="lg",
        text_size="md",
        font=[gr.themes.GoogleFont("Manrope"), "ui-sans-serif", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
    )

    with gr.Blocks(
        theme=theme,
        css=ocean_css,
        title="Water Management Handbook Assistant",
    ) as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                """
                <section class="topbar">
                  <div class="eyebrow">Water Knowledge Assistant</div>
                  <h1>Ask The Water Management Handbook<br/>In Plain Language</h1>
                  <p>
                    Search the indexed water management manual for equations,
                    pumping guidance, definitions, and supporting excerpts.
                    The interface is tuned for practical handbook lookup:
                    ask a question, read the answer, and inspect the source text beside it.
                  </p>
                </section>
                """
            )

            with gr.Row(equal_height=False, elem_classes=["surface-row"]):
                with gr.Column(elem_classes=["surface"], scale=8):
                    gr.HTML(
                        """
                        <div class="section-copy">
                          <h2>Water Handbook Search</h2>
                          <p>
                            Try questions about pumping cost formulas, irrigation terms,
                            water delivery guidance, or any section from the indexed manual.
                          </p>
                        </div>
                        """
                    )
                    question = gr.Textbox(
                        label="Question",
                        placeholder="Example: What is the formula for unit pumping cost for electric-powered pumping plants?",
                        lines=2,
                    )
                    ask_button = gr.Button("Search Handbook", elem_classes=["secondary"])
                    with gr.Row(equal_height=False):
                        answer = gr.Textbox(
                            label="Answer",
                            lines=8,
                            interactive=False,
                            elem_classes=["result-panel"],
                            scale=7,
                        )
                        sources = gr.Dataframe(
                            headers=["Source", "Score", "Snippet"],
                            datatype=["str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            row_count=5,
                            col_count=(3, "fixed"),
                            label="Top Sources",
                            scale=8,
                        )
                    gr.HTML(
                        """
                        <div class="footnote">
                          Indexed sources appear on the right so you can verify each answer against the handbook text.
                        </div>
                        """
                    )

            ask_button.click(
                handle_query,
                inputs=question,
                outputs=[answer, sources],
            )

    return demo


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_documents": service.repository.count()}


@app.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    try:
        chunk_count = service.ingest_file(request.path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"path": request.path, "chunks_ingested": chunk_count}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    return service.answer(request.question)


gradio_app = build_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/")
