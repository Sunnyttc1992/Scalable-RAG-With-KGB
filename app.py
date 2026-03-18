from __future__ import annotations

import os

import gradio as gr

from rag.service import RAGService

service = RAGService()


# Convert retrieved match objects into table rows for the UI.
def _matches_to_rows(result: dict) -> list[list[str]]:
    matches = []
    for item in result["matches"]:
        source = item["metadata"].get("filename", "unknown")
        score = item.get("rerank_score")
        if score is None:
            score = item.get("vector_score")
        snippet = item["text"][:220].replace("\n", " ").strip()
        matches.append([source, f"{score:.4f}", snippet])
    return matches


# Run a one-shot query and return the answer with its top source snippets.
def handle_query(question: str) -> tuple[str, list[list[str]]]:
    if not question.strip():
        return "Ask a question about the indexed water management handbook.", []

    try:
        result = service.answer(question)
    except Exception as exc:
        return f"Query failed: {exc}", []

    return result["answer"], _matches_to_rows(result)


# Process one chat turn and preserve enough history for follow-up questions.
def handle_chat(message, history):
    if not message.strip():
        empty_history = history or []
        return empty_history, empty_history, [], ""

    chat_history = list(history or [])
    prior_history = []
    for user_message, assistant_message in chat_history:
        if user_message:
            prior_history.append({"role": "user", "content": user_message})
        if assistant_message:
            prior_history.append({"role": "assistant", "content": assistant_message})

    try:
        result = service.answer(message, history=prior_history)
        answer = result["answer"]
        matches = _matches_to_rows(result)
    except Exception as exc:
        answer = f"Query failed: {exc}"
        matches = []

    chat_history.append([message, answer])
    return chat_history, chat_history, matches, ""


# Reset the chat messages, stored history, source table, and input box.
def clear_chat():
    return [], [], [], ""


# Build the full Gradio interface for the handbook assistant.
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
                          <h2>Water Handbook Chat</h2>
                          <p>
                            Ask follow-up questions naturally. The assistant keeps recent
                            conversation in memory so you can say things like "explain that"
                            or "compare it with diesel pumping cost."
                          </p>
                        </div>
                        """
                    )
                    chat_history = gr.State([])
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=520,
                        elem_classes=["result-panel"],
                    )
                    question = gr.Textbox(
                        label="Message",
                        placeholder="Example: What is the formula for unit pumping cost for electric-powered pumping plants?",
                        lines=2,
                    )
                    with gr.Row():
                        ask_button = gr.Button("Send Message", elem_classes=["secondary"], scale=5)
                        clear_button = gr.Button("Clear Chat", scale=2)
                    with gr.Row(equal_height=False):
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
                          Each reply is grounded in retrieved handbook text, and the latest turn's source evidence appears on the right.
                        </div>
                        """
                    )

            ask_button.click(
                handle_chat,
                inputs=[question, chat_history],
                outputs=[chatbot, chat_history, sources, question],
                api_name=False,
            )

            question.submit(
                handle_chat,
                inputs=[question, chat_history],
                outputs=[chatbot, chat_history, sources, question],
                api_name=False,
            )

            clear_button.click(
                clear_chat,
                outputs=[chatbot, chat_history, sources, question],
                api_name=False,
            )

    return demo


demo = build_interface()
app = demo


if __name__ == "__main__":
    # Start the Gradio app when running this file directly.
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
