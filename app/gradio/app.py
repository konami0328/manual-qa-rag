"""
Gradio frontend for the Tesla Model Y RAG QA system.

Layout:
    Left panel  — question input + streaming answer
    Right panel — retrieved context chunks

Communicates with FastAPI via SSE stream on POST /ask.
Parses [CONTEXT], [TOKEN], [DONE], [ERROR] prefixes from the stream.

Usage:
    python app/gradio/app.py
"""

import re
import httpx
import gradio as gr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL   = "http://localhost:8001/ask"
NO_ANSWER = "This information is not covered in the provided context."

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------

CREAM  = "#FFFDEB"
STEEL  = "#7DAACB"
SAND   = "#E8DBB3"
RED    = "#CE2626"
DARK   = "#2C2C2C"
MUTED  = "#6B6B6B"

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;600&family=DM+Sans:wght@300;400;500&display=swap');

* {{
    box-sizing: border-box;
}}

body, .gradio-container {{
    background-color: {CREAM} !important;
    font-family: 'DM Sans', sans-serif !important;
    color: {DARK} !important;
}}

/* Header */
#header {{
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
    border-bottom: 1px solid {SAND};
    margin-bottom: 2rem;
}}

#header h1 {{
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 2.4rem !important;
    font-weight: 600 !important;
    color: {DARK} !important;
    letter-spacing: 0.02em;
    margin: 0 0 0.3rem 0;
}}

#header p {{
    font-size: 0.85rem;
    color: {MUTED};
    margin: 0;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

/* Input */
#question-input textarea {{
    background: white !important;
    border: 1.5px solid {SAND} !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: {DARK} !important;
    padding: 0.8rem 1rem !important;
    resize: none !important;
    transition: border-color 0.2s;
}}

#question-input textarea:focus {{
    border-color: {STEEL} !important;
    outline: none !important;
    box-shadow: 0 0 0 3px {STEEL}22 !important;
}}

#question-input label {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    margin-bottom: 0.4rem !important;
}}

/* Submit button */
#submit-btn {{
    background: {STEEL} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.8rem !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100% !important;
}}

#submit-btn:hover {{
    background: #6799ba !important;
    transform: translateY(-1px) !important;
}}

#submit-btn:active {{
    transform: translateY(0) !important;
}}

/* Answer box */
#answer-box textarea {{
    background: white !important;
    border: 1.5px solid {SAND} !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    color: {DARK} !important;
    padding: 1rem !important;
    min-height: 220px !important;
}}

#answer-box label {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    margin-bottom: 0.4rem !important;
}}

/* Context box */
#context-box {{
    background: {SAND}55 !important;
    border: 1.5px solid {SAND} !important;
    border-radius: 8px !important;
    padding: 1rem 1.2rem !important;
    min-height: 360px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    color: {DARK} !important;
    overflow-y: auto !important;
}}

#context-box label {{
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: {MUTED} !important;
    margin-bottom: 0.4rem !important;
}}

/* Tesla accent line */
.tesla-accent {{
    width: 40px;
    height: 3px;
    background: {RED};
    margin: 0.8rem auto 0 auto;
    border-radius: 2px;
}}

/* Divider between chunks */
.chunk-divider {{
    border: none;
    border-top: 1px solid {SAND};
    margin: 0.8rem 0;
}}
"""

# ---------------------------------------------------------------------------
# Stream handler
# ---------------------------------------------------------------------------

def ask(question: str):
    """
    Generator that yields (answer, context_html) tuples as the SSE stream
    arrives from FastAPI.

    Yields intermediate states so Gradio updates both panels in real time.
    """
    if not question.strip():
        yield "", "<p style='color:#aaa;font-style:italic;'>Ask a question to see retrieved context.</p>"
        return

    answer_text   = ""
    context_parts = []
    context_html  = "<p style='color:#aaa;font-style:italic;'>Retrieving context...</p>"

    yield answer_text, context_html

    try:
        with httpx.Client(timeout=60.0) as client:
            with client.stream("POST", API_URL, json={"question": question}) as resp:
                buffer = ""
                for raw in resp.iter_text():
                    buffer += raw
                    while "\n\n" in buffer:
                        msg, buffer = buffer.split("\n\n", 1)
                        if not msg.startswith("data: "):
                            continue
                        data = msg[len("data: "):]

                        if data.startswith("[CONTEXT]"):
                            # parse: "p.<page>  score:<score>\n<text>"
                            body       = data[len("[CONTEXT]"):]
                            lines      = body.split("\n", 1)
                            meta_line  = lines[0].strip()
                            chunk_text = lines[1].strip() if len(lines) > 1 else ""

                            page_match  = re.search(r"p\.(\S+)", meta_line)
                            score_match = re.search(r"score:(-?[\d.]+)", meta_line)
                            page  = page_match.group(1)  if page_match  else "?"
                            score = score_match.group(1) if score_match else "?"

                            part = (
                                f"<div style='margin-bottom:0.6rem'>"
                                f"<span style='color:{STEEL};font-weight:500;font-size:0.8rem;'>"
                                f"p.{page}</span>"
                                f"<span style='color:{MUTED};font-size:0.78rem;margin-left:0.8rem;'>"
                                f"relevance {score}</span>"
                                f"<div style='margin-top:0.3rem;color:{DARK};'>{chunk_text}</div>"
                                f"</div>"
                                f"<hr style='border:none;border-top:1px solid {SAND};margin:0.6rem 0;'>"
                            )
                            context_parts.append(part)
                            context_html = "".join(context_parts)
                            yield answer_text, context_html

                        elif data.startswith("[TOKEN]"):
                            token        = data[len("[TOKEN]"):]
                            answer_text += token
                            yield answer_text, context_html

                        elif data.startswith("[DONE]"):
                            if not answer_text:
                                answer_text = NO_ANSWER
                                context_html = (
                                    f"<p style='color:{MUTED};font-style:italic;'>"
                                    f"No relevant context found in the manual.</p>"
                                )
                            yield answer_text, context_html
                            return

                        elif data.startswith("[ERROR]"):
                            error_msg    = data[len("[ERROR]"):]
                            answer_text  = f"⚠ Error: {error_msg}"
                            yield answer_text, context_html
                            return

    except Exception as e:
        yield f"⚠ Connection error: {str(e)}", context_html


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(css=CSS, title="Tesla Model Y QA") as demo:

    gr.HTML("""
        <div id="header">
            <h1>Tesla Model Y</h1>
            <p>Owner's Manual · Intelligent Q&amp;A</p>
            <div class="tesla-accent"></div>
        </div>
    """)

    with gr.Row(equal_height=False):

        # --- Left panel ---
        with gr.Column(scale=5):
            question_input = gr.Textbox(
                label       = "Your Question",
                placeholder = "e.g. How do I enable Autopilot?",
                lines       = 3,
                elem_id     = "question-input",
            )
            submit_btn = gr.Button(
                "Ask",
                elem_id = "submit-btn",
                variant = "primary",
            )
            answer_box = gr.Textbox(
                label       = "Answer",
                interactive = False,
                lines       = 10,
                elem_id     = "answer-box",
            )

        # --- Right panel ---
        with gr.Column(scale=5):
            context_box = gr.HTML(
                value   = "<p style='color:#aaa;font-style:italic;padding:1rem;'>Retrieved context will appear here.</p>",
                label   = "Retrieved Context",
                elem_id = "context-box",
            )

    # --- Wire up ---
    submit_btn.click(
        fn      = ask,
        inputs  = [question_input],
        outputs = [answer_box, context_box],
    )
    question_input.submit(
        fn      = ask,
        inputs  = [question_input],
        outputs = [answer_box, context_box],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
    )