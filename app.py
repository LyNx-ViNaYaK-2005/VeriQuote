"""Folio's Streamlit entry point. All uploaded content is session-owned."""
import logging

import streamlit as st

from src.config import Config
from src.errors import AppError
from src.export.chat_export import export_chat
from src.rag.pipeline import Pipeline
from src.ui.browser_events import protect_session
from src.ui.components import brand, empty_state, how_it_works, render_answer
from src.ui.styles import apply_styles


st.set_page_config(page_title="Folio · Document workspace", page_icon="▤", layout="wide")
logging.basicConfig(level=logging.WARNING)
apply_styles()

try:
    config = Config.from_env()
except AppError as exc:
    st.error(str(exc))
    st.stop()

if "pipeline" not in st.session_state:
    st.session_state.pipeline = Pipeline(config)
    st.session_state.upload_version = 0
    st.session_state.exported_turns = 0
    st.session_state.processing_results = []

pipeline = st.session_state.pipeline


def mark_exported() -> None:
    st.session_state.exported_turns = len(st.session_state.pipeline.history)


def reset_session() -> None:
    st.session_state.pipeline = Pipeline(config)
    st.session_state.upload_version += 1
    st.session_state.exported_turns = 0
    st.session_state.processing_results = []
    st.session_state.pop("failed_question", None)


def downloads(prefix: str) -> None:
    names = [d.source for d in pipeline.documents.values()]
    for col, fmt, label in zip(st.columns(2), ("md", "txt"), ("Markdown", "Plain text")):
        with col:
            st.download_button(label, export_chat(pipeline.history, names, format=fmt),
                               file_name=f"folio-chat.{fmt}",
                               mime="text/markdown" if fmt == "md" else "text/plain",
                               key=f"{prefix}_{fmt}", on_click=mark_exported, width="stretch")


@st.dialog("Start a new session?")
def confirm_reset() -> None:
    st.write("This will clear your documents and conversation. Export your chat first if you want to keep it.")
    downloads("reset")
    keep, clear = st.columns(2)
    if keep.button("Keep this session", width="stretch"):
        st.rerun()
    if clear.button("Clear and start new", type="primary", width="stretch", on_click=reset_session):
        st.rerun()


with st.sidebar:
    brand()
    st.html('<div class="folio-eyebrow">Session workspace</div>')
    st.subheader("Your documents")
    if not pipeline.documents:
        st.caption("Your reading starts here. Added PDFs will appear in this space.")
    for doc in list(pipeline.documents.values()):
        with st.container(border=True):
            st.text(doc.source)
            st.caption(f"{doc.page_count} pages · {len(doc.chunks)} passages · {doc.size / 1024:.0f} KB")
            left, right = st.columns([3, 1])
            left.caption("● Indexed")
            if right.button("", icon=":material/close:", key=f"remove_{doc.id}",
                            help=f"Remove {doc.source}; previous answers remain available"):
                try:
                    pipeline.remove_document(doc.id)
                    st.session_state.processing_results = []
                    st.rerun()
                except AppError as exc:
                    st.error(str(exc))
            if doc.text_pages < doc.page_count:
                st.caption(f"{doc.page_count - doc.text_pages} pages had no text and were skipped.")
    if pipeline.documents:
        if st.button("Clear all documents", icon=":material/delete_sweep:", width="stretch"):
            pipeline.clear_documents()
            st.session_state.processing_results = []
            st.session_state.upload_version += 1
            st.rerun()
        st.caption("Removing PDFs keeps earlier answers and their evidence until you start a new session.")
    st.divider()
    with st.expander("Answer preferences", icon=":material/tune:"):
        style = st.selectbox("Response style", ["Concise", "Detailed", "Explain simply"])
        top_k = st.slider("Retrieved passages", 3, 8, 5,
                          help="Maximum passages to consider. Repetitive or low-similarity passages are excluded.")
        follow_up = st.checkbox("Use previous questions for follow-ups", value=True,
                               help="Helps resolve phrases such as 'explain that'. Earlier answers are never evidence.")
    if st.button("New session", icon=":material/add:", width="stretch"):
        if pipeline.history:
            confirm_reset()
        else:
            reset_session()
            st.rerun()
    st.caption("Temporary by design. No account. No permanent chat history.")
    with st.expander("Privacy & limits", icon=":material/info:"):
        st.write("PDFs and search data stay in server memory for this session. Embeddings run locally on the server's CPU. Selected passages and questions go to Groq; its data policies apply.")
        st.caption("Text PDFs only · 20 MB per file · 50 MB total · 8 documents · 500 pages per file · 3,000 passages · 40 questions.")
        st.caption("Reloads, disconnections, or server restarts may end the session. Browser exit warnings are best effort, especially on mobile.")

count = len(pipeline.documents)
status_text = f"{count} {'document' if count == 1 else 'documents'} indexed" if count else "Session ready"
st.html(f'<div class="folio-topline"><span>FOLIO &nbsp; / &nbsp; DOCUMENT WORKSPACE</span>'
        f'<span class="folio-status">● &nbsp; {status_text}</span></div>')

if not config.groq_api_key:
    with st.expander("Connect Groq to answer questions", expanded=True, icon=":material/settings:"):
        st.info("Server setup needed: GROQ_API_KEY")
        st.write("Add the key to `.env` (see `.env.example`) or your hosting environment, then restart the app. Documents can be indexed without an API key. No visitor account is required.")

if not pipeline.documents and not pipeline.history:
    empty_state()
else:
    title, actions = st.columns([3, 1])
    with title:
        st.html('<div class="folio-eyebrow">From reading to understanding</div>')
        st.subheader("A conversation with your documents")
    with actions:
        st.metric("Source pages", sum(d.text_pages for d in pipeline.documents.values()))

upload_area = st.expander("Add documents", expanded=not bool(pipeline.documents), icon=":material/upload_file:")
with upload_area:
    uploads = st.file_uploader("Bring your PDFs into focus", type=["pdf"], accept_multiple_files=True,
                               key=f"uploads_{st.session_state.upload_version}",
                               help="Text-based PDFs, up to 20 MB each. Indexing runs locally on the server; selected passages go to Groq when you ask a question.")
    st.caption("Indexing runs locally on the server's CPU. First use may take longer while the model downloads. Selected passages are sent to Groq when you ask a question.")
    if st.button("Index documents", type="primary", icon=":material/arrow_forward:",
                 disabled=not uploads):
        results = []
        for upload in uploads:
            with st.status(f"Processing {upload.name}", expanded=True) as status:
                st.write(f"Upload received · {upload.size / 1024:.0f} KB")
                try:
                    doc = pipeline.add_pdf(upload.getvalue(), upload.name, st.write)
                    detail = f"{doc.source} · {doc.text_pages}/{doc.page_count} pages indexed · {len(doc.chunks)} passages"
                    if doc.text_pages < doc.page_count:
                        detail += " · pages without text were skipped"
                    results.append((True, detail))
                    status.update(label="Ready", state="complete", expanded=False)
                except AppError as exc:
                    results.append((False, f"{upload.name}: {exc}"))
                    status.update(label="Could not index this file", state="error", expanded=True)
        st.session_state.processing_results = results
        st.session_state.upload_version += 1
        st.rerun()

for ok, message in st.session_state.processing_results:
    (st.success if ok else st.error)(message)
if st.session_state.processing_results and st.button("Dismiss processing results", type="tertiary"):
    st.session_state.processing_results = []
    st.rerun()

if pipeline.history:
    export_col, notice_col = st.columns([1, 2.4])
    with export_col:
        with st.popover("Export Chat", icon=":material/download:", width="stretch"):
            downloads("main")
    with notice_col:
        st.caption("This conversation isn't saved. Export it before leaving if you want to keep a copy.")
    st.divider()

protect_session(bool(pipeline.history) and st.session_state.exported_turns < len(pipeline.history))

for turn in pipeline.history:
    with st.chat_message("user", avatar=":material/person:"):
        st.text(turn.question)
    with st.chat_message("assistant", avatar=":material/menu_book:"):
        render_answer(turn.answer, set(pipeline.documents))

question = None
if pipeline.documents and not pipeline.history:
    st.subheader("Where would you like to begin?")
    st.caption("A few starting points. Answers depend on what your documents contain.")
    suggestions = ["Summarize the key ideas in these documents.", "What are the main concepts discussed?",
                   "Which limitations are described?"]
    for col, suggestion in zip(st.columns(3), suggestions):
        with col:
            if st.button(suggestion, width="stretch", disabled=not config.groq_api_key):
                question = suggestion

if not pipeline.documents and not pipeline.history:
    how_it_works()

if st.session_state.get("failed_question"):
    st.caption("Your last question wasn't completed. You can retry it without re-indexing.")
    st.text(st.session_state.failed_question)
    if st.button("Retry question", disabled=not pipeline.documents):
        question = st.session_state.failed_question

prompt = None
if pipeline.documents or pipeline.history:
    prompt = st.chat_input("Ask about your documents…" if pipeline.documents else "Index a PDF to start a conversation",
                           disabled=not pipeline.documents or not config.groq_api_key, max_chars=1500)
question = prompt or question
if question:
    try:
        with st.spinner("Finding passages and checking supporting evidence…", show_time=True):
            pipeline.ask(question, top_k, style, follow_up)
        st.session_state.pop("failed_question", None)
        st.rerun()
    except AppError as exc:
        st.session_state.failed_question = question
        st.error(str(exc))
        st.caption("Your documents and existing conversation are still available. Submit again to retry.")
