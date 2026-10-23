"""Streamlit UI for the AI Document Assistant.

Loads local PDF and Excel files once, then sends questions to the existing
Gemini agent. The API key stays in the environment and is never shown.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import ask_question
from app.models import AgentAnswer
from app.document_loader import default_documents_dir, load_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def should_skip_duplicate_request(clicked: bool, in_flight: bool) -> bool:
    """Return True when a Streamlit rerun would repeat an in-progress Gemini call."""
    return bool(clicked and in_flight)


def main() -> None:
    """Draw the page and handle one question at a time."""
    st.set_page_config(page_title="AI Document Assistant", layout="centered")
    st.title("AI Document Assistant")
    st.caption(
        "Ask questions about the PDF and Excel files in the documents folder. "
        "Answers are grounded in those files and include source references."
    )
    # load all chuncks 
    chunks = _ensure_documents_loaded()
    _show_document_status(chunks)
    _show_question_form(chunks)
    _show_answer()


def _ensure_documents_loaded() -> list:
    """Load documents into session state on first visit, or when asked to reload."""
    # we proceed any question yes or no ? first time open its false 
    if "in_flight" not in st.session_state:
        st.session_state.in_flight = False
        # if no chuncks read and save the chuncks 
    if "chunks" not in st.session_state:
        _reload_documents()
        # return after save any modification require click reload documents 
    return st.session_state.chunks


def _folder_signature(folder: Path) -> str:
    """Fingerprint the folder so Streamlit recaches only when files change."""
    if not folder.exists() or not folder.is_dir():
        return "missing"
    parts: list[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        # read all file and get only last update time in nano sec and the size 
        parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    return "|".join(parts)


@st.cache_data(show_spinner=False)
def _load_documents_cached(folder: str, signature: str) -> list:
    """Cache parsed PDF/Excel chunks across Streamlit reruns."""
    started = time.perf_counter()
    chunks = load_documents(folder)
    logger.info(
        "timing.ui_load_documents cache_build seconds=%.3f signature_chars=%s",
        time.perf_counter() - started,
        len(signature),
    )
    return chunks


def _reload_documents() -> None:
    """Read the documents/ folder and reset any previous answer."""
    documents_dir = default_documents_dir()
    signature = _folder_signature(documents_dir)
    with st.spinner("Loading documents..."):
        # add chuncks
        st.session_state.chunks = _load_documents_cached(str(documents_dir), signature)
        # remove old question
    st.session_state.answer = None
    st.session_state.documents_dir = str(documents_dir)


def _show_document_status(chunks: list) -> None:
    """Show how many files and searchable chunks are ready."""
    filenames = sorted({chunk.filename for chunk in chunks})
    folder = st.session_state.get("documents_dir", str(default_documents_dir()))

    left, right = st.columns(2)
    left.metric("Loaded files", len(filenames))
    right.metric("Searchable chunks", len(chunks))
    # its a section can open and close 
    with st.expander("Document details", expanded=not filenames):
        st.write(f"Folder: `{folder}`")
        if filenames:
            for name in filenames:
                st.write(f"- {name}")
        else:
            st.warning(
                "The documents folder is empty, or no PDF/XLSX files could be read. "
                "Add files to the documents folder and click Reload documents."
            )

    if st.button("Reload documents"):
        _load_documents_cached.clear()
        _reload_documents()
        st.rerun()


def _show_question_form(chunks: list) -> None:
    """Collect a question and call the existing agent when Ask AI is submitted."""
    with st.form("ask_form"):
        question = st.text_area(
            "Your question",
            placeholder="Example: What is the pump pressure rating?",
            height=100,
        )
        ask_clicked = st.form_submit_button("Ask AI", type="primary")

    if should_skip_duplicate_request(ask_clicked, st.session_state.get("in_flight", False)):
        st.info("The previous question is still being processed.")
        return

    if not ask_clicked:
        return

    if not chunks:
        st.error("No documents are loaded, so the question cannot be answered.")
        return

    cleaned = question.strip()
    if not cleaned:
        st.warning("Please enter a question before clicking Ask AI.")
        return

    st.session_state.in_flight = True
    started = time.perf_counter()
    try:
        with st.spinner("Asking the AI..."):
            st.session_state.answer = ask_question(cleaned, chunks)
    finally:
        st.session_state.in_flight = False
        logger.info(
            "timing.ui_ask seconds=%.2f",
            time.perf_counter() - started,
        )


def _show_answer() -> None:
    """Render the latest agent answer, sources, and measured response time."""
    answer = st.session_state.get("answer")
    if answer is None:
        return
    if not isinstance(answer, AgentAnswer):
        st.error("The assistant returned an unexpected result.")
        return

    st.subheader("Answer")
    if answer.is_fallback:
        st.warning("Local Search Results — AI temporarily unavailable.")
        fallback_text = answer.text.split("\n", 1)
        st.write(fallback_text[1] if len(fallback_text) > 1 else answer.text)
    elif answer.error:
        st.error(answer.text)
    else:
        st.write(answer.text)

    if answer.model:
        st.caption(f"Gemini model: {answer.model}")

    if answer.elapsed_seconds is not None:
        st.caption(f"Response time: {answer.elapsed_seconds:.2f} seconds")

    labels = answer.source_labels()
    st.subheader("Sources")
    if labels:
        for label in labels:
            st.write(f"- {label}")
    else:
        st.info("No document sources were returned for this answer.")


if __name__ == "__main__":
    main()
