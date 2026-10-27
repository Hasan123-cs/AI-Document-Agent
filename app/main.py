"""Streamlit dashboard for asking questions about local PDF and Excel files."""

from __future__ import annotations

import hashlib
import logging
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent import _api_key_from_env, ask_question
from app.document_loader import (
    MAX_UPLOAD_FILE_BYTES,
    MAX_UPLOAD_TOTAL_BYTES,
    load_documents,
    load_uploaded_document,
    default_documents_dir,
    safe_uploaded_filename,
)
from app.models import AgentAnswer
from app.tools import analyze_excel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXAMPLE_QUESTIONS = (
    "Summarize the uploaded PDF.",
    "What is the average cost?",
    "Show items with a cost greater than 1000.",
    "What is the total quantity?",
    "Which page mentions payment conditions?",
)
MAX_UPLOAD_COUNT = 10


def should_skip_duplicate_request(clicked: bool, in_flight: bool) -> bool:
    """Return True when a Streamlit rerun would repeat an in-progress Gemini call."""
    return bool(clicked and in_flight)


def main() -> None:
    """Draw the page and handle one question at a time."""
    st.set_page_config(
        page_title="AI Document Assistant",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_styles()
    _ensure_documents_loaded()
    _show_header()
    _show_upload_section()
    chunks = _current_document_chunks()
    _show_sidebar(chunks)
    _show_summary_cards(chunks)
    _show_question_form(chunks)
    _show_conversation(chunks)


def _render_styles() -> None:
    """Apply a lightweight responsive theme using Streamlit's built-in styling."""
    st.markdown(
        """
        <style>
        :root {
            --ink: #172554;
            --muted: #64748b;
            --brand: #4f46e5;
            --brand-soft: #eef2ff;
            --line: #e2e8f0;
            --surface: #ffffff;
        }
        .stApp {
            background: linear-gradient(180deg, #f8faff 0%, #f6f8fc 45%, #f8fafc 100%);
            color: var(--ink);
        }
        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.2rem;
        }
        .block-container {
            max-width: 1280px;
            padding: 2rem clamp(1rem, 3vw, 2.75rem) 3rem;
        }
        h1, h2, h3, p, label, button, input, textarea {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }
        .dashboard-header {
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.2rem;
            border: 1px solid #dfe6f5;
            border-radius: 22px;
            background: radial-gradient(circle at 90% 10%, #e0e7ff 0, transparent 34%),
                linear-gradient(125deg, #ffffff 15%, #f5f7ff 100%);
            box-shadow: 0 12px 32px rgba(30, 41, 90, 0.06);
        }
        .dashboard-eyebrow {
            color: var(--brand);
            font-size: .75rem;
            font-weight: 750;
            letter-spacing: .12em;
            text-transform: uppercase;
        }
        .dashboard-header h1 {
            margin: .35rem 0 .45rem;
            color: #172554;
            font-size: clamp(1.7rem, 3vw, 2.35rem);
            letter-spacing: -.04em;
        }
        .dashboard-header p {
            max-width: 730px;
            margin: 0;
            color: #53627d;
            line-height: 1.65;
        }
        [data-testid="stMetric"] {
            min-height: 118px;
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            border-radius: 16px;
            background: var(--surface);
            box-shadow: 0 5px 16px rgba(30, 41, 90, 0.04);
        }
        [data-testid="stMetricLabel"] {
            color: #64748b;
            font-weight: 600;
        }
        [data-testid="stMetricValue"] {
            color: #1e293b;
        }
        [data-testid="stChatMessage"] {
            border: 1px solid #e6eaf2;
            border-radius: 16px;
            background: rgba(255, 255, 255, .9);
            box-shadow: 0 4px 14px rgba(30, 41, 90, .035);
        }
        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            border-color: #dbeafe;
            background: #f4f7ff;
        }
        div[data-testid="stForm"] {
            padding: 1rem;
            border: 1px solid #dfe6f5;
            border-radius: 18px;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(30, 41, 90, .05);
        }
        .sidebar-brand {
            display: flex;
            align-items: center;
            gap: .7rem;
            margin-bottom: .35rem;
            color: #1e293b;
            font-size: 1.02rem;
            font-weight: 750;
        }
        .sidebar-brand-mark {
            display: grid;
            width: 2.35rem;
            height: 2.35rem;
            place-items: center;
            border-radius: 12px;
            background: linear-gradient(135deg, #4f46e5, #6366f1);
            color: white;
        }
        .sidebar-subtitle, .muted-caption {
            color: var(--muted);
            font-size: .83rem;
        }
        .example-caption {
            margin: .15rem 0 .6rem;
            color: #64748b;
            font-size: .9rem;
        }
        .stButton > button {
            border-radius: 10px;
            border-color: #d9e0ed;
            font-weight: 600;
        }
        @media (max-width: 760px) {
            .block-container { padding: 1rem .75rem 2rem; }
            .dashboard-header { padding: 1.2rem; border-radius: 16px; }
            [data-testid="stMetric"] { min-height: 96px; padding: .8rem; }
            [data-testid="stMetricValue"] { font-size: 1.45rem; }
            [data-testid="stSidebar"] { min-width: 0; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_documents_loaded() -> list:
    """Load documents into session state on first visit, or when asked to reload."""
    if "in_flight" not in st.session_state:
        st.session_state.in_flight = False
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
    if "uploaded_documents" not in st.session_state:
        st.session_state.uploaded_documents = {}
    if "upload_widget_generation" not in st.session_state:
        st.session_state.upload_widget_generation = 0
    if "chunks" not in st.session_state:
        _reload_documents()
    return _current_document_chunks()


def _current_document_chunks() -> list:
    """Combine the cached local files with this Streamlit session's uploads."""
    return _combine_document_chunks(
        st.session_state.get("chunks", []),
        st.session_state.get("uploaded_documents", {}),
    )


def _combine_document_chunks(local_chunks: list, uploaded_documents: dict) -> list:
    """Combine local chunks with successful uploads from one session state."""
    chunks = list(local_chunks)
    for uploaded in uploaded_documents.values():
        if uploaded.get("status") == "ready":
            chunks.extend(uploaded["chunks"])
    return chunks


def _folder_signature(folder: Path) -> str:
    """Fingerprint the folder so Streamlit recaches only when files change."""
    if not folder.exists() or not folder.is_dir():
        return "missing"
    parts: list[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
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
    """Read the documents/ folder and reset the current conversation."""
    documents_dir = default_documents_dir()
    signature = _folder_signature(documents_dir)
    with st.spinner("Loading documents..."):
        st.session_state.chunks = _load_documents_cached(str(documents_dir), signature)
    st.session_state.answer = None
    st.session_state.conversation = []
    st.session_state.documents_dir = str(documents_dir)


def _document_counts(chunks: list) -> tuple[list[str], int, int]:
    filenames = sorted({chunk.filename for chunk in chunks})
    pdf_count = sum(Path(name).suffix.lower() == ".pdf" for name in filenames)
    excel_count = sum(Path(name).suffix.lower() in {".xlsx", ".xls"} for name in filenames)
    return filenames, pdf_count, excel_count


def _format_file_size(size: int) -> str:
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _show_upload_section() -> None:
    """Accept, validate, and list uploads without persisting their raw bytes."""
    st.markdown("### Add documents")
    st.caption(
        "Upload PDF or Excel files to analyze them alongside documents in the "
        "workspace folder. Files stay in this browser session."
    )
    generation = st.session_state.upload_widget_generation
    uploader_key = f"document_uploader_{generation}"
    uploaded_files = st.file_uploader(
        "Upload PDF or Excel documents",
        type=["pdf", "xlsx", "xls"],
        accept_multiple_files=True,
        key=uploader_key,
        max_upload_size=MAX_UPLOAD_FILE_BYTES // (1024 * 1024),
        help=(
            "Up to 10 files and 30 MB total per session; maximum 10 MB per file. "
            "Supported formats: PDF, XLSX, XLS."
        ),
    )
    if uploaded_files:
        if _process_uploaded_files(uploaded_files):
            st.session_state.upload_widget_generation += 1
            st.rerun()

    documents = st.session_state.uploaded_documents
    if documents:
        st.markdown("#### Uploaded documents")
        for upload_id, uploaded in list(documents.items()):
            suffix = Path(uploaded["filename"]).suffix.lower()
            icon = "📕" if suffix == ".pdf" else "📊"
            columns = st.columns([6, 2, 1])
            columns[0].write(f"{icon} {uploaded['filename']}")
            columns[1].caption(_format_file_size(uploaded["size"]))
            if uploaded["status"] == "ready":
                columns[1].success(f"{len(uploaded['chunks'])} chunks")
            else:
                st.error(f"{uploaded['filename']}: {uploaded['error']}")
            columns[2].button(
                "Remove",
                key=f"remove_upload_{upload_id}",
                on_click=_remove_uploaded_document,
                args=(upload_id,),
                use_container_width=True,
            )
        st.caption(
            f"{sum(item['size'] for item in documents.values()):,} bytes across "
            f"{len(documents)} upload(s)"
        )
    st.button(
        "Clear Uploaded Documents",
        key="clear_uploaded_documents",
        on_click=_clear_uploaded_documents,
        disabled=not documents,
    )


def _process_uploaded_files(uploaded_files: list) -> bool:
    """Process each new upload once, retaining chunks and status only."""
    documents = st.session_state.uploaded_documents
    processed_new_file = False
    for uploaded in uploaded_files:
        content = uploaded.getvalue()
        filename = safe_uploaded_filename(uploaded.name)
        upload_id = hashlib.sha256(
            filename.encode("utf-8") + b"\0" + content
        ).hexdigest()
        if upload_id in documents:
            continue

        processed_new_file = True

        size = len(content)
        record: dict[str, object] = {
            "filename": filename,
            "size": size,
            "status": "error",
            "chunks": [],
            "error": "",
        }
        documents[upload_id] = record

        if len(documents) > MAX_UPLOAD_COUNT:
            record["error"] = f"Session limit is {MAX_UPLOAD_COUNT} uploaded files."
            del content
            continue

        total_size = size + sum(
            int(item["size"])
            for item in documents.values()
            if item.get("status") == "ready"
        )
        if size > MAX_UPLOAD_FILE_BYTES:
            record["error"] = (
                f"File exceeds the {MAX_UPLOAD_FILE_BYTES // (1024 * 1024)} MB per-file limit."
            )
            del content
            continue
        if total_size > MAX_UPLOAD_TOTAL_BYTES:
            record["error"] = (
                f"Session upload limit is "
                f"{MAX_UPLOAD_TOTAL_BYTES // (1024 * 1024)} MB total."
            )
            del content
            continue

        try:
            used_names = {chunk.filename for chunk in _current_document_chunks()}
            if filename in used_names:
                path = Path(filename)
                filename = (
                    f"{path.stem} (upload {upload_id[:8]}){path.suffix.lower()}"
                )
                record["filename"] = filename
            with st.spinner(f"Processing {filename}..."):
                chunks = load_uploaded_document(filename, content)
            record["chunks"] = chunks
            record["status"] = "ready"
            _invalidate_conversation()
            st.success(f"Uploaded and processed {filename}.")
        except ValueError as exc:
            record["error"] = str(exc)
            logger.info(
                "upload.rejected filename=%s size=%s reason=%s",
                filename,
                size,
                str(exc),
            )
        except Exception as exc:
            record["error"] = "The document could not be processed."
            logger.exception(
                "upload.processing_failed filename=%s error_type=%s",
                filename,
                type(exc).__name__,
            )
        finally:
            del content
    return processed_new_file


def _remove_uploaded_document(upload_id: str) -> None:
    """Remove one upload and its chunks from this user's active collection."""
    st.session_state.uploaded_documents.pop(upload_id, None)
    st.session_state.upload_widget_generation += 1
    _invalidate_conversation()


def _clear_uploaded_documents() -> None:
    """Clear every uploaded document and reset the uploader widget selection."""
    st.session_state.uploaded_documents = {}
    st.session_state.upload_widget_generation += 1
    _invalidate_conversation()


def _invalidate_conversation() -> None:
    st.session_state.answer = None
    st.session_state.conversation = []


def _show_sidebar(chunks: list) -> None:
    """Render branding, loaded-file status, and conversation/document controls."""
    filenames, pdf_count, excel_count = _document_counts(chunks)
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="sidebar-brand-mark">✦</span>'
            '<span>AI Document Assistant</span></div>'
            '<div class="sidebar-subtitle">Your workspace for document intelligence</div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("#### Workspace")
        status_columns = st.columns(2)
        status_columns[0].metric("Documents", len(filenames))
        status_columns[1].metric("PDF / Excel", f"{pdf_count} / {excel_count}")

        with st.expander("Loaded files", expanded=True):
            if filenames:
                uploaded_names = {
                    item["filename"]
                    for item in st.session_state.uploaded_documents.values()
                    if item.get("status") == "ready"
                }
                for filename in filenames:
                    suffix = Path(filename).suffix.lower()
                    icon = "📕" if suffix == ".pdf" else "📊"
                    source_label = "Uploaded" if filename in uploaded_names else "Local"
                    st.text(f"{icon} {filename} · {source_label}")
            else:
                st.info("No supported documents are loaded.")
            st.caption(f"Folder: {st.session_state.get('documents_dir', str(default_documents_dir()))}")

        if st.button("Reload documents", use_container_width=True):
            _load_documents_cached.clear()
            _reload_documents()
            st.rerun()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.conversation = []
            st.session_state.answer = None
            st.rerun()

        api_configured = bool(_api_key_from_env())
        st.divider()
        st.caption(
            f"Gemini API · {'Configured' if api_configured else 'Not configured'}"
        )
        if st.session_state.get("answer") and st.session_state.answer.model:
            st.caption(f"Active model · {st.session_state.answer.model}")


def _show_header() -> None:
    st.markdown(
        """
        <section class="dashboard-header">
            <div class="dashboard-eyebrow">AI-powered document workspace</div>
            <h1>Find answers in your documents.</h1>
            <p>Ask questions across your PDFs and spreadsheets. Get grounded answers,
            transparent calculations, and source references you can verify.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _show_summary_cards(chunks: list) -> None:
    filenames, pdf_count, excel_count = _document_counts(chunks)
    cards = st.columns(4)
    cards[0].metric("Documents", len(filenames), help="Loaded supported files")
    cards[1].metric("PDF files", pdf_count)
    cards[2].metric("Excel files", excel_count)
    cards[3].metric("Searchable chunks", len(chunks))


def _show_question_form(chunks: list) -> None:
    """Collect a question and call the existing agent when Ask AI is submitted."""
    st.markdown("### Ask your documents")
    if not st.session_state.conversation:
        st.markdown(
            '<p class="example-caption">Try a question to get started</p>',
            unsafe_allow_html=True,
        )
        example_columns = st.columns(2)
        for index, example in enumerate(EXAMPLE_QUESTIONS):
            if example_columns[index % 2].button(
                example,
                key=f"example_question_{index}",
                use_container_width=True,
            ):
                st.session_state.question_input = example
                st.rerun()

    with st.form("ask_form", clear_on_submit=True):
        question = st.text_area(
            "Your question",
            key="question_input",
            placeholder="Ask about a cost, a project requirement, or a specific source...",
            height=90,
            label_visibility="collapsed",
        )
        ask_clicked = st.form_submit_button("Ask AI  →", type="primary")

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
        with st.spinner("Analyzing your documents..."):
            answer = ask_question(cleaned, chunks)
            st.session_state.answer = answer
            st.session_state.conversation.append(
                {"question": cleaned, "answer": answer}
            )
    finally:
        st.session_state.in_flight = False
        logger.info("timing.ui_ask seconds=%.2f", time.perf_counter() - started)


def _show_conversation(chunks: list) -> None:
    """Render user/assistant turns with readable sources and structured data."""
    conversation = st.session_state.get("conversation", [])
    for turn_index, turn in enumerate(conversation):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        answer = turn["answer"]
        if not isinstance(answer, AgentAnswer):
            with st.chat_message("assistant"):
                st.error("The assistant returned an unexpected result.")
            continue
        with st.chat_message("assistant"):
            _render_answer_text(answer)
            _render_excel_results(answer, chunks, turn_index)
            if answer.model or answer.elapsed_seconds is not None:
                details = []
                if answer.model:
                    details.append(f"Model: {answer.model}")
                if answer.elapsed_seconds is not None:
                    details.append(f"{answer.elapsed_seconds:.2f}s")
                st.caption(" · ".join(details))
            _render_sources(answer)

    if not conversation:
        st.info(
            "Your answers will appear here with source citations and any relevant "
            "spreadsheet results."
        )


def _render_answer_text(answer: AgentAnswer) -> None:
    if answer.is_fallback:
        st.warning("Local search results — Gemini is temporarily unavailable.")
        text = answer.text.split("\n", 1)
        st.markdown(text[1] if len(text) > 1 else answer.text)
    elif answer.error:
        st.error(answer.text)
    else:
        st.markdown(answer.text)


def _render_excel_results(
    answer: AgentAnswer,
    chunks: list,
    turn_index: int,
) -> None:
    """Show structured calculations and matching records without dumping JSON."""
    seen: set[tuple[object, ...]] = set()
    for result_index, result in enumerate(answer.excel_results):
        operation = result.get("operation")
        if not operation or result.get("error"):
            continue
        key = (
            operation,
            result.get("column"),
            result.get("value"),
            result.get("offset"),
            result.get("row_count"),
        )
        if key in seen:
            continue
        seen.add(key)

        references = result.get("source_references") or []
        if operation in {"average", "sum", "min", "max", "count"}:
            if "value" in result:
                label = {
                    "average": "Average",
                    "sum": "Sum",
                    "min": "Minimum",
                    "max": "Maximum",
                    "count": "Count",
                }[str(operation)]
                st.metric(f"{label} · {result.get('column', 'selected column')}", result["value"])
                st.caption(f"{result.get('row_count', 0)} numeric value(s) included")
            elif operation == "average" and result.get("row_count") == 0:
                st.info(result.get("message", "No numeric values matched the criteria."))
        elif operation == "filter":
            st.markdown(
                f"**{result.get('row_count', 0)} matching records**"
                f" · showing matches {int(result.get('offset', 0)) + 1}–"
                f"{int(result.get('offset', 0)) + len(result.get('rows') or [])}"
            )
        elif operation == "list_columns":
            sheets = result.get("sheets") or []
            if sheets:
                st.dataframe(
                    [
                        {
                            "File": sheet.get("filename"),
                            "Sheet": sheet.get("sheet_name"),
                            "Columns": ", ".join(sheet.get("columns") or []),
                            "Rows": sheet.get("row_count"),
                        }
                        for sheet in sheets
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        rows = result.get("rows") or []
        if rows and isinstance(rows, list):
            table_rows = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                fields = row.get("fields")
                table_rows.append(
                    {
                        "File": row.get("filename"),
                        "Sheet": row.get("sheet_name"),
                        "Row": row.get("row_number"),
                        **(fields if isinstance(fields, dict) else {}),
                    }
                )
            if table_rows:
                st.dataframe(table_rows, use_container_width=True, hide_index=True)

        next_offset = result.get("next_offset")
        if operation == "filter" and next_offset is not None:
            if st.button(
                "Load more matching rows",
                key=f"excel_next_{turn_index}_{result_index}_{next_offset}",
            ):
                next_page = analyze_excel(
                    chunks,
                    operation="filter",
                    column=str(result.get("column", "")),
                    threshold=result.get("threshold"),
                    comparison=str(result.get("comparison", "greater_than")),
                    offset=next_offset,
                    limit=max(1, len(rows)),
                )
                result["next_offset"] = None
                answer.excel_results.append(next_page)
                st.rerun()

        if references:
            st.caption("Excel sources: " + " · ".join(str(item) for item in references))


def _render_sources(answer: AgentAnswer) -> None:
    labels = answer.source_labels()
    if not labels:
        if answer.sources or answer.reference_labels:
            st.info("No document sources were returned for this answer.")
        return
    with st.expander(f"Sources · {len(labels)}", expanded=False):
        for label in labels:
            st.markdown(f"- {label}")


if __name__ == "__main__":
    main()
