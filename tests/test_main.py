"""Tests for Streamlit helpers. These do not start the web app or call Gemini."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.main import should_skip_duplicate_request
from app import main as main_module
from app.agent import _configured_model
from app.models import AgentAnswer, DocumentChunk


class FakeSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def test_duplicate_in_flight_request_is_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=True) is True


def test_fresh_click_is_not_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=False) is False


def test_idle_rerun_is_not_treated_as_a_request() -> None:
    assert should_skip_duplicate_request(clicked=False, in_flight=False) is False
    assert should_skip_duplicate_request(clicked=False, in_flight=True) is False


def test_streamlit_secrets_configure_key_and_model_without_logging_values(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "old-env-key")
    monkeypatch.setenv("GEMINI_MODEL", "old-env-model")
    cloud_key = "cloud-test-key-not-for-logs"
    cloud_model = "cloud-test-model"
    fake_streamlit = SimpleNamespace(
        secrets={"GEMINI_API_KEY": cloud_key, "GEMINI_MODEL": cloud_model}
    )

    with patch.object(main_module, "st", fake_streamlit):
        main_module._configure_gemini_from_streamlit_secrets()

    assert main_module._api_key_from_env() == cloud_key
    assert _configured_model() == cloud_model
    assert cloud_key not in caplog.text


def test_conversation_passes_loaded_chunks_to_excel_renderer() -> None:
    chunks = [object()]
    answer = AgentAnswer(text="Answer")
    fake_streamlit = SimpleNamespace(
        session_state={"conversation": [{"question": "Question?", "answer": answer}]},
        chat_message=Mock(return_value=nullcontext()),
        markdown=Mock(),
        caption=Mock(),
    )

    with (
        patch.object(main_module, "st", fake_streamlit),
        patch.object(main_module, "_render_answer_text"),
        patch.object(main_module, "_render_excel_results") as render_excel,
        patch.object(main_module, "_render_sources"),
    ):
        main_module._show_conversation(chunks)

    render_excel.assert_called_once_with(answer, chunks, 0)


def test_document_collections_are_isolated_by_session_state() -> None:
    first_chunk = DocumentChunk("First", "first.pdf", "pdf", page_number=1)
    second_chunk = DocumentChunk("Second", "second.pdf", "pdf", page_number=1)
    first_state = FakeSessionState(
        chunks=[],
        uploaded_documents={
            "first": {"status": "ready", "chunks": [first_chunk]}
        },
    )
    second_state = FakeSessionState(
        chunks=[],
        uploaded_documents={
            "second": {"status": "ready", "chunks": [second_chunk]}
        },
    )
    fake_streamlit = SimpleNamespace(session_state=first_state)

    with patch.object(main_module, "st", fake_streamlit):
        assert main_module._current_document_chunks() == [first_chunk]
        fake_streamlit.session_state = second_state
        assert main_module._current_document_chunks() == [second_chunk]


def test_removing_uploaded_document_invalidates_previous_answers() -> None:
    answer = AgentAnswer(text="Old answer")
    state = FakeSessionState(
        uploaded_documents={
            "upload": {"status": "ready", "chunks": [object()]}
        },
        upload_widget_generation=2,
        answer=answer,
        conversation=[{"question": "Old question", "answer": answer}],
    )

    with patch.object(main_module, "st", SimpleNamespace(session_state=state)):
        main_module._remove_uploaded_document("upload")

    assert state.uploaded_documents == {}
    assert state.upload_widget_generation == 3
    assert state.answer is None
    assert state.conversation == []


def test_clearing_uploads_invalidates_previous_answers() -> None:
    answer = AgentAnswer(text="Old answer")
    state = FakeSessionState(
        uploaded_documents={"upload": {"status": "ready", "chunks": [object()]}},
        upload_widget_generation=0,
        answer=answer,
        conversation=[{"question": "Old question", "answer": answer}],
    )

    with patch.object(main_module, "st", SimpleNamespace(session_state=state)):
        main_module._clear_uploaded_documents()

    assert state.uploaded_documents == {}
    assert state.upload_widget_generation == 1
    assert state.answer is None
    assert state.conversation == []


def test_processing_uploads_adds_chunks_to_current_session(
    sample_pdf_path,
    sample_xlsx_path,
) -> None:
    class FakeUpload:
        def __init__(self, path) -> None:
            self.name = path.name
            self._content = path.read_bytes()

        def getvalue(self) -> bytes:
            return self._content

    local_chunk = DocumentChunk("Local", "local.pdf", "pdf", page_number=1)
    state = FakeSessionState(
        uploaded_documents={},
        chunks=[local_chunk],
        answer=AgentAnswer(text="Old answer"),
        conversation=[{"question": "Old", "answer": AgentAnswer(text="Old")}],
    )
    fake_streamlit = SimpleNamespace(
        session_state=state,
        spinner=Mock(side_effect=lambda _: nullcontext()),
        success=Mock(),
    )

    with patch.object(main_module, "st", fake_streamlit):
        changed = main_module._process_uploaded_files(
            [FakeUpload(sample_pdf_path), FakeUpload(sample_xlsx_path)]
        )
        combined = main_module._current_document_chunks()

    assert changed is True
    assert {chunk.filename for chunk in combined} == {
        "local.pdf",
        "sample.pdf",
        "sample.xlsx",
    }
    assert len(state.uploaded_documents) == 2
    assert all(
        "content" not in record
        for record in state.uploaded_documents.values()
    )
    assert state.answer is None
    assert state.conversation == []


def test_processing_upload_respects_total_session_size_limit(
    sample_pdf_path,
) -> None:
    class FakeUpload:
        name = sample_pdf_path.name

        def getvalue(self) -> bytes:
            return sample_pdf_path.read_bytes()

    state = FakeSessionState(
        uploaded_documents={
            "existing": {
                "filename": "existing.pdf",
                "size": 30 * 1024 * 1024 - 1,
                "status": "ready",
                "chunks": [],
            }
        },
        chunks=[],
        answer=None,
        conversation=[],
    )
    fake_streamlit = SimpleNamespace(
        session_state=state,
        spinner=Mock(side_effect=lambda _: nullcontext()),
        success=Mock(),
    )

    with patch.object(main_module, "st", fake_streamlit):
        main_module._process_uploaded_files([FakeUpload()])

    rejected = [
        item
        for item in state.uploaded_documents.values()
        if item["filename"] == "sample.pdf"
    ][0]
    assert rejected["status"] == "error"
    assert "30 MB total" in rejected["error"]
