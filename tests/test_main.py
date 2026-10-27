"""Tests for Streamlit helpers. These do not start the web app or call Gemini."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.models import AgentAnswer
from app.main import should_skip_duplicate_request
from app import main as main_module


def test_duplicate_in_flight_request_is_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=True) is True


def test_fresh_click_is_not_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=False) is False


def test_idle_rerun_is_not_treated_as_a_request() -> None:
    assert should_skip_duplicate_request(clicked=False, in_flight=False) is False
    assert should_skip_duplicate_request(clicked=False, in_flight=True) is False


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
