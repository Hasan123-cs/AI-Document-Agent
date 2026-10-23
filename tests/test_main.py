"""Tests for Streamlit helpers. These do not start the web app or call Gemini."""

from app.main import should_skip_duplicate_request


def test_duplicate_in_flight_request_is_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=True) is True


def test_fresh_click_is_not_skipped() -> None:
    assert should_skip_duplicate_request(clicked=True, in_flight=False) is False


def test_idle_rerun_is_not_treated_as_a_request() -> None:
    assert should_skip_duplicate_request(clicked=False, in_flight=False) is False
    assert should_skip_duplicate_request(clicked=False, in_flight=True) is False
