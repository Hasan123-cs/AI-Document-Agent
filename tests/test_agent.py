"""Tests for Gemini function calling. These tests never call the live API."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from google.genai import types

from app.models import DocumentChunk
from app.agent import (
    ANALYZE_EXCEL_DECLARATION,
    MODEL_NAME,
    REQUEST_TIMEOUT_MS,
    SEARCH_DOCUMENTS_DECLARATION,
    SYSTEM_INSTRUCTION,
    _api_key_from_env,
    _configured_model,
    ask_question,
)
from app.tools import search_documents


def _pdf_chunk() -> DocumentChunk:
    return DocumentChunk(
        text="The pump pressure rating is 150 PSI.",
        filename="guide.pdf",
        source_type="pdf",
        page_number=2,
    )


def _xlsx_chunk() -> DocumentChunk:
    return DocumentChunk(
        text="Item: Widget | Qty: 3",
        filename="inventory.xlsx",
        source_type="xlsx",
        sheet_name="Inventory",
        row_number=2,
    )


def _progress_chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            text=f"Activity: {activity} | Actual Progress %: {progress}",
            filename="unfamiliar.xlsx",
            source_type="xlsx",
            sheet_name="Controls",
            row_number=row,
            excel_values={"Activity": activity, "Actual Progress %": progress},
        )
        for row, activity, progress in (
            (3, "Design", 40),
            (4, "Construction", 100),
        )
    ]


def _function_call(
    query: str,
    max_results: int = 5,
    name: str = "search_documents",
    args: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        args=args or {"query": query, "max_results": max_results},
    )


def _response(*, text: str | None = None, function_calls: list | None = None):
    """Build a generate_content-like object the agent can inspect."""
    if function_calls:
        parts = [
            SimpleNamespace(function_call=call, text=None) for call in function_calls
        ]
    else:
        parts = [SimpleNamespace(function_call=None, text=text)]
    content = SimpleNamespace(role="model", parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(
        text=text,
        function_calls=function_calls,
        candidates=[candidate],
    )


def _call_response(call: SimpleNamespace):
    content = SimpleNamespace(
        role="model",
        parts=[SimpleNamespace(function_call=call, text=None)],
    )
    return SimpleNamespace(
        text=None,
        candidates=[SimpleNamespace(content=content, finish_reason="STOP")],
    )


def _mock_client(responses: list) -> MagicMock:
    client = MagicMock()
    client.models.generate_content.side_effect = responses
    return client


def test_successful_function_calling_uses_local_search_and_sources() -> None:
    chunks = [_pdf_chunk(), _xlsx_chunk()]
    client = _mock_client(
        [
            _response(function_calls=[_function_call("pump pressure")]),
            _response(
                text=(
                    "The pump pressure rating is 150 PSI "
                    "(guide.pdf, page 2)."
                )
            ),
        ]
    )

    answer = ask_question("What is the pump pressure?", chunks, client=client)

    assert answer.error is None
    assert "150 PSI" in answer.text
    assert "guide.pdf" in answer.text
    assert len(answer.sources) == 1
    assert answer.sources[0].filename == "guide.pdf"
    assert answer.sources[0].page_number == 2
    assert answer.source_labels() == ["guide.pdf, page 2"]

    calls = client.models.generate_content.call_args_list
    assert len(calls) == 2
    assert [call.kwargs["model"] for call in calls] == [MODEL_NAME, MODEL_NAME]
    first_kwargs = calls[0].kwargs
    assert first_kwargs["model"] == MODEL_NAME
    declared_names = [
        declaration.name
        for declaration in first_kwargs["config"].tools[0].function_declarations
    ]
    assert declared_names == [
        SEARCH_DOCUMENTS_DECLARATION.name,
        ANALYZE_EXCEL_DECLARATION.name,
    ]
    assert "average" in ANALYZE_EXCEL_DECLARATION.parameters.properties[
        "operation"
    ].enum
    assert first_kwargs["config"].thinking_config.thinking_level == types.ThinkingLevel.LOW
    assert (
        first_kwargs["config"].tool_config.function_calling_config.mode
        == types.FunctionCallingConfigMode.AUTO
    )
    assert first_kwargs["config"].automatic_function_calling.disable is True

    second_contents = calls[1].kwargs["contents"]
    function_responses = [
        part.function_response
        for content in second_contents
        for part in content.parts
        if getattr(part, "function_response", None) is not None
    ]
    assert function_responses
    payload = function_responses[0].response
    assert payload["matches"][0]["filename"] == "guide.pdf"
    assert payload["matches"][0]["page_number"] == 2
    assert "150 PSI" in payload["matches"][0]["text"]
    assert second_contents[1].role == "model"
    assert second_contents[2].role == "user"


def test_typo_ridden_prompt_is_searched_and_returns_source() -> None:
    client = _mock_client(
        [
            _response(function_calls=[_function_call("give me all prgress")]),
            _response(text="Project progress is recorded (guide.pdf, page 2)."),
        ]
    )
    chunk = DocumentChunk(
        text="Project progress includes engineering, procurement, and construction milestones.",
        filename="status.pdf",
        source_type="pdf",
        page_number=1,
    )

    answer = ask_question("give me all prgress", [chunk], client=client)

    assert answer.error is None
    assert answer.sources[0].filename == "status.pdf"
    assert answer.sources[0].page_number == 1
    assert "spelling mistakes" in SYSTEM_INSTRUCTION


def test_excel_source_metadata_is_returned_to_gemini() -> None:
    client = _mock_client(
        [
            _response(function_calls=[_function_call("Widget")]),
            _response(text="Widget quantity is 3 (inventory.xlsx, sheet Inventory, row 2)."),
        ]
    )

    answer = ask_question("How many widgets?", [_xlsx_chunk()], client=client)

    assert answer.sources[0].sheet_name == "Inventory"
    assert answer.sources[0].row_number == 2
    assert answer.source_labels() == ["inventory.xlsx, sheet Inventory, row 2"]


def test_excel_analysis_is_requested_and_executed_as_a_real_tool_call() -> None:
    chunks = [
        DocumentChunk(
            text="Item: pump | Cost (SAR): 1500000",
            filename="boq.xlsx",
            source_type="xlsx",
            sheet_name="Costs",
            row_number=2,
            excel_values={"Item": "pump", "Cost (SAR)": 1_500_000},
        ),
        DocumentChunk(
            text="Item: valve | Cost (SAR): 2500000",
            filename="boq.xlsx",
            source_type="xlsx",
            sheet_name="Costs",
            row_number=3,
            excel_values={"Item": "valve", "Cost (SAR)": 2_500_000},
        ),
    ]
    client = _mock_client(
        [
            _response(
                function_calls=[
                    _function_call(
                        "",
                        name="analyze_excel",
                        args={
                            "operation": "max",
                            "column": "cost",
                        },
                    )
                ]
            ),
            _response(text="The maximum cost is SAR 2,500,000 (boq.xlsx, Costs, row 3)."),
        ]
    )

    answer = ask_question("What is the maximum cost?", chunks, client=client)

    assert answer.error is None
    assert answer.text.startswith("The maximum cost")
    assert answer.reference_labels == ["boq.xlsx, sheet Costs, rows 3"]
    assert answer.excel_results[0]["operation"] == "max"
    assert answer.excel_results[0]["value"] == 2_500_000
    tool_response = next(
        part.function_response
        for content in client.models.generate_content.call_args_list[1].kwargs["contents"]
        for part in content.parts
        if getattr(part, "function_response", None) is not None
    )
    assert tool_response.name == "analyze_excel"
    assert tool_response.response["value"] == 2_500_000


def test_max_progress_phrasings_use_full_excel_analysis() -> None:
    phrasings = (
        "What is the maximum progress?",
        "Tell me about max progress",
        "Waht is max progres?",
        "Max progress",
        "What is max progress",
        "Show highest progress",
    )
    for question in phrasings:
        client = _mock_client(
            [
                _response(
                    function_calls=[
                        _function_call(
                            "",
                            name="analyze_excel",
                            args={
                                "operation": "max",
                                "column": "progres" if "progres" in question.lower() else "progress",
                            },
                        )
                    ]
                ),
                _response(text="Maximum actual progress is 100%."),
            ]
        )

        answer = ask_question(question, _progress_chunks(), client)

        assert answer.error is None
        assert "100%" in answer.text
        assert answer.source_labels() == [
            "unfamiliar.xlsx, sheet Controls, rows 4"
        ]
        first_config = client.models.generate_content.call_args_list[0].kwargs["config"]
        declarations = first_config.tools[0].function_declarations
        assert "analyze_excel" in [item.name for item in declarations]
        tool_response = next(
            part.function_response
            for content in client.models.generate_content.call_args_list[1].kwargs[
                "contents"
            ]
            for part in content.parts
            if getattr(part, "function_response", None) is not None
        )
        assert tool_response.name == "analyze_excel"
        assert tool_response.response["value"] == 100


def test_missing_excel_column_is_resolved_from_question_without_guessing() -> None:
    chunks = [
        DocumentChunk(
            text=f"Activity: {activity} | Actual Progress %: {progress} | Budget: {budget}",
            filename="unfamiliar.xlsx",
            source_type="xlsx",
            sheet_name="Controls",
            row_number=row,
            excel_values={
                "Activity": activity,
                "Actual Progress %": progress,
                "Budget": budget,
            },
        )
        for row, activity, progress, budget in (
            (3, "Design", 40, 500),
            (4, "Construction", 100, 900),
            (5, "Testing", 90, 700),
        )
    ]
    call = _function_call(
        "",
        name="analyze_excel",
        args={"operation": "max"},
    )
    client = _mock_client(
        [
            _response(function_calls=[call]),
            _response(text="Maximum progress is 100%."),
        ]
    )

    answer = ask_question("Tell me about max progres", chunks, client)

    assert answer.error is None
    assert "100%" in answer.text
    assert answer.source_labels() == [
        "unfamiliar.xlsx, sheet Controls, rows 4"
    ]
    tool_response = next(
        part.function_response
        for content in client.models.generate_content.call_args_list[1].kwargs[
            "contents"
        ]
        for part in content.parts
        if getattr(part, "function_response", None) is not None
    )
    assert tool_response.response["operation"] == "max"
    assert tool_response.response["column"] == "Actual Progress %"
    assert tool_response.response["value"] == 100


def test_verified_excel_max_survives_gemini_504() -> None:
    call = _function_call(
        "",
        name="analyze_excel",
        args={"operation": "max"},
    )
    client = _mock_client(
        [
            _response(function_calls=[call]),
            RuntimeError("504 GATEWAY TIMEOUT"),
            RuntimeError("504 GATEWAY TIMEOUT"),
        ]
    )

    answer = ask_question("What is maximum progress?", _progress_chunks(), client)

    assert answer.error is None
    assert not answer.is_fallback
    assert answer.text.startswith("Maximum for Actual Progress %: 100.")
    assert answer.source_labels() == [
        "unfamiliar.xlsx, sheet Controls, rows 4"
    ]


def test_function_call_only_response_is_executed_and_final_answer_follows() -> None:
    client = _mock_client(
        [
            _call_response(
                _function_call(
                    "",
                    name="analyze_excel",
                    args={"operation": "max", "column": "progress"},
                )
            ),
            _response(text="Maximum actual progress is 100%."),
        ]
    )

    answer = ask_question("Max progress", _progress_chunks(), client)

    assert answer.error is None
    assert "100%" in answer.text
    assert len(client.models.generate_content.call_args_list) == 2
    follow_up_contents = client.models.generate_content.call_args_list[1].kwargs[
        "contents"
    ]
    assert any(
        getattr(part, "function_response", None) is not None
        for content in follow_up_contents
        for part in content.parts
    )


def test_empty_final_text_uses_deterministic_structured_result() -> None:
    client = _mock_client(
        [
            _response(
                function_calls=[
                    _function_call(
                        "",
                        name="analyze_excel",
                        args={"operation": "max", "column": "progress"},
                    )
                ]
            ),
            _response(text=""),
        ]
    )

    answer = ask_question("What is max progress?", _progress_chunks(), client)

    assert answer.error is None
    assert answer.text.startswith("Maximum for Actual Progress %: 100.")
    assert "unfamiliar.xlsx, sheet Controls, rows 4" in answer.text
    assert answer.source_labels() == [
        "unfamiliar.xlsx, sheet Controls, rows 4"
    ]


def test_empty_final_text_uses_deterministic_average_result() -> None:
    chunks = [
        DocumentChunk(
            text=f"Activity: {activity} | Budget: {budget}",
            filename="boq.xlsx",
            source_type="xlsx",
            sheet_name="BOQ",
            row_number=row,
            excel_values={"Activity": activity, "Budget": budget},
        )
        for row, activity, budget in (
            (2, "Design", 100),
            (3, "Build", 200),
            (4, "Test", 300),
        )
    ]
    call = _function_call(
        "",
        name="analyze_excel",
        args={"operation": "average", "column": "budget"},
    )
    client = _mock_client(
        [
            _response(function_calls=[call]),
            _response(text=""),
        ]
    )

    answer = ask_question("What is the average budget?", chunks, client)

    assert answer.error is None
    assert answer.text.startswith("Average for Budget: 200.")
    assert answer.source_labels() == ["boq.xlsx, sheet BOQ, rows 2-4"]
    assert answer.excel_results[0]["operation"] == "average"
    assert answer.excel_results[0]["value"] == 200


def test_repeated_calls_force_synthesis_without_reexecuting_tool() -> None:
    call = _function_call(
        "",
        name="analyze_excel",
        args={"operation": "max", "column": "progress"},
    )
    client = _mock_client(
        [
            _response(function_calls=[call]),
            _response(function_calls=[call]),
            _response(text="Maximum actual progress is 100%."),
        ]
    )

    answer = ask_question("Max progress", _progress_chunks(), client)

    assert answer.error is None
    assert "100%" in answer.text
    assert len(client.models.generate_content.call_args_list) == 3
    assert client.models.generate_content.call_args_list[2].kwargs["config"].tools is None


def test_tool_call_on_final_request_returns_verified_excel_result() -> None:
    function_call = _function_call(
        "",
        name="analyze_excel",
        args={"operation": "max", "column": "progress"},
    )
    client = _mock_client(
        [
            _response(function_calls=[function_call]),
            _response(function_calls=[function_call]),
            _call_response(function_call),
        ]
    )

    answer = ask_question("Max progress", _progress_chunks(), client)

    assert answer.error is None
    assert answer.text.startswith("Maximum for Actual Progress %: 100.")
    assert answer.source_labels() == [
        "unfamiliar.xlsx, sheet Controls, rows 4"
    ]
    assert len(client.models.generate_content.call_args_list) == 3


def test_ambiguous_maximum_requests_numeric_column_clarification() -> None:
    chunks = [
        DocumentChunk(
            text="Progress: 75 | Budget: 500",
            filename="unfamiliar.xlsx",
            source_type="xlsx",
            sheet_name="Controls",
            row_number=3,
            excel_values={"Progress": 75, "Budget": 500},
        )
    ]
    client = _mock_client(
        [
            _response(
                function_calls=[
                    _function_call(
                        "",
                        name="analyze_excel",
                        args={"operation": "max"},
                    )
                ]
            ),
            _response(text=""),
        ]
    )

    answer = ask_question("What is max?", chunks, client)

    assert answer.error is None
    assert "Which numeric column" in answer.text
    assert "Budget" in answer.text
    assert "Progress" in answer.text
    assert "Maximum for" not in answer.text


def test_tool_round_limit_returns_last_verified_excel_calculation() -> None:
    calls = [
        _function_call(
            "",
            name="analyze_excel",
            args={"operation": "max", "column": "progress", "offset": index},
        )
        for index in range(3)
    ]
    client = _mock_client(
        [_response(function_calls=[call]) for call in calls]
        + [_response(function_calls=[calls[-1]])]
    )

    answer = ask_question("Show highest progress", _progress_chunks(), client)

    assert answer.error is None
    assert answer.text.startswith("Maximum for Actual Progress %: 100.")


def test_greeting_does_not_force_a_function_call() -> None:
    client = _mock_client([_response(text="Hello! How can I help with your documents?")])

    answer = ask_question("Hello", [_pdf_chunk()], client=client)

    assert answer.text.startswith("Hello!")
    assert len(client.models.generate_content.call_args_list) == 1
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.tool_config.function_calling_config.mode == types.FunctionCallingConfigMode.AUTO


def test_request_timeout_meets_gemini_minimum_deadline() -> None:
    assert REQUEST_TIMEOUT_MS >= 10_000


def test_no_search_matches_sends_empty_results_to_gemini() -> None:
    client = _mock_client(
        [
            _response(function_calls=[_function_call("turbine")]),
            _response(text="The information was not found in the loaded documents."),
        ]
    )

    answer = ask_question("What about the turbine?", [_pdf_chunk()], client=client)

    assert answer.sources == []
    assert "not found" in answer.text.lower()
    payload = None
    for content in client.models.generate_content.call_args_list[1].kwargs["contents"]:
        for part in content.parts:
            if getattr(part, "function_response", None) is not None:
                payload = part.function_response.response
    assert payload == {
        "matches": [],
        "message": "No matching passages were found in the loaded documents.",
    }


def test_missing_api_key_does_not_call_gemini(monkeypatch) -> None:
    monkeypatch.setattr("app.agent._api_key_from_env", lambda: "")
    client_factory = MagicMock()
    monkeypatch.setattr("app.agent.genai.Client", client_factory)

    answer = ask_question("What is the pump pressure?", [_pdf_chunk()])

    assert answer.error == "missing_api_key"
    assert "GEMINI_API_KEY" in answer.text
    client_factory.assert_not_called()


def test_gemini_configuration_is_read_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "local-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "local-test-model")
    monkeypatch.setattr("app.agent.load_dotenv", lambda: None)

    assert _api_key_from_env() == "local-test-key"
    assert _configured_model() == "local-test-model"


def test_gemini_api_failure_is_handled() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("service unavailable")

    answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

    assert answer.error == "unavailable"
    assert answer.is_fallback is True
    assert "Local Search Results — AI temporarily unavailable." in answer.text
    assert "150 PSI" in answer.text


def test_gemini_authentication_failure_is_actionable() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("401 UNAUTHENTICATED")

    answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

    assert answer.error == "authentication"
    assert "GEMINI_API_KEY" in answer.text


def test_gemini_invalid_request_failure_is_actionable() -> None:
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("404 NOT_FOUND")

    answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

    assert answer.error == "model_not_found"
    assert "configured Gemini model" in answer.text


def test_gemini_does_not_silently_switch_models(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "missing-model")
    client = MagicMock()
    client.models.generate_content.side_effect = RuntimeError("404 NOT_FOUND")

    answer = ask_question("Hello", [_pdf_chunk()], client=client)

    models = [
        call.kwargs["model"]
        for call in client.models.generate_content.call_args_list
    ]
    assert models == ["missing-model"]
    assert answer.error == "model_not_found"
    assert answer.model == "missing-model"


def test_configured_model_is_reused_for_follow_up_function_call(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_MODEL", "configured-model")
    client = MagicMock()
    client.models.generate_content.side_effect = [
        _response(function_calls=[_function_call("pump pressure")]),
        _response(text="The pump pressure is 150 PSI (guide.pdf, page 2)."),
    ]

    answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

    models = [
        call.kwargs["model"]
        for call in client.models.generate_content.call_args_list
    ]
    assert models == ["configured-model", "configured-model"]
    assert answer.error is None
    assert answer.sources[0].page_number == 2


def test_429_and_503_use_clear_local_fallbacks() -> None:
    for failure, expected in (
        ("429 RESOURCE_EXHAUSTED", "quota or rate limit"),
        ("503 UNAVAILABLE", "temporarily unavailable"),
    ):
        client = MagicMock()
        client.models.generate_content.side_effect = RuntimeError(failure)

        answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

        assert answer.error == (
            "rate_limited" if failure.startswith("429") else "unavailable"
        )
        assert answer.is_fallback is True
        assert expected in answer.text
        assert "guide.pdf, page 2" in answer.source_labels()


def test_timeout_and_network_failures_use_local_fallback() -> None:
    for failure in (TimeoutError("request timed out"), ConnectionError("network unreachable")):
        client = MagicMock()
        client.models.generate_content.side_effect = failure

        answer = ask_question("What is the pump pressure?", [_pdf_chunk()], client=client)

        assert answer.error == (
            "timeout" if isinstance(failure, TimeoutError) else "network"
        )
        assert answer.is_fallback is True
        assert "150 PSI" in answer.text


def test_empty_documents_skip_gemini() -> None:
    client = _mock_client([])

    answer = ask_question("What is the pump pressure?", [], client=client)

    assert "No documents are loaded" in answer.text
    client.models.generate_content.assert_not_called()


def test_repeated_search_query_does_not_scan_chunks_again() -> None:
    chunks = [_pdf_chunk()]
    client = _mock_client(
        [
            _response(
                function_calls=[
                    _function_call("pump pressure"),
                    _function_call("PUMP pressure"),
                ]
            ),
            _response(text="The pump pressure rating is 150 PSI (guide.pdf, page 2)."),
        ]
    )

    with patch("app.agent.search_documents", wraps=search_documents) as mocked_search:
        answer = ask_question("What is the pump pressure?", chunks, client=client)

    assert mocked_search.call_count == 1
    assert answer.sources[0].page_number == 2
    assert len(client.models.generate_content.call_args_list) == 2
