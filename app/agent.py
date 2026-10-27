"""Gemini agent that answers document questions using genuine function calling.

The model receives local search and structured Excel tools. Python executes
requested calls and returns their results before the model writes its answer.
Documents are never uploaded to Gemini in bulk.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from app.models import AgentAnswer, DocumentChunk
from app.tools import (
    MAX_EXCEL_PAGE_SIZE,
    SearchResult,
    analyze_excel,
    search_documents,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-3.1-flash-lite"
FALLBACK_MODEL_NAMES = ("gemini-3.8-flash",)
MAX_MODEL_ATTEMPTS = 2
MAX_TOOL_ROUNDS = 3
MAX_TOOL_CALLS = 8
MAX_RESULTS_TO_MODEL = 5
MAX_PASSAGE_CHARS = 800
MAX_QUESTION_CHARS = 4000
THINKING_LEVEL = types.ThinkingLevel.LOW
REQUEST_TIMEOUT_MS = 30_000
RETRY_ATTEMPTS = 1

_cached_client: object | None = None

SYSTEM_INSTRUCTION = """
You are a document question-answering assistant.

Rules:
- Use an appropriate document tool before answering factual questions:
  search_documents for text and analyze_excel for spreadsheet analysis.
- Use analyze_excel for spreadsheet column discovery, numeric filters, extrema,
  sums, averages, and counts. Never calculate an aggregate from a partial search result.
- For questions asking for the maximum, minimum, total, average, mean, or count from a
  spreadsheet, call analyze_excel with the matching operation; do not use
  search_documents for numerical aggregation.
- If the user names a metric such as progress but not its exact header, pass that
  metric as the analyze_excel column argument. For an unclear metric, ask which
  numeric column they mean rather than selecting one arbitrarily.
- Do not call tools for greetings or unrelated general questions. Be clear that
  this assistant answers questions about the loaded documents.
- Interpret obvious spelling mistakes, grammatical errors, and paraphrases without
  asking the user to rephrase when the intent is clear. Search using meaningful
  subject terms rather than generic request words.
- Use only evidence returned by the tools: search passages for textual answers
  and analyze_excel results for spreadsheet calculations.
- Never invent filenames, page numbers, sheet names, row numbers, or numeric values.
- If the tools return no matching evidence, say the information was not found
  in the loaded documents. Do not guess numeric answers.
- When you use a passage, cite its source: filename and PDF page, or filename plus Excel sheet and row.
- Call search_documents only when you need new evidence. Do not repeat the same query.
- If the tool results already answer the question, respond immediately without another tool call.
- For paginated Excel filters, request each next_offset until it is null or the
  tool-call limit is reached; never claim a truncated result is complete.
""".strip()

SEARCH_DOCUMENTS_DECLARATION = types.FunctionDeclaration(
    name="search_documents",
    description=(
        "Search already-loaded PDF and Excel documents. Returns matching "
        "passages with source metadata: filename, PDF page_number, Excel "
        "sheet_name and row_number. Normalize obvious misspellings and use "
        "meaningful subject terms from the user's question. Call this before "
        "answering. Do not call it again with the same query."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "query": types.Schema(
                type=types.Type.STRING,
                description="Keywords to search for in the loaded documents.",
            ),
            "max_results": types.Schema(
                type=types.Type.INTEGER,
                description="Maximum number of passages to return (1 to 5).",
            ),
        },
        required=["query"],
    ),
)

ANALYZE_EXCEL_DECLARATION = types.FunctionDeclaration(
    name="analyze_excel",
    description=(
        "Analyze all loaded Excel rows using typed cell values. Use list_columns "
        "first when the relevant sheet or numeric column is unclear. Supports "
        "filtering numeric values by threshold, max, min, sum, average, and count. "
        "For max, min, sum, average, count, and filter, provide the column argument. "
        "Omit column only for list_columns. If the question clearly identifies "
        "a metric but column is omitted, the application can resolve it from "
        "the original question only when exactly one numeric column matches. "
        "If unclear, ask the user which column they mean. For filter results, "
        "request subsequent pages using next_offset until it is null. "
        "Never calculate sums from search hits."
    ),
    parameters=types.Schema(
        type=types.Type.OBJECT,
        properties={
            "operation": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "list_columns",
                    "filter",
                    "max",
                    "min",
                    "sum",
                    "average",
                    "count",
                ],
            ),
            "column": types.Schema(
                type=types.Type.STRING,
                description=(
                    "Required for max, min, sum, average, count, and filter; omit only "
                    "for list_columns. Use the numeric header or a clear "
                    "variation such as cost, amount, progress, or quantity."
                ),
            ),
            "threshold": types.Schema(
                type=types.Type.STRING,
                description="Numeric threshold; currency symbols and separators are accepted.",
            ),
            "comparison": types.Schema(
                type=types.Type.STRING,
                enum=[
                    "greater_than",
                    "greater_than_or_equal",
                    "less_than",
                    "less_than_or_equal",
                    "equal",
                ],
            ),
            "offset": types.Schema(type=types.Type.INTEGER),
            "limit": types.Schema(
                type=types.Type.INTEGER,
                description=f"Rows per filter page, from 1 to {MAX_EXCEL_PAGE_SIZE}.",
            ),
        },
        required=["operation"],
    ),
)


class DocumentAgent:
    """Ask Gemini questions grounded in preloaded DocumentChunk objects."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None,
        client: object | None = None,
    ) -> None:
        self.chunks = list(chunks or [])
        self._client = client
        self._tool_cache: dict[str, tuple[dict, list[SearchResult]]] = {}

    def ask(self, question: str) -> AgentAnswer:
        """Run the function-calling loop for one user question."""
        started = time.perf_counter()
        self._tool_cache = {}
        cleaned = _clean_question(question)
        if not cleaned:
            return _timed_answer(
                AgentAnswer(text="Please ask a question about the loaded documents."),
                started,
            )

        if not self.chunks:
            return _timed_answer(
                AgentAnswer(
                    text="No documents are loaded, so the question cannot be answered."
                ),
                started,
            )

        try:
            client = self._get_client()
        except Exception as exc:
            category = _classify_api_error(exc)
            logger.warning("Gemini client initialization failed category=%s", category)
            return _timed_answer(
                self._local_fallback(
                    cleaned,
                    _api_failure_message(category),
                    error="api_failure",
                ),
                started,
            )
        if client is None:
            return _timed_answer(
                self._local_fallback(
                    cleaned,
                    "GEMINI_API_KEY is not set. Add it to your environment or .env file.",
                    error="missing_api_key",
                ),
                started,
            )

        contents: list[types.Content] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=cleaned)],
            )
        ]
        sources: list[SearchResult] = []
        reference_labels: list[str] = []
        gemini_calls = 0
        tool_executions = 0
        selected_model: str | None = None
        attempted_models: set[str] = set()

        tool_history: set[str] = set()
        # assign via gemini agent 
        structured_results: list[dict] = []

        final_only_next = False

        for round_index in range(MAX_TOOL_ROUNDS + 1):
            allow_tools = round_index < MAX_TOOL_ROUNDS and not final_only_next
            final_only_next = False
            try:
                response, selected_model = _generate_content(
                    client,
                    contents,
                    allow_tools=allow_tools,
                    selected_model=selected_model,
                    attempted_models=attempted_models,
                )
                gemini_calls += 1
                calls = _response_function_calls(response)
                finish_reason = _response_finish_reason(response)
                response_text = (getattr(response, "text", None) or "").strip()
                logger.info(
                    "agent.model_response model=%s response_type=%s finish_reason=%s "
                    "allow_tools=%s function_calls=%s has_text=%s",
                    selected_model,
                    type(response).__name__,
                    finish_reason,
                    allow_tools,
                    len(calls),
                    bool(response_text),
                )
            except ModelRequestError as exc:
                logger.warning(
                    "Gemini request failed model=%s category=%s",
                    exc.model,
                    exc.category,
                )
                deterministic = _deterministic_excel_answer(structured_results)
                if deterministic:
                    deterministic.model = selected_model
                    logger.warning(
                        "agent.termination reason=api_failure_after_excel_analysis "
                        "using_structured_result=true"
                    )
                    return _timed_answer(deterministic, started)
                clarification = _clarification_from_results(structured_results)
                if clarification:
                    logger.warning(
                        "agent.termination reason=api_failure_after_ambiguous_analysis "
                        "asking_column_clarification=true"
                    )
                    return _timed_answer(
                        AgentAnswer(
                            text=clarification,
                            model=selected_model,
                        ),
                        started,
                    )
                return _timed_answer(
                    self._local_fallback(
                        cleaned,
                        _api_failure_message(exc.category),
                        error="api_failure",
                        sources=sources,
                        reference_labels=reference_labels,
                    ),
                    started,
                )

            if not calls:
                if not response_text:
                    if structured_results:
                        deterministic = _deterministic_excel_answer(structured_results)
                        if deterministic:
                            deterministic.sources = _unique_sources(sources)
                            deterministic.reference_labels = list(
                                dict.fromkeys(reference_labels)
                            )
                            deterministic.model = selected_model
                            logger.warning(
                                "agent.termination reason=empty_model_text "
                                "using_structured_result=true"
                            )
                            return _timed_answer(deterministic, started)
                        clarification = _clarification_from_results(structured_results)
                        if clarification:
                            logger.warning(
                                "agent.termination reason=empty_model_text "
                                "asking_column_clarification=true"
                            )
                            return _timed_answer(
                                AgentAnswer(
                                    text=clarification,
                                    sources=_unique_sources(sources),
                                    reference_labels=list(
                                        dict.fromkeys(reference_labels)
                                    ),
                                    model=selected_model,
                                ),
                                started,
                            )
                    logger.warning(
                        "agent.termination reason=empty_model_text "
                        "structured_results=%s finish_reason=%s",
                        len(structured_results),
                        finish_reason,
                    )
                    text = "No answer was returned from the model."
                    error = "empty_response"
                else:
                    text = response_text
                    error = None
                logger.info(
                    "timing.ask_complete model=%s gemini_calls=%s tool_executions=%s seconds=%.2f",
                    selected_model,
                    gemini_calls,
                    tool_executions,
                    time.perf_counter() - started,
                )
                return _timed_answer(
                    AgentAnswer(
                        text=text,
                        sources=_unique_sources(sources),
                        reference_labels=list(dict.fromkeys(reference_labels)),
                        error=error,
                        model=selected_model,
                        excel_results=list(structured_results),
                    ),
                    started,
                )

            if not allow_tools:
                logger.warning(
                    "agent.termination reason=function_call_returned_without_tool_config "
                    "function_calls=%s",
                    len(calls),
                )
                _append_model_turn(contents, response, calls)
                for call in calls:
                    if tool_executions >= MAX_TOOL_CALLS:
                        payload, new_sources = (
                            {"error": "The maximum number of tool calls was reached."},
                            [],
                        )
                    else:
                        payload, new_sources = self._run_tool_call(
                            call,
                            structured_results,
                            question_context=cleaned,
                        )
                        tool_executions += 1
                    reference_labels.extend(payload.get("source_references", []))
                    sources.extend(new_sources)
                    tool_parts = [_function_response_part(call, payload)]
                    contents.append(types.Content(role="user", parts=tool_parts))
                # If the model returned a function call but tools are disabled, we will not
                # allow another tool call. Instead, we will return the model's text or a fallback.
                deterministic = _deterministic_excel_answer(structured_results)
                if deterministic:
                    deterministic.sources = _unique_sources(sources)
                    deterministic.reference_labels = list(
                        dict.fromkeys(reference_labels)
                    )
                    deterministic.model = selected_model
                    return _timed_answer(deterministic, started)
                clarification = _clarification_from_results(structured_results)
                if clarification:
                    return _timed_answer(
                        AgentAnswer(
                            text=clarification,
                            sources=_unique_sources(sources),
                            reference_labels=list(dict.fromkeys(reference_labels)),
                            model=selected_model,
                        ),
                        started,
                    )
                return _timed_answer(
                    AgentAnswer(
                        text=(
                            "The model requested another tool after the tool-call "
                            "budget was exhausted, and no verified Excel calculation "
                            "is available to answer safely."
                        ),
                        sources=_unique_sources(sources),
                        error="tool_limit",
                        reference_labels=list(dict.fromkeys(reference_labels)),
                        model=selected_model,
                    ),
                    started,
                )

            _append_model_turn(contents, response, calls)
            # result of response tools to gemini
            tool_parts: list[types.Part] = []
            # if we have repeated calls we will use cashing strategy here
            repeated_calls: list[bool] = []
            # for many calls in same round .. 
            for call in calls:
                if tool_executions >= MAX_TOOL_CALLS:
                    payload, new_sources = (
                        {"error": "The maximum number of tool calls was reached."},
                        [],
                    )
                    logger.error(
                        "agent.tool_execution name=%s success=false reason=call_limit",
                        _call_name(call),
                    )
                else:
                    payload, new_sources = self._run_tool_call(
                        call,
                        structured_results,
                        question_context=cleaned,
                    )
                    tool_executions += 1
                call_key = _tool_call_cache_key(call)
                repeated = call_key in tool_history
                if call_key:
                    tool_history.add(call_key)
                repeated_calls.append(repeated)
                if repeated:
                    payload = dict(payload)
                    payload["repeated_request"] = True
                reference_labels.extend(payload.get("source_references", []))
                sources.extend(new_sources)
                # change the response of function call to structured response from sdk gemini 
                tool_parts.append(_function_response_part(call, payload))
            # on second round same tool call will be repeated and we will use cached response from previous call
            # send the content to get a respone till reach final response  
            contents.append(types.Content(role="user", parts=tool_parts))
            
            if repeated_calls and all(repeated_calls):
                final_only_next = True
        # section for final answer after tool call limit reached
        deterministic = _deterministic_excel_answer(structured_results)
        if deterministic:
            deterministic.sources = _unique_sources(sources)
            deterministic.reference_labels = list(dict.fromkeys(reference_labels))
            deterministic.model = selected_model
            logger.warning(
                "agent.termination reason=tool_round_limit "
                "using_structured_result=true",
            )
            return _timed_answer(deterministic, started)
        clarification = _clarification_from_results(structured_results)
        if clarification:
            logger.warning("agent.termination reason=ambiguous_excel_column")
            return _timed_answer(
                AgentAnswer(
                    text=clarification,
                    sources=_unique_sources(sources),
                    reference_labels=list(dict.fromkeys(reference_labels)),
                    model=selected_model,
                ),
                started,
            )
        logger.warning(
            "agent.termination reason=tool_round_limit "
            "using_structured_result=false tool_executions=%s",
            tool_executions,
        )
        return _timed_answer(
            AgentAnswer(
                text=(
                    "The model did not produce a final answer within the tool-call "
                    "limit, and no verified structured result is available."
                ),
                sources=_unique_sources(sources),
                error="tool_limit",
                reference_labels=list(dict.fromkeys(reference_labels)),
                model=selected_model,
            ),
            started,
        )

    def _local_fallback(
        self,
        question: str,
        reason: str,
        error: str,
        sources: list[SearchResult] | None = None,
        reference_labels: list[str] | None = None,
    ) -> AgentAnswer:
        """Return clearly labeled keyword results without presenting them as AI."""
        results = search_documents(
            self.chunks,
            question,
            max_results=MAX_RESULTS_TO_MODEL,
        )
        lines = [
            "Local Search Results — AI temporarily unavailable.",
            f"Reason: {reason}",
            (
                "These are local keyword matches, not an AI-generated answer. "
                "They do not perform spreadsheet calculations or establish that "
                "the listed matches are complete."
            ),
        ]
        if results:
            lines.append("")
            for result in results:
                lines.append(
                    f"- {_truncate(result.text, MAX_PASSAGE_CHARS)} "
                    f"({format_source(result)})"
                )
        else:
            lines.extend(
                [
                    "",
                    "No local keyword matches were found. The question cannot be "
                    "answered accurately without the AI or a relevant structured "
                    "Excel analysis.",
                ]
            )
        return AgentAnswer(
            text="\n".join(lines),
            sources=_unique_sources([*(sources or []), *results]),
            error=error,
            reference_labels=list(dict.fromkeys(reference_labels or [])),
            is_fallback=True,
        )

    def _get_client(self) -> object | None:
        """Return the injected client, or a reused client built from GEMINI_API_KEY."""
        if self._client is not None:
            return self._client
        api_key = _api_key_from_env()
        if not api_key:
            return None
        return _shared_client(api_key)
# call is the request (gemini sdk need to extract name and arguments) of gemini while question_context is the real question of the client 
# return is payload and and results its send back to generate a response via gemini
    def _execute_tool(
        self,
        call: object,
        question_context: str | None = None,
    ) -> tuple[dict, list[SearchResult]]:
        """Execute a model-requested local search or structured Excel tool."""
        name = _call_name(call)
        args = _call_args(call)
        cache_args = args
        if name == "search_documents":
            cache_args = {
                "query": _query_cache_key(args.get("query")),
                "max_results": min(
                    MAX_RESULTS_TO_MODEL,
                    max(1, _safe_int(args.get("max_results"), MAX_RESULTS_TO_MODEL)),
                ),
            }
            # create a json format and sort key mean if the argument is same then,
            #  we will use the cache result instead of calling the function again
            # and if the argument is (x,y) or (y,x) we also use the cashe result because sorting key 
        cache_key = json.dumps(
            [name, cache_args],
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if cache_key in self._tool_cache:
            payload, cached_results = self._tool_cache[cache_key]
            repeated_payload = dict(payload)
            repeated_payload["repeated_request"] = True
            return repeated_payload, cached_results

        if name == "search_documents":
            requested_limit = args.get("max_results", MAX_RESULTS_TO_MODEL)
            try:
                result_limit = int(requested_limit)
            except (TypeError, ValueError, OverflowError):
                result_limit = MAX_RESULTS_TO_MODEL
            result_limit = max(1, min(result_limit, MAX_RESULTS_TO_MODEL))
            search_started = time.perf_counter()
            results = search_documents(
                self.chunks,
                args.get("query"),
                max_results=result_limit,
            )
            # create to structured dictonary to send back to gemini 
            payload = _search_payload(results)
            logger.info(
                "timing.local_search hits=%s seconds=%.4f",
                len(results),
                time.perf_counter() - search_started,
            )
            tool_result = (payload, results)
        elif name == "analyze_excel":
            payload = analyze_excel(
                self.chunks,
                operation=str(args.get("operation", "")),
                column=args.get("column"),
                threshold=args.get("threshold"),
                comparison=str(args.get("comparison", "greater_than")),
                offset=args.get("offset", 0),
                limit=args.get("limit", MAX_EXCEL_PAGE_SIZE),
                question_context=question_context,
            )
            # not (payload, results) because excel send the analyzed data in payload 
            tool_result = (payload, [])
        else:
            tool_result = ({"error": f"Unknown tool: {name}"}, [])
        # save the result into cashe 
        self._tool_cache[cache_key] = tool_result
        return tool_result
# safe layer that call execute tool 
    def _run_tool_call(
        self,
        call: object,
        # list of the excel structured result 
        structured_results: list[dict],
        question_context: str | None = None,
    ) -> tuple[dict, list[SearchResult]]:
        """Run one model call and retain structured results for safe synthesis."""
        name = _call_name(call)
        args = _call_args(call)
        logger.info(
            "agent.tool_requested name=%s arguments=%s",
            name or "<missing>",
            _safe_log_arguments(args),
        )
        try:
            payload, new_sources = self._execute_tool(
                call,
                question_context=question_context,
            )
        except Exception as exc:
            safe_error = _safe_tool_error(str(exc))
            logger.error(
                "agent.tool_execution name=%s success=false error_type=%s error=%s",
                name or "<missing>",
                type(exc).__name__,
                safe_error,
            )
            return (
                {
                    "error": (
                        f"Tool execution failed ({type(exc).__name__}): "
                        f"{safe_error}"
                    )
                },
                [],
            )

        if name == "analyze_excel":
            structured_results.append(payload)
        if "error" in payload:
            logger.warning(
                "agent.tool_execution name=%s success=false error=%s",
                name or "<missing>",
                _safe_tool_error(str(payload["error"])),
            )
        logger.info(
            "agent.tool_execution name=%s success=%s result_keys=%s hit_count=%s",
            name or "<missing>",
            "error" not in payload,
            sorted(payload),
            len(new_sources),
        )
        return payload, new_sources


def ask_question(
    question: str,
    chunks: list[DocumentChunk] | None,
    client: object | None = None,
) -> AgentAnswer:
    """Convenience wrapper around DocumentAgent.ask."""
    return DocumentAgent(chunks, client=client).ask(question)


def format_source(result: SearchResult) -> str:
    """Format one search hit as a citation string."""
    if result.source_type == "pdf":
        return f"{result.filename}, page {result.page_number}"
    return f"{result.filename}, sheet {result.sheet_name}, row {result.row_number}"


def _api_key_from_env() -> str:
    """Read the key from the environment or Streamlit's app secrets."""
    load_dotenv()
    environment_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if environment_key:
        return environment_key
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except (KeyError, StreamlitSecretNotFoundError):
        return ""


def _shared_client(api_key: str) -> object:
    """Reuse one Gemini client so later questions skip extra connection setup."""
    global _cached_client
    if _cached_client is None:
        _cached_client = genai.Client(
            api_key=api_key,
            http_options=_http_options(),
        )
    return _cached_client


def _http_options() -> types.HttpOptions:
    """Cap request latency and avoid SDK-level retry storms."""
    return types.HttpOptions(
        timeout=REQUEST_TIMEOUT_MS,
        retry_options=types.HttpRetryOptions(
            attempts=RETRY_ATTEMPTS,
            http_status_codes=[408, 500, 502, 503, 504],
        ),
    )


class ModelRequestError(Exception):
    """A safe, classified API error without retaining provider request details."""

    def __init__(self, category: str, model: str) -> None:
        self.category = category
        self.model = model
        super().__init__(f"{category} for {model}")


def _configured_model() -> str:
    load_dotenv()
    return (os.getenv("GEMINI_MODEL") or MODEL_NAME).strip() or MODEL_NAME


def _model_candidates() -> list[str]:
    """Return a short list of models selected for Gemini tool-calling support."""
    return list(
        dict.fromkeys([_configured_model(), MODEL_NAME, *FALLBACK_MODEL_NAMES])
    )


def _generate_content(
    client: object,
    contents: list,
    allow_tools: bool,
    selected_model: str | None = None,
    attempted_models: set[str] | None = None,
) -> tuple[object, str]:
    # if response is succes return the response and model name if not try another model
    """Try at most two eligible models and return the model that succeeded."""
    # use the attempted_models set to avoid retrying the same model in a loop
    attempted = attempted_models if attempted_models is not None else set()
    # get all model distinct in a list [model-A , model-B]
    candidates = _model_candidates()
    if selected_model:
        # if the selected model success in last time resort the list too start with select model
        candidates = [selected_model] + [
            candidate for candidate in candidates if candidate != selected_model
        ]
        # delete the attempt model from the list of candidates to avoid retrying it
    candidates = [model for model in candidates if model not in attempted]
    # how model can try for example each call for method try 2 models
    candidates = candidates[:MAX_MODEL_ATTEMPTS]
    last_error: ModelRequestError | None = None

    for model in candidates:
        started = time.perf_counter()
        response = None
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=_build_config(allow_tools=allow_tools),
            )
            logger.info(
                "timing.gemini_request model=%s allow_tools=%s seconds=%.2f",
                model,
                allow_tools,
                time.perf_counter() - started,
            )
            return response, model
        except Exception as exc:
            attempted.add(model)
            category = _classify_api_error(exc)
            logger.warning(
                "Gemini request failed model=%s category=%s seconds=%.2f",
                model,
                category,
                time.perf_counter() - started,
            )
            last_error = ModelRequestError(category, model)
            # can fallback if the error is not authentication
            #  or invalid request because these errors are not temporary and will not be solved 
            # by retrying another model
            if not _can_fallback(category):
                raise last_error from None

    if last_error is not None:
        raise last_error
    raise ModelRequestError("unavailable", selected_model or MODEL_NAME)


def _classify_api_error(error: Exception) -> str:
    """Classify expected provider failures without logging raw exception text."""
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    status_text = str(status or "").upper()
    details = f"{type(error).__name__} {error}".upper()
    if "429" in status_text or "429" in details or "RESOURCE_EXHAUSTED" in details:
        return "rate_limited"
    if "404" in status_text or "404" in details or "NOT_FOUND" in details:
        return "not_found"
    if "503" in status_text or "503" in details or "UNAVAILABLE" in details:
        return "unavailable"
    if any(
        marker in status_text or marker in details
        for marker in ("504", "DEADLINE_EXCEEDED", "GATEWAY TIMEOUT")
    ):
        return "timeout"
    if any(marker in status_text or marker in details for marker in ("401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED")):
        return "authentication"
    if any(marker in status_text or marker in details for marker in ("400", "INVALID_ARGUMENT")):
        return "invalid_request"
    if isinstance(error, TimeoutError) or "TIMEOUT" in details:
        return "timeout"
    if any(marker in details for marker in ("CONNECTIONERROR", "CONNECTERROR", "NETWORKERROR", "CONNECTION RESET")):
        return "network"
    return "unknown"


def _can_fallback(category: str) -> bool:
    return category in {
        "not_found",
        "rate_limited",
        "unavailable",
        "timeout",
        "network",
    }


def _api_failure_message(category: str) -> str:
    messages = {
        "not_found": "The configured Gemini model was not found for this API key.",
        "rate_limited": "Gemini API quota or rate limit was reached for the available models.",
        "unavailable": "Gemini is temporarily unavailable or at high demand.",
        "timeout": "The Gemini request timed out.",
        "network": "A network error prevented the Gemini request.",
        "authentication": "Gemini authentication failed. Check GEMINI_API_KEY and its API access.",
        "invalid_request": "Gemini rejected the request configuration or selected model.",
    }
    return messages.get(category, "The Gemini API request failed.")


def _build_config(allow_tools: bool) -> types.GenerateContentConfig:
    """Use SDK function declarations but execute all tools in local Python."""
    config_kwargs: dict = {
        "system_instruction": SYSTEM_INSTRUCTION,
        # here mean only python can do the calling 
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
        "max_output_tokens": 4096,
        "thinking_config": types.ThinkingConfig(
            thinking_level=THINKING_LEVEL,
            include_thoughts=False,
        ),
        "http_options": _http_options(),
    }
    # if allow tools enable can add if not remember ask we are at final round and canot be added 
    if allow_tools:
        config_kwargs["tools"] = [
            types.Tool(
                function_declarations=[
                    SEARCH_DOCUMENTS_DECLARATION,
                    ANALYZE_EXCEL_DECLARATION,
                ]
            )
        ]
        # auto here mean gemini choose return text or a function calling 
        config_kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.AUTO
            )
        )
    return types.GenerateContentConfig(**config_kwargs)


def _append_model_turn(
    contents: list,
    response: object,
    calls: list | None = None,
) -> None:
    """Keep the model function-call turn in the conversation history."""
    # get the response from candidates 
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        if calls:
            contents.append(
                types.Content(
                    role="model",
                    # each function call has a part and save id if have 
                    parts=[
                        types.Part(
                            function_call=types.FunctionCall(
                                name=_call_name(call),
                                args=_call_args(call),
                                id=getattr(call, "id", None),
                            )
                        )
                        for call in calls
                    ],
                )
            )
        return
    content = getattr(candidates[0], "content", None)
    if content is not None:
        # to know the response from each masseg 
        contents.append(content)


def _response_function_calls(response: object) -> list:
    # its use to search for the function call in response 
    """Read calls from the SDK convenience property or candidate content parts."""
    calls = list(getattr(response, "function_calls", None) or [])
    if calls:
        return calls

    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return []
    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    return [
        call
        for part in parts
        # := mean assign call to getattr then check its python syntax
        if (call := getattr(part, "function_call", None)) is not None
    ]
# parts in gemini like : 
# parts = [
#     Part(text="I will analyze the Excel file."),
#     Part(
#         function_call=FunctionCall(
#             name="analyze_excel",
#             args={
#                 "operation": "max",
#                 "column": "Amount SAR"
#             }
#         )
#     )
# ]
# this the response at end : 
# calls = [
#     FunctionCall(
#         name="analyze_excel",
#         args={"operation": "max", "column": "Cost"}
#     )
# ]

def _response_finish_reason(response: object) -> str:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "unknown"
    reason = getattr(candidates[0], "finish_reason", None)
    if reason is None:
        return "unknown"
    return str(getattr(reason, "name", None) or reason)

# for no appear sensitive key or info use this methode to hide the security info
def _safe_log_arguments(arguments: dict) -> dict:
    """Log short tool inputs while redacting values that resemble credentials."""
    safe: dict[str, object] = {}
    for key, value in arguments.items():
        normalized_key = str(key).lower()
        if any(marker in normalized_key for marker in ("key", "token", "secret", "password")):
            safe[str(key)] = "[redacted]"
        elif isinstance(value, (str, int, float, bool)) or value is None:
            safe[str(key)] = _truncate(str(value), 160) if value is not None else None
        else:
            safe[str(key)] = f"<{type(value).__name__}>"
    return safe


def _safe_tool_error(error: str, limit: int = 300) -> str:
    """Keep tool errors useful in logs without leaking credentials or control chars."""
    safe_error = re.sub(r"[\r\n\t]+", " ", error)
    safe_error = re.sub(
        r"(?i)(api[_ -]?key|token|secret|password)(\s*[:=]\s*)\S+",
        r"\1\2[redacted]",
        safe_error,
    )
    return _truncate(safe_error, limit)

# get result has the same tool name and operation (argument )
def _tool_call_cache_key(call: object) -> str:
    return json.dumps(
        [_call_name(call), _call_args(call)],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _deterministic_excel_answer(results: list[dict]) -> AgentAnswer | None:
    """Render a verified local Excel result when the model cannot synthesize."""
    for result in reversed(results):
        operation = result.get("operation")
        if result.get("error"):
            continue
        if operation in {"max", "min", "sum", "count", "average"} and "value" in result:
            rows = result.get("rows") or []
            column = str(
                (rows[0].get("column") if rows else None)
                or result.get("column")
                or "numeric column"
            )
            value = result["value"]
            action = {
                "max": "Maximum",
                "min": "Minimum",
                "sum": "Total",
                "count": "Count",
                "average": "Average",
            }[operation]
            text = f"{action} for {column}: {value}."
            if rows:
                row_summaries = [
                    ", ".join(
                        f"{key}: {value}"
                        for key, value in row.get("fields", {}).items()
                        if key != row.get("column")
                    )
                    for row in rows[:5]
                ]
                details = [item for item in row_summaries if item]
                if details:
                    text += " Matching row details: " + "; ".join(details) + "."
            references = result.get("source_references") or []
            if references:
                text += " Sources: " + "; ".join(references) + "."
            return AgentAnswer(
                text=text,
                reference_labels=list(references),
                excel_results=list(results),
            )

        if operation == "average" and result.get("row_count") == 0:
            return AgentAnswer(
                text=(
                    "No numeric values matched the selected column and criteria, "
                    f"so an average for {result.get('column', 'the selected column')} "
                    "could not be calculated."
                ),
                reference_labels=list(result.get("source_references") or []),
                excel_results=list(results),
            )

        if operation == "filter":
            rows = result.get("rows") or []
            if not rows:
                return AgentAnswer(
                    text=(
                        f"No rows matched the filter for {result.get('column', 'the selected column')}."
                    ),
                    reference_labels=list(result.get("source_references") or []),
                    excel_results=list(results),
                )
            references = result.get("source_references") or []
            shown = ", ".join(
                f"{row.get('filename')}, sheet {row.get('sheet_name')}, "
                f"row {row.get('row_number')}: {row.get('value')}"
                for row in rows[:5]
            )
            text = (
                f"The filter matched {result.get('row_count', len(rows))} rows. "
                f"Shown result page: {shown}."
            )
            if references:
                text += " Sources: " + "; ".join(references) + "."
            return AgentAnswer(
                text=text,
                reference_labels=list(references),
                excel_results=list(results),
            )

    return None

# results = [
#     {"operation": "count", "value": 1000},
#     {"error": "Column required", "numeric_columns": ["Amount SAR", "Quantity"]}
# ]
def _clarification_from_results(results: list[dict]) -> str | None:
    for result in reversed(results):
        if result.get("numeric_columns"):
            columns = ", ".join(str(item) for item in result["numeric_columns"])
            return (
                "Which numeric column should I use for this calculation? "
                f"Available columns: {columns}."
            )
    return None

# type.part is a chunck of gemini masseg 
def _function_response_part(call: object, payload: dict) -> types.Part:
    """Send tool output back, including the call id when Gemini provided one."""
    return types.Part(
        function_response=types.FunctionResponse(
            name=_call_name(call) or "search_documents",
            response=payload,
            # if have many fct calling to know each one 
            id=getattr(call, "id", None),
        )
    )


def _search_payload(results: list[SearchResult], repeated: bool = False) -> dict:
    """JSON-serializable tool output with bounded passage text."""
    if not results:
        return {
            "matches": [],
            "message": "No matching passages were found in the loaded documents.",
        }
    message = (
        "Answer using only these passages. Cite filename and page "
        "or sheet/row. Do not invent values. If this evidence is enough, "
        "do not call search_documents again."
    )
    if repeated:
        message = "This query was already searched. " + message
    return {
        "matches": [_match_dict(result) for result in results],
        "message": message,
    }


def _match_dict(result: SearchResult) -> dict:
    """Copy source metadata for Gemini without sending oversized text."""
    return {
        "text": _truncate(result.text, MAX_PASSAGE_CHARS),
        "filename": result.filename,
        "source_type": result.source_type,
        "page_number": result.page_number,
        "sheet_name": result.sheet_name,
        "row_number": result.row_number,
    }

# get unique source dont duplicate 
def _unique_sources(sources: list[SearchResult]) -> list[SearchResult]:
    """Preserve first-seen source locations across multiple tool calls."""
    unique: list[SearchResult] = []
    seen: set[tuple] = set()
    for source in sources:
        key = (
            source.filename,
            source.source_type,
            source.page_number,
            source.sheet_name,
            source.row_number,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(source)
    return unique

# query1 = "Python Developer"
# query2 = "python developer"
# query3 = "  PYTHON   Developer  "
# this 3 is the same 
# create key from tool and argument name for cashing 
def _query_cache_key(query: object) -> str:
    """Normalize a search query so repeated Gemini calls can reuse local results."""
    if not isinstance(query, str):
        return ""
    return " ".join(query.lower().split())


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _call_name(call: object) -> str:
    return str(getattr(call, "name", "") or "")


def _call_args(call: object) -> dict:
    args = getattr(call, "args", None)
    return dict(args) if isinstance(args, dict) else {}


def _clean_question(question: object) -> str:
    if not isinstance(question, str):
        return ""
    return _truncate(question.strip(), MAX_QUESTION_CHARS)


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _timed_answer(answer: AgentAnswer, started: float) -> AgentAnswer:
    answer.elapsed_seconds = round(time.perf_counter() - started, 2)
    return answer
