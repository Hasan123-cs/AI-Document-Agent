# AI Document Assistant — Project Context and Method Reference

> **Purpose for AI coding assistants:** Read this file first to understand the current application without re-reading every source file. This document describes the supplied code; it is **not a substitute for opening the actual function when editing it**. Before modifying code, inspect the current source and tests, since this reference can become outdated.

## Project overview

A local-document question-answering application built with **Python, Streamlit, Google Gemini (google-genai SDK), pandas, openpyxl, and PyMuPDF**. It reads `.pdf` and `.xlsx` files from the project-level `documents/` folder (the supplied UI does **not** implement browser uploads). It extracts source-aware `DocumentChunk`s, offers Gemini two local Python tools (`search_documents`, `analyze_excel`), and displays grounded answers and source references. Gemini selects function calls; **Python executes them**. The application does not upload complete documents to Gemini in bulk. Search passages and bounded tool results are sent to the model.

## Directory and module responsibilities

```text
project/
├── app/
│   ├── main.py             # Streamlit UI, cached local loading, question form, rendering
│   ├── document_loader.py  # Read PDFs and XLSX into source-aware DocumentChunk objects
│   ├── models.py           # DocumentChunk and AgentAnswer dataclasses
│   ├── tools.py            # Deterministic keyword search and typed Excel analysis
│   └── agent.py            # Gemini function-calling loop, cache, error/fallback handling
├── documents/              # Local input PDF/XLSX files
├── tests/                  # Tests referenced in project discussion (exact contents not supplied)
├── .env                    # Local secret configuration; do not commit
├── .env.example            # Placeholder configuration (if maintained in repository)
└── project.md              # This AI-readable project map
```

## End-to-end flow

1. Streamlit `main()` calls `_ensure_documents_loaded()` and displays loaded-file/chunk counts.
2. `load_documents()` scans the local `documents/` folder, dispatching `.pdf` to `load_pdf()` and `.xlsx` to `load_xlsx()`.
3. PDF: one non-empty text chunk per page, with filename and 1-based page number. XLSX: detect likely header row per sheet, then one non-empty chunk per data row, preserving filename, sheet, original 1-based Excel row and typed `excel_values`.
4. User submits question; UI invokes `ask_question(question, chunks)`.
5. `DocumentAgent.ask()` validates question/chunks, obtains Gemini client, and sends user question with system instruction and function declarations.
6. Gemini may return text or one/more function calls. `_response_function_calls()` extracts calls; `_run_tool_call()` safely invokes `_execute_tool()`, which dispatches local Python search/Excel operations and caches results.
7. Tool results are returned to Gemini as `types.FunctionResponse` parts. Loop is bounded by tool-round and call budgets; repeated calls are tracked.
8. Agent returns `AgentAnswer(text, sources, reference_labels, error, elapsed_seconds, is_fallback, model)`. Streamlit displays answer, model, response time, and deduplicated citations.
9. On missing/unavailable Gemini API, local keyword fallback is clearly labeled; after completed Excel operations, deterministic Excel answer can be synthesized without Gemini. Ambiguous numeric column can trigger a clarification.

## Data contracts

**`DocumentChunk`** (`@dataclass(frozen=True)`): `text: str`, `filename: str`, `source_type: str`, optional `page_number: int`, `sheet_name: str`, `row_number: int`, `excel_values: dict[str, object]`. Excel text is a human-readable `column: value | ...` string; `excel_values` retains typed cells for numeric operations.

**`SearchResult`** (`@dataclass(frozen=True)` in `tools.py`): `text`, `filename`, `source_type`, optional `page_number`, `sheet_name`, `row_number`, `score: int`.

**`AgentAnswer`** (`@dataclass`): `text`, `sources: list[SearchResult]`, `error`, `elapsed_seconds`, `reference_labels: list[str]`, `is_fallback: bool`, `model`. `source_labels()` combines `format_source(source)` for sources with `reference_labels`, preserving order and removing duplicates.

## Gemini configuration and safety boundaries (`app/agent.py`)

- Default `MODEL_NAME = "gemini-3.1-flash-lite"`; `FALLBACK_MODEL_NAMES = ("gemini-3.8-flash",)`; `GEMINI_MODEL` environment variable can override preferred model. Model availability depends on actual provider/account.
- `MAX_MODEL_ATTEMPTS = 2`, `MAX_TOOL_ROUNDS = 3`, `MAX_TOOL_CALLS = 8`, `MAX_RESULTS_TO_MODEL = 5`, `MAX_PASSAGE_CHARS = 800`, `MAX_QUESTION_CHARS = 4000`.
- `THINKING_LEVEL = LOW`, `REQUEST_TIMEOUT_MS = 30000`, `RETRY_ATTEMPTS = 1` (SDK retry configuration; do not assume this guarantees a retry).
- Tool definitions are `SEARCH_DOCUMENTS_DECLARATION` and `ANALYZE_EXCEL_DECLARATION`. `automatic_function_calling` is disabled, so Python explicitly executes returned function calls. Function-calling mode `AUTO` lets Gemini decide between tool call and text.
- `SYSTEM_INSTRUCTION` requires evidence-based answers, source citations, no fabricated numbers, numeric aggregation through `analyze_excel` rather than search hits, clarification when ambiguous, and pagination awareness.
- `.env` supplies `GEMINI_API_KEY`; `_safe_log_arguments()` and `_safe_tool_error()` redact potentially sensitive log details. Never paste actual secrets into this file or commits.
- The tool cache is reset at the start of each `DocumentAgent.ask()` invocation. Streamlit's `@st.cache_data` for document parsing is a **different cache**.

## Tool semantics (`app/tools.py`)

**`search_documents(chunks, query, max_results=5)`**: deterministic local keyword ranking (no embeddings or vector database). Normalizes to lowercase alphanumeric tokens; `_score_chunk` gives +100 for each distinct matched query token, +20 for full phrase match, and a term-frequency tie breaker. Supports one-edit typo matching for sufficiently long tokens. Returns top ranked `SearchResult`s with source metadata.

**`analyze_excel(chunks, operation, column=None, threshold=None, comparison="greater_than", offset=0, limit=50, question_context=None)`**: works only on XLSX chunks. Operations: `list_columns`, `filter`, `max`, `min`, `sum`, `count`. Uses `_resolve_column` to map user column names to sheet headers, including normalized direct matches, aliases and limited typo matching; refuses ambiguous matches. Uses `_parse_number` returning `Decimal` or `None` to ignore nonnumeric cells and handle currencies, separators, negatives and k/m/b suffixes. Filters numeric records by optional threshold/comparison. `filter` returns `row_count` (total matches), current `rows`, `offset`, and `next_offset`; each page capped at 50. `max/min` returns extremum and matching rows (up to 50); `sum/count` aggregates all matching numeric records, not just a page. Outputs include source references for Excel rows.

**Important:** `count` counts valid numeric matching records in the chosen column, not all rows in a sheet. `_resolve_column()` returns `None` on ambiguity rather than guessing. Excel `row_number` is the original 1-based worksheet row, not a pandas index. `_find_header_row()` is a heuristic, not guaranteed perfect.

## Fallback behavior and edge cases

- `DocumentAgent.ask()` handles blank question, no chunks, missing API key, client initialization failures, API failures, empty model output, unknown tools, tool errors, repeated calls, and tool limits.
- `_generate_content()` tries eligible models (up to 2 per invocation), classifies provider errors, and only tries alternate models for fallback-eligible error categories. `_classify_api_error()` distinguishes rate limits, not found, unavailable, timeout, authentication, invalid request, network and unknown.
- `_deterministic_excel_answer(structured_results)` uses **previously executed Excel results in the current question** for `max/min/sum/count/filter`; it is not a general AI substitute and does not calculate from incomplete search snippets.
- `_clarification_from_results()` asks the user to select a numeric column when tool results contain `numeric_columns`.
- `_local_fallback()` performs keyword search over loaded chunks, labels it as **local matches, not an AI-generated answer**, and explicitly says it does not perform spreadsheet calculations.
- The UI uses `should_skip_duplicate_request` and `st.session_state.in_flight` to reduce duplicate submissions; `_folder_signature` and `@st.cache_data` control parsing cache invalidation.

## AI assistant working instructions

When asked to implement a change: (1) identify the affected module/functions from the index below; (2) inspect the latest corresponding source before editing; (3) preserve existing public signatures and `DocumentChunk`/`AgentAnswer` contracts unless change is explicitly requested; (4) keep Gemini as selector and Python as executor; (5) do not turn keyword search into semantic search unless requested; (6) do not invent results or citations; (7) handle numeric calculations with `Decimal` and source metadata; (8) add/update focused tests; (9) avoid leaking API keys or raw provider error details; (10) explain modified files, behavior, and how to test. The context file is an index, not an instruction to skip source inspection for modifications.

## Complete function/method index from supplied code

The following index includes **all defined functions and methods in the supplied `agent.py` and `tools.py`**, extracted from the code, plus the functions provided inline for `main.py`, `document_loader.py`, and `models.py`. Signatures in the code-derived sections are automatically extracted. Private underscore-prefixed helpers are included.

### `app/main.py`

- **`should_skip_duplicate_request(clicked, in_flight)`** — Prevent repeat submission during an in-flight request.
- **`main()`** — Configure and render Streamlit page.
- **`_ensure_documents_loaded()`** — Initialize session state and load chunks once.
- **`_folder_signature(folder)`** — Fingerprint local files by filename, modification time and size.
- **`_load_documents_cached(folder, signature)`** — Streamlit-cached parsing of local documents.
- **`_reload_documents()`** — Reload local chunks and reset prior answer.
- **`_show_document_status(chunks)`** — Show counts, folder, filenames and reload button.
- **`_show_question_form(chunks)`** — Validate question and call ask_question, with in-flight guard.
- **`_show_answer()`** — Render AgentAnswer, fallback/error status, model, time and sources.

### `app/document_loader.py`

- **`default_documents_dir()`** — Return project documents/ path.
- **`load_documents(directory=None)`** — Read supported files; skip unsupported/broken files and log errors.
- **`load_pdf(path)`** — PyMuPDF text extraction; one chunk per nonempty page.
- **`load_xlsx(path)`** — pandas/openpyxl reading of every worksheet.
- **`_load_file(path)`** — Dispatch by extension.
- **`_iter_files(folder)`** — Stable sorted nonrecursive file iteration.
- **`_chunks_from_sheet(filename, sheet_name, frame, row_number_offset=2)`** — Convert each nonempty Excel data row into a chunk.
- **`_find_header_row(frame, scan_rows=10)`** — Choose most populated row among first ten as header.
- **`_frame_with_headers(frame, header_row)`** — Build clean unique headers and remaining data frame.
- **`_excel_row_number(pandas_index, offset=2)`** — Map data index to original Excel row number.
- **`_format_excel_row(row)`** — Serialize nonempty cells as column: value pairs.
- **`_excel_values(row)`** — Preserve typed cell values, nulls and date representations.
- **`_is_blank(value)`** — Identify missing/blank pandas cell values.

### `app/models.py`

- **`DocumentChunk (dataclass)`** — Immutable source-aware extracted text chunk.
- **`AgentAnswer (dataclass)`** — Answer text, citations, metadata and fallback/error flags.
- **`AgentAnswer.source_labels()`** — Combine formatted SearchResult citations and explicit reference labels, deduplicated.

### `app/agent.py`

- **`class DocumentAgent`** — Ask Gemini questions grounded in preloaded DocumentChunk objects. (line 154)
  - **`DocumentAgent.__init__(self, chunks: list[DocumentChunk] | None, client: object | None=None)`** — Internal method; see implementation. (line 157)
  - **`DocumentAgent.ask(self, question: str)`** — Run the function-calling loop for one user question. (line 166)
  - **`DocumentAgent._local_fallback(self, question: str, reason: str, error: str, sources: list[SearchResult] | None=None, reference_labels: list[str] | None=None)`** — Return clearly labeled keyword results without presenting them as AI. (line 492)
  - **`DocumentAgent._get_client(self)`** — Return the injected client, or a reused client built from GEMINI_API_KEY. (line 539)
  - **`DocumentAgent._execute_tool(self, call: object, question_context: str | None=None)`** — Execute a model-requested local search or structured Excel tool. (line 549)
  - **`DocumentAgent._run_tool_call(self, call: object, structured_results: list[dict], question_context: str | None=None)`** — Run one model call and retain structured results for safe synthesis. (line 621)
- **`ask_question(question: str, chunks: list[DocumentChunk] | None, client: object | None=None)`** — Convenience wrapper around DocumentAgent.ask. (line 677)
- **`format_source(result: SearchResult)`** — Format one search hit as a citation string. (line 686)
- **`_api_key_from_env()`** — Read GEMINI_API_KEY without logging or returning a missing value as secrets. (line 693)
- **`_shared_client(api_key: str)`** — Reuse one Gemini client so later questions skip extra connection setup. (line 699)
- **`_http_options()`** — Cap request latency and avoid SDK-level retry storms. (line 710)
- **`class ModelRequestError`** — A safe, classified API error without retaining provider request details. (line 721)
  - **`ModelRequestError.__init__(self, category: str, model: str)`** — Internal method; see implementation. (line 724)
- **`_configured_model()`** — Internal helper; see implementation. (line 730)
- **`_model_candidates()`** — Return a short list of models selected for Gemini tool-calling support. (line 735)
- **`_generate_content(client: object, contents: list, allow_tools: bool, selected_model: str | None=None, attempted_models: set[str] | None=None)`** — Try at most two eligible models and return the model that succeeded. (line 742)
- **`_classify_api_error(error: Exception)`** — Classify expected provider failures without logging raw exception text. (line 803)
- **`_can_fallback(category: str)`** — Internal helper; see implementation. (line 832)
- **`_api_failure_message(category: str)`** — Internal helper; see implementation. (line 842)
- **`_build_config(allow_tools: bool)`** — Use SDK function declarations but execute all tools in local Python. (line 855)
- **`_append_model_turn(contents: list, response: object, calls: list | None=None)`** — Keep the model function-call turn in the conversation history. (line 887)
- **`_response_function_calls(response: object)`** — Read calls from the SDK convenience property or candidate content parts. (line 920)
- **`_response_finish_reason(response: object)`** — Internal helper; see implementation. (line 959)
- **`_safe_log_arguments(arguments: dict)`** — Log short tool inputs while redacting values that resemble credentials. (line 969)
- **`_safe_tool_error(error: str, limit: int=300)`** — Keep tool errors useful in logs without leaking credentials or control chars. (line 983)
- **`_tool_call_cache_key(call: object)`** — Internal helper; see implementation. (line 994)
- **`_deterministic_excel_answer(results: list[dict])`** — Render a verified local Excel result when the model cannot synthesize. (line 1003)
- **`_clarification_from_results(results: list[dict])`** — Internal helper; see implementation. (line 1073)
- **`_function_response_part(call: object, payload: dict)`** — Send tool output back, including the call id when Gemini provided one. (line 1084)
- **`_search_payload(results: list[SearchResult], repeated: bool=False)`** — JSON-serializable tool output with bounded passage text. (line 1096)
- **`_match_dict(result: SearchResult)`** — Copy source metadata for Gemini without sending oversized text. (line 1116)
- **`_unique_sources(sources: list[SearchResult])`** — Preserve first-seen source locations across multiple tool calls. (line 1128)
- **`_query_cache_key(query: object)`** — Normalize a search query so repeated Gemini calls can reuse local results. (line 1151)
- **`_safe_int(value: object, default: int)`** — Internal helper; see implementation. (line 1158)
- **`_call_name(call: object)`** — Internal helper; see implementation. (line 1167)
- **`_call_args(call: object)`** — Internal helper; see implementation. (line 1171)
- **`_clean_question(question: object)`** — Internal helper; see implementation. (line 1176)
- **`_truncate(text: str, limit: int)`** — Internal helper; see implementation. (line 1182)
- **`_timed_answer(answer: AgentAnswer, started: float)`** — Internal helper; see implementation. (line 1188)

### `app/tools.py`

- **`class SearchResult`** — One ranked search hit with the original source location copied through. (line 43)
- **`analyze_excel(chunks: list[DocumentChunk] | None, operation: str, column: str | None=None, threshold: object=None, comparison: str='greater_than', offset: int=0, limit: int=MAX_EXCEL_PAGE_SIZE, question_context: str | None=None)`** — Analyze typed Excel rows, using question context only for unambiguous columns. (line 55)
- **`search_documents(chunks: list[DocumentChunk] | None, query: str | None, max_results: int=DEFAULT_MAX_RESULTS)`** — Return the highest-scoring chunks that match *query*. (line 202)
- **`_coerce_max_results(max_results: object)`** — Turn *max_results* into an int, falling back to the default when invalid. (line 242)
- **`_excel_sheet_columns(chunks: list[DocumentChunk])`** — Internal helper; see implementation. (line 256)
- **`_numeric_columns(chunks: list[DocumentChunk])`** — Internal helper; see implementation. (line 277)
- **`_chunk_excel_values(chunk: DocumentChunk)`** — Internal helper; see implementation. (line 286)
- **`_resolve_column(requested: str, columns: list[str])`** — Internal helper; see implementation. (line 297)
- **`_edit_distance_at_most_one(left: str, right: str)`** — Match a column token with at most one insertion, deletion, or substitution. (line 337)
- **`_parse_number(value: object)`** — Internal helper; see implementation. (line 361)
- **`_passes_threshold(value: Decimal, threshold: Decimal | None, comparison: str)`** — Internal helper; see implementation. (line 416)
- **`_safe_int(value: object, default: int)`** — Internal helper; see implementation. (line 432)
- **`_number_for_json(value: Decimal | None)`** — Internal helper; see implementation. (line 441)
- **`_excel_record(record: tuple[DocumentChunk, str, Decimal, dict[str, object]])`** — Internal helper; see implementation. (line 449)
- **`_json_safe_cell(value: object)`** — Internal helper; see implementation. (line 466)
- **`_source_ranges(records: list[tuple[DocumentChunk, str, Decimal, dict[str, object]]])`** — Internal helper; see implementation. (line 474)
- **`_compress_rows(rows: set[int])`** — Internal helper; see implementation. (line 491)
- **`_tokenize(text: object)`** — Split text into lowercase alphanumeric tokens. (line 505)
- **`_normalize(text: str)`** — Lowercase text and replace punctuation/whitespace with single spaces. (line 513)
- **`_score_chunk(text: str, query_tokens: list[str])`** — Score how well *text* matches the query tokens. (line 519)
- **`_single_typo_match(token: str, candidates: Counter[str])`** — Return an unambiguous candidate one edit away from a likely misspelling. (line 552)
- **`_is_one_edit_apart(left: str, right: str)`** — Check for one insertion, deletion, or substitution without transpositions. (line 565)
- **`_has_phrase(chunk_tokens: list[str], query_tokens: list[str])`** — Return True when query tokens appear consecutively in the chunk. (line 588)
- **`_to_search_result(chunk: DocumentChunk, score: int)`** — Copy chunk text and source metadata into a ranked search hit. (line 599)

## Examples of intended questions

- PDF: "What is the pump pressure rating?" → `search_documents` → source-aware passages → Gemini answer with PDF page.
- Excel: "What is the maximum Amount SAR?" → `analyze_excel(operation="max", column="Amount SAR")` → Python `Decimal` max → Gemini response with source row(s).
- Excel: "Show rows where Amount SAR is greater than 100000" → `analyze_excel(operation="filter", column="Amount SAR", threshold="100000", comparison="greater_than")` → paginated matching rows.
- Ambiguous: "What is the maximum?" with multiple numeric columns → `list_columns`/clarification rather than inventing a metric.

## Known limitations / future improvements

- Local keyword search is not embedding-based semantic retrieval.
- Scanned/image-only PDFs without selectable text are not OCR-processed by the supplied loader.
- Excel header selection is heuristic and may fail for unusual sheets.
- The UI reads a local `documents/` folder; browser upload is **not** implemented in the supplied `main.py`.
- Local search fallback is not equivalent to an AI answer or a structured Excel calculation.
- No evidence was supplied here for the exact contents or coverage of the tests; do not claim specific tests exist without inspecting them.

## Scope and source accuracy

This reference was built from the user's supplied `app/agent.py`, `app/tools.py`, and pasted `app/main.py`, `app/document_loader.py`, `app/models.py`. If the live repository differs, the repository is authoritative. Additional project files (tests, requirements, README, deployment configuration) must be inspected separately before making claims about them.
