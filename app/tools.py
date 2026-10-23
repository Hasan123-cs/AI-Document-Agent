"""Local document search tools used by the agent.

Matching is deterministic keyword search over already-extracted
DocumentChunk text. This module does not call Gemini or load files.
"""

from __future__ import annotations

import logging
import re
import time
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.document_loader import DocumentChunk

logger = logging.getLogger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
DEFAULT_MAX_RESULTS = 5
MAX_EXCEL_PAGE_SIZE = 50
_CURRENCY_MARKERS = re.compile(r"(?i)\b(?:sar|sr|usd|eur|gbp|aed|qar|kwd|bhd|omr)\b")
_COLUMN_ALIASES = {
    "amount": {"amount", "amounts", "cost", "costs", "price", "prices", "value", "total"},
    "cost": {"cost", "costs", "amount", "amounts", "price", "prices", "value", "total"},
    "price": {"price", "prices", "cost", "costs", "amount", "amounts", "value", "rate"},
    "progress": {
        "progress",
        "completion",
        "complete",
        "completed",
        "percent",
        "percentage",
        "progresspercent",
    },
    "quantity": {"quantity", "quantities", "qty", "units"},
}


@dataclass(frozen=True)
class SearchResult:
    """One ranked search hit with the original source location copied through."""

    text: str
    filename: str
    source_type: str
    page_number: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    score: int = 0


def analyze_excel(
    chunks: list[DocumentChunk] | None,
    operation: str,
    column: str | None = None,
    threshold: object = None,
    comparison: str = "greater_than",
    offset: int = 0,
    limit: int = MAX_EXCEL_PAGE_SIZE,
    question_context: str | None = None,
) -> dict[str, Any]:
    """Analyze typed Excel rows, using question context only for unambiguous columns."""
    excel_chunks = [
        chunk
        for chunk in (chunks or [])
        if isinstance(chunk, DocumentChunk) and chunk.source_type == "xlsx"
    ]
    sheets = _excel_sheet_columns(excel_chunks)
    if operation == "list_columns":
        return {"sheets": sheets}
    if operation not in {"filter", "max", "min", "sum", "count"}:
        return {"error": f"Unsupported Excel analysis operation: {operation}"}

    selected_column = column.strip() if isinstance(column, str) else ""
    threshold_value = _parse_number(threshold) if threshold is not None else None
    if (
        operation != "filter"
        and not selected_column
        and isinstance(threshold, str)
        and threshold_value is None
    ):
        selected_column = _resolve_column(
            threshold,
            sorted(
                {
                    column_name
                    for sheet in sheets
                    for column_name in sheet["columns"]
                }
            ),
        ) or ""
        if selected_column:
            threshold = None

    threshold_value = _parse_number(threshold) if threshold is not None else None
    if threshold is not None and threshold_value is None:
        return {
            "error": "The supplied numeric threshold is invalid.",
            "available_sheets": sheets,
        }
    if operation == "filter" and threshold_value is None:
        return {"error": "A numeric threshold is required for filtering."}

    if not selected_column:
        numeric_columns = _numeric_columns(excel_chunks)
        if question_context:
            selected_column = _resolve_column(question_context, numeric_columns) or ""
        if len(numeric_columns) == 1:
            selected_column = selected_column or numeric_columns[0]
        elif not selected_column:
            return {
                "error": (
                    "More than one numeric column is available; ask the user "
                    "which column they mean."
                    if numeric_columns
                    else "No numeric Excel columns are available to analyze."
                ),
                "numeric_columns": numeric_columns,
                "available_sheets": sheets,
            }

    if comparison not in {
        "greater_than",
        "greater_than_or_equal",
        "less_than",
        "less_than_or_equal",
        "equal",
    }:
        return {"error": f"Unsupported numeric comparison: {comparison}"}

    records: list[tuple[DocumentChunk, str, Decimal, dict[str, object]]] = []
    matched_sheets: set[tuple[str, str | None]] = set()
    for chunk in excel_chunks:
        values = _chunk_excel_values(chunk)
        actual_column = _resolve_column(selected_column, list(values))
        if actual_column is None:
            continue
        numeric_value = _parse_number(values.get(actual_column))
        if numeric_value is None or not _passes_threshold(
            numeric_value, threshold_value, comparison
        ):
            continue
        records.append((chunk, actual_column, numeric_value, values))
        matched_sheets.add((chunk.filename, chunk.sheet_name))

    if not records:
        return {
            "operation": operation,
            "column": selected_column,
            "row_count": 0,
            "message": "No numeric values matched the selected column and criteria.",
            "available_sheets": sheets,
        }

    if operation in {"max", "min"}:
        target = (max if operation == "max" else min)(record[2] for record in records)
        selected_records = [record for record in records if record[2] == target]
        result: dict[str, Any] = {
            "operation": operation,
            "column": selected_column,
            "value": _number_for_json(target),
            "row_count": len(selected_records),
            "rows": [_excel_record(record) for record in selected_records[:MAX_EXCEL_PAGE_SIZE]],
            "truncated": len(selected_records) > MAX_EXCEL_PAGE_SIZE,
        }
        source_records = selected_records
    elif operation == "filter":
        safe_limit = max(1, min(_safe_int(limit, MAX_EXCEL_PAGE_SIZE), MAX_EXCEL_PAGE_SIZE))
        safe_offset = max(0, _safe_int(offset, 0))
        page = records[safe_offset : safe_offset + safe_limit]
        result = {
            "operation": operation,
            "column": selected_column,
            "comparison": comparison,
            "threshold": _number_for_json(threshold_value),
            "row_count": len(records),
            "offset": safe_offset,
            "rows": [_excel_record(record) for record in page],
            "next_offset": safe_offset + len(page) if safe_offset + len(page) < len(records) else None,
        }
        source_records = records
    else:
        total = sum((record[2] for record in records), Decimal(0))
        result = {
            "operation": operation,
            "column": selected_column,
            "value": len(records) if operation == "count" else _number_for_json(total),
            "row_count": len(records),
        }
        source_records = records

    result["source_references"] = _source_ranges(source_records)
    result["sheets"] = sorted(
        {f"{filename}, sheet {sheet_name}" for filename, sheet_name in matched_sheets}
    )
    return result


def search_documents(
    chunks: list[DocumentChunk] | None,
    query: str | None,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> list[SearchResult]:
    """Return the highest-scoring chunks that match *query*.

    Matching is case-insensitive. Punctuation and extra whitespace are
    ignored. Results keep the original chunk text and source metadata.
    """
    started = time.perf_counter()
    limit = _coerce_max_results(max_results)
    if limit <= 0:
        return []

    query_tokens = _tokenize(query)
    if not query_tokens or not chunks:
        return []

    scored: list[tuple[int, int, SearchResult]] = []
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, DocumentChunk):
            continue
        score = _score_chunk(chunk.text, query_tokens)
        if score <= 0:
            continue
        # append to scored with score, index, and SearchResult to maintain stable sorting
        scored.append((score, index, _to_search_result(chunk, score)))
    # sort by score descending, then by original index ascending to maintain stable order
    scored.sort(key=lambda item: (-item[0], item[1]))
    results = [result for _, _, result in scored[:limit]]
    logger.info(
        "timing.search_documents chunks=%s hits=%s seconds=%.4f",
        len(chunks),
        len(results),
        time.perf_counter() - started,
    )
    return results


def _coerce_max_results(max_results: object) -> int:
    """Turn *max_results* into an int, falling back to the default when invalid."""
    if isinstance(max_results, bool) or max_results is None:
        return DEFAULT_MAX_RESULTS
    if isinstance(max_results, int):
        return max_results
    if isinstance(max_results, float) and max_results.is_integer():
        return int(max_results)
    try:
        return int(str(max_results).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_RESULTS


def _excel_sheet_columns(chunks: list[DocumentChunk]) -> list[dict[str, object]]:
    sheets: dict[tuple[str, str | None], set[str]] = {}
    row_counts: Counter[tuple[str, str | None]] = Counter()
    for chunk in chunks:
        key = (chunk.filename, chunk.sheet_name)
        values = _chunk_excel_values(chunk)
        sheets.setdefault(key, set()).update(values)
        row_counts[key] += 1
    return [
        {
            "filename": filename,
            "sheet_name": sheet_name,
            "columns": sorted(columns),
            "row_count": row_counts[(filename, sheet_name)],
        }
        for (filename, sheet_name), columns in sorted(
            sheets.items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
    ]


def _numeric_columns(chunks: list[DocumentChunk]) -> list[str]:
    columns: set[str] = set()
    for chunk in chunks:
        for column, value in _chunk_excel_values(chunk).items():
            if _parse_number(value) is not None:
                columns.add(column)
    return sorted(columns)


def _chunk_excel_values(chunk: DocumentChunk) -> dict[str, object]:
    if chunk.excel_values is not None:
        return chunk.excel_values
    parsed: dict[str, object] = {}
    for field in chunk.text.split("|"):
        column, separator, value = field.partition(":")
        if separator and column.strip():
            parsed[column.strip()] = value.strip()
    return parsed


def _resolve_column(requested: str, columns: list[str]) -> str | None:
    requested_key = _normalize(requested).replace(" ", "")
    if not requested_key:
        return None
    exact = [
        column
        for column in columns
        if _normalize(column).replace(" ", "") == requested_key
    ]
    if len(exact) == 1:
        return exact[0]
    if exact:
        return None

    terms = set(_normalize(requested).split())
    aliases = set().union(
        *(
            values
            for key, values in _COLUMN_ALIASES.items()
            if key in terms or terms.intersection(values)
        )
    ) if terms else set()
    if not aliases:
        aliases = terms
    candidates: list[str] = []
    for column in columns:
        column_terms = set(_normalize(column).split())
        direct_match = bool(terms.intersection(column_terms))
        alias_match = bool(aliases.intersection(column_terms))
        typo_match = any(
            _edit_distance_at_most_one(request_term, column_term)
            for request_term in terms
            if len(request_term) >= 5
            for column_term in column_terms
        )
        if direct_match or alias_match or typo_match:
            candidates.append(column)
    return candidates[0] if len(candidates) == 1 else None


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    """Match a column token with at most one insertion, deletion, or substitution."""
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False

    shorter, longer = sorted((left, right), key=len)
    short_index = long_index = edits = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(shorter) == len(longer):
            short_index += 1
        long_index += 1
    edits += len(longer) - long_index
    return edits <= 1


def _parse_number(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    if not isinstance(value, str):
        return None

    text = _CURRENCY_MARKERS.sub("", value).strip()
    if not text:
        return None
    lower_text = text.lower()
    multiplier = Decimal(1)
    for markers, scale in (
        (r"\b(?:billion|bn)\b|(?<=\d)\s*b\b", Decimal(1_000_000_000)),
        (r"\b(?:million|mn)\b|(?<=\d)\s*m\b", Decimal(1_000_000)),
        (r"\b(?:thousand|k)\b|(?<=\d)\s*k\b", Decimal(1_000)),
    ):
        if re.search(markers, lower_text):
            multiplier = scale
            break
    negative = "(" in text and ")" in text
    text = re.sub(r"[^\d,.\-+]", "", text)
    if not text or not re.search(r"\d", text):
        return None

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        decimal_part = text.rsplit(",", 1)[-1]
        if len(decimal_part) in (1, 2):
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")
    elif text.count(".") > 1:
        parts = text.split(".")
        text = "".join(parts[:-1]) + "." + parts[-1]

    try:
        number = Decimal(text)
    except InvalidOperation:
        return None
    if negative:
        number = -abs(number)
    number *= multiplier
    return number if number.is_finite() else None


def _passes_threshold(
    value: Decimal,
    threshold: Decimal | None,
    comparison: str,
) -> bool:
    if threshold is None:
        return True
    return {
        "greater_than": value > threshold,
        "greater_than_or_equal": value >= threshold,
        "less_than": value < threshold,
        "less_than_or_equal": value <= threshold,
        "equal": value == threshold,
    }[comparison]


def _safe_int(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _number_for_json(value: Decimal | None) -> int | str | None:
    if value is None:
        return None
    if value == value.to_integral_value():
        return int(value)
    return str(value)


def _excel_record(
    record: tuple[DocumentChunk, str, Decimal, dict[str, object]],
) -> dict[str, object]:
    chunk, column, value, fields = record
    return {
        "filename": chunk.filename,
        "sheet_name": chunk.sheet_name,
        "row_number": chunk.row_number,
        "column": column,
        "value": _number_for_json(value),
        "fields": {
            key: _json_safe_cell(cell)
            for key, cell in list(fields.items())[:20]
        },
    }


def _json_safe_cell(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value if not isinstance(value, str) else value[:240]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)[:240]


def _source_ranges(
    records: list[tuple[DocumentChunk, str, Decimal, dict[str, object]]],
) -> list[str]:
    grouped: dict[tuple[str, str | None], set[int]] = {}
    for chunk, _, _, _ in records:
        if chunk.row_number is not None:
            grouped.setdefault((chunk.filename, chunk.sheet_name), set()).add(
                chunk.row_number
            )
    return [
        f"{filename}, sheet {sheet_name}, rows {_compress_rows(rows)}"
        for (filename, sheet_name), rows in sorted(
            grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")
        )
    ]


def _compress_rows(rows: set[int]) -> str:
    ordered = sorted(rows)
    ranges: list[str] = []
    start = previous = ordered[0]
    for row in ordered[1:]:
        if row == previous + 1:
            previous = row
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = row
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(ranges)


def _tokenize(text: object) -> list[str]:
    """Split text into lowercase alphanumeric tokens."""
    if not isinstance(text, str):
        return []
    normalized = _normalize(text)
    return normalized.split() if normalized else []


def _normalize(text: str) -> str:
    """Lowercase text and replace punctuation/whitespace with single spaces."""
    collapsed = _NON_ALNUM.sub(" ", text.lower())
    return collapsed.strip()


def _score_chunk(text: str, query_tokens: list[str]) -> int:
    """Score how well *text* matches the query tokens.

    The score is explainable and ordered by importance:
    1. Distinct query tokens found (+100 each) so broader coverage ranks first.
    2. One-character spelling corrections for words of at least five characters.
    3. Phrase bonus (+20) when the full normalized query appears in order.
    4. Raw term frequency (+1 per occurrence) as a tie-breaker.
    """
    chunk_tokens = _tokenize(text)
    if not chunk_tokens:
        return 0

    counts = Counter(chunk_tokens)
    unique_matches = 0
    term_frequency = 0
    for token in dict.fromkeys(query_tokens):
        occurrences = counts.get(token, 0)
        if not occurrences:
            corrected = _single_typo_match(token, counts)
            if corrected is not None:
                occurrences = counts[corrected]
        if occurrences:
            unique_matches += 1
            term_frequency += occurrences

    if unique_matches == 0:
        return 0

    phrase_bonus = 20 if _has_phrase(chunk_tokens, query_tokens) else 0
    return unique_matches * 100 + phrase_bonus + term_frequency


def _single_typo_match(token: str, candidates: Counter[str]) -> str | None:
    """Return an unambiguous candidate one edit away from a likely misspelling."""
    if len(token) < 5:
        return None

    matches = [
        candidate
        for candidate in candidates
        if _is_one_edit_apart(token, candidate)
    ]
    return matches[0] if len(matches) == 1 else None


def _is_one_edit_apart(left: str, right: str) -> bool:
    """Check for one insertion, deletion, or substitution without transpositions."""
    if abs(len(left) - len(right)) > 1 or left == right:
        return False

    shorter, longer = sorted((left, right), key=len)
    short_index = long_index = edits = 0
    while short_index < len(shorter) and long_index < len(longer):
        if shorter[short_index] == longer[long_index]:
            short_index += 1
            long_index += 1
            continue
        edits += 1
        if edits > 1:
            return False
        if len(shorter) == len(longer):
            short_index += 1
        long_index += 1

    edits += len(longer) - long_index
    return edits == 1


def _has_phrase(chunk_tokens: list[str], query_tokens: list[str]) -> bool:
    """Return True when query tokens appear consecutively in the chunk."""
    query_len = len(query_tokens)
    if query_len == 0 or query_len > len(chunk_tokens):
        return False
    for start in range(len(chunk_tokens) - query_len + 1):
        if chunk_tokens[start : start + query_len] == query_tokens:
            return True
    return False


def _to_search_result(chunk: DocumentChunk, score: int) -> SearchResult:
    """Copy chunk text and source metadata into a ranked search hit."""
    return SearchResult(
        text=chunk.text,
        filename=chunk.filename,
        source_type=chunk.source_type,
        page_number=chunk.page_number,
        sheet_name=chunk.sheet_name,
        row_number=chunk.row_number,
        score=score,
    )
