"""Tests for local keyword search over DocumentChunk objects."""

from __future__ import annotations

from app.document_loader import DocumentChunk
from app.tools import analyze_excel, search_documents


def _pdf(text: str, page: int, filename: str = "guide.pdf") -> DocumentChunk:
    return DocumentChunk(
        text=text,
        filename=filename,
        source_type="pdf",
        page_number=page,
    )


def _xlsx(
    text: str,
    sheet: str,
    row: int,
    filename: str = "inventory.xlsx",
    excel_values: dict[str, object] | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        filename=filename,
        source_type="xlsx",
        sheet_name=sheet,
        row_number=row,
        excel_values=excel_values,
    )


PDF_CHUNKS = [
    _pdf("Safety valves must be inspected yearly.", page=1),
    _pdf("The pump pressure rating is 150 PSI.", page=2),
    _pdf("Inspect filters after every shift.", page=3),
]

EXCEL_CHUNKS = [
    _xlsx("Name: Ada | Role: Engineer", sheet="People", row=2),
    _xlsx("Name: Grace | Role: Admiral", sheet="People", row=4),
    _xlsx("Item: Widget | Qty: 3", sheet="Inventory", row=2),
]


def test_pdf_match_keeps_filename_and_page_number() -> None:
    results = search_documents(PDF_CHUNKS, "pump pressure")

    assert len(results) == 1
    hit = results[0]
    assert hit.filename == "guide.pdf"
    assert hit.source_type == "pdf"
    assert hit.page_number == 2
    assert hit.sheet_name is None
    assert hit.row_number is None
    assert "pump pressure" in hit.text.lower()


def test_excel_match_keeps_filename_sheet_and_row() -> None:
    results = search_documents(EXCEL_CHUNKS, "Widget")

    assert len(results) == 1
    hit = results[0]
    assert hit.filename == "inventory.xlsx"
    assert hit.source_type == "xlsx"
    assert hit.sheet_name == "Inventory"
    assert hit.row_number == 2
    assert hit.page_number is None
    assert "Widget" in hit.text


def test_search_is_case_insensitive_and_ignores_punctuation() -> None:
    chunks = [_pdf("Coolant: check the pump-pressure, then restart.", page=4)]
    results = search_documents(chunks, "  PUMP Pressure!!  ")

    assert len(results) == 1
    assert results[0].page_number == 4


def test_search_corrects_single_character_misspellings() -> None:
    chunk = _pdf(
        "Project progress includes engineering, procurement, and construction milestones.",
        page=5,
    )

    for query in ("give me all progress", "give me all progres", "give me all prgress"):
        results = search_documents([chunk], query)
        assert len(results) == 1
        assert results[0].page_number == 5


def test_search_does_not_fuzzy_match_short_or_ambiguous_words() -> None:
    chunks = [_pdf("The project and projecy are both listed.", page=1)]

    assert search_documents(chunks, "cos") == []
    assert search_documents(chunks, "projecx") == []


def test_ranking_prefers_more_distinct_query_terms() -> None:
    chunks = [
        _pdf("The inspection checklist is on the wall.", page=1),
        _pdf("Replace the safety valve during inspection.", page=2),
        _pdf("The safety valve is closed.", page=3),
    ]

    results = search_documents(chunks, "safety valve inspection")

    assert [hit.page_number for hit in results] == [2, 3, 1]
    assert results[0].score > results[1].score > results[2].score


def test_no_matches_returns_empty_list() -> None:
    assert search_documents(PDF_CHUNKS + EXCEL_CHUNKS, "turbine") == []


def test_empty_chunks_or_query_returns_empty_list() -> None:
    assert search_documents([], "pump") == []
    assert search_documents(None, "pump") == []
    assert search_documents(PDF_CHUNKS, "") == []
    assert search_documents(PDF_CHUNKS, "   ???   ") == []
    assert search_documents(PDF_CHUNKS, None) == []


def test_invalid_max_results_is_handled_gracefully() -> None:
    chunks = PDF_CHUNKS + EXCEL_CHUNKS

    assert search_documents(chunks, "inspect", max_results=0) == []
    assert search_documents(chunks, "inspect", max_results=-2) == []
    fallback = search_documents(chunks, "inspect", max_results="not-a-number")
    assert fallback == search_documents(chunks, "inspect")
    limited = search_documents(chunks, "inspect", max_results=1)
    assert len(limited) == 1


def test_max_results_limits_output() -> None:
    chunks = [
        _pdf("cooling pump", page=1),
        _pdf("backup pump", page=2),
        _pdf("spare pump", page=3),
    ]
    results = search_documents(chunks, "pump", max_results=2)
    assert len(results) == 2


def test_source_metadata_is_copied_not_changed() -> None:
    original = _xlsx("Item: Widget | Qty: 3", sheet="Inventory", row=2)
    results = search_documents([original], "widget")

    assert original.filename == "inventory.xlsx"
    assert original.sheet_name == "Inventory"
    assert original.row_number == 2
    assert original.text == "Item: Widget | Qty: 3"
    assert results[0].filename == original.filename
    assert results[0].sheet_name == original.sheet_name
    assert results[0].row_number == original.row_number
    assert results[0].text == original.text


def test_excel_analysis_finds_maximum_with_column_variations() -> None:
    chunks = [
        _xlsx("Progress: 75%", "Schedule", 2, excel_values={"Progress (%)": "75%"}),
        _xlsx("Progress: 100%", "Schedule", 3, excel_values={"Progress (%)": 100}),
        _xlsx("Progress: 92%", "Schedule", 4, excel_values={"Progress (%)": "92 %"}),
    ]

    result = analyze_excel(chunks, operation="max", column="progress")

    assert result["value"] == 100
    assert result["row_count"] == 1
    assert result["rows"][0]["row_number"] == 3
    assert result["source_references"] == [
        "inventory.xlsx, sheet Schedule, rows 3"
    ]


def test_excel_analysis_supports_minimum_count_and_column_discovery() -> None:
    chunks = [
        _xlsx("Progress: 75%", "Schedule", 2, excel_values={"Progress (%)": "75%"}),
        _xlsx("Progress: 100%", "Schedule", 3, excel_values={"Progress (%)": 100}),
    ]

    minimum = analyze_excel(chunks, operation="min", column="progress")
    count = analyze_excel(chunks, operation="count", column="progress")
    columns = analyze_excel(chunks, operation="list_columns")

    assert minimum["value"] == 75
    assert count["value"] == 2
    assert columns["sheets"][0]["sheet_name"] == "Schedule"
    assert columns["sheets"][0]["columns"] == ["Progress (%)"]


def test_excel_analysis_sums_numeric_formats_and_references_all_rows() -> None:
    chunks = [
        _xlsx(
            "Cost: SAR 1,250,000",
            "BOQ",
            2,
            excel_values={"Unit Amount (SAR)": "SAR 1,250,000"},
        ),
        _xlsx(
            "Cost: 2,000,000 SAR",
            "BOQ",
            3,
            excel_values={"Unit Amount (SAR)": "2,000,000 SAR"},
        ),
        _xlsx(
            "Cost: (50,000)",
            "BOQ",
            4,
            excel_values={"Unit Amount (SAR)": "(50,000)"},
        ),
        _xlsx("Cost: n/a", "BOQ", 5, excel_values={"Unit Amount (SAR)": None}),
    ]

    result = analyze_excel(chunks, operation="sum", column="cost")

    assert result["value"] == 3_200_000
    assert result["row_count"] == 3
    assert result["source_references"] == [
        "inventory.xlsx, sheet BOQ, rows 2-4"
    ]


def test_excel_analysis_filters_numeric_thresholds_and_paginates() -> None:
    chunks = [
        _xlsx(
            f"Item: Cost item {row} | Cost: {amount}",
            "BOQ",
            row,
            excel_values={"Item": f"Cost item {row}", "Costs (SAR)": amount},
        )
        for row, amount in ((2, 999_999), (3, "1,250,000 SAR"), (4, 2_000_000))
    ]

    first_page = analyze_excel(
        chunks,
        operation="filter",
        column="cost",
        threshold="SAR 1 million",
        comparison="greater_than",
        limit=1,
    )
    second_page = analyze_excel(
        chunks,
        operation="filter",
        column="cost",
        threshold="SAR 1 million",
        comparison="greater_than",
        offset=first_page["next_offset"],
        limit=1,
    )

    assert first_page["row_count"] == 2
    assert first_page["rows"][0]["row_number"] == 3
    assert first_page["rows"][0]["fields"]["Item"] == "Cost item 3"
    assert first_page["next_offset"] == 1
    assert second_page["rows"][0]["row_number"] == 4
    assert second_page["next_offset"] is None
