"""Tests for local PDF and Excel loading without calling Gemini."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.document_loader import (
    MAX_UPLOAD_FILE_BYTES,
    load_documents,
    load_pdf,
    load_uploaded_document,
    load_xlsx,
)
from app.models import DocumentChunk


def test_empty_folder_returns_no_chunks(empty_documents_dir: Path) -> None:
    assert load_documents(empty_documents_dir) == []


def test_missing_folder_returns_no_chunks(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    assert load_documents(missing) == []


def test_pdf_preserves_filename_and_page_numbers(sample_pdf_path: Path) -> None:
    chunks = load_pdf(sample_pdf_path)

    assert len(chunks) == 2
    assert chunks[0] == DocumentChunk(
        text="Alpha page content",
        filename="sample.pdf",
        source_type="pdf",
        page_number=1,
    )
    assert chunks[1].page_number == 2
    assert chunks[1].text == "Beta page content"
    assert chunks[1].filename == "sample.pdf"


def test_blank_pdf_yields_no_chunks(blank_pdf_path: Path) -> None:
    assert load_pdf(blank_pdf_path) == []


def test_xlsx_preserves_filename_sheet_and_row(sample_xlsx_path: Path) -> None:
    chunks = load_xlsx(sample_xlsx_path)

    people_rows = [chunk for chunk in chunks if chunk.sheet_name == "People"]
    inventory_rows = [chunk for chunk in chunks if chunk.sheet_name == "Inventory"]

    assert {chunk.filename for chunk in chunks} == {"sample.xlsx"}
    assert {chunk.source_type for chunk in chunks} == {"xlsx"}
    assert len(people_rows) == 2
    assert people_rows[0].row_number == 2
    assert "Ada" in people_rows[0].text
    assert people_rows[0].excel_values == {"Name": "Ada", "Role": "Engineer"}
    assert people_rows[1].row_number == 4
    assert "Grace" in people_rows[1].text
    assert len(inventory_rows) == 1
    assert inventory_rows[0].row_number == 2
    assert "Widget" in inventory_rows[0].text


def test_xlsx_detects_header_after_title_and_preserves_original_rows(
    xlsx_with_title_and_blank_header_path: Path,
) -> None:
    chunks = load_xlsx(xlsx_with_title_and_blank_header_path)

    assert [chunk.row_number for chunk in chunks] == [3, 4]
    assert chunks[0].excel_values == {
        "Activity": "Design",
        "Actual Progress %": 40,
        "Budget (SAR)": 1000,
    }
    assert chunks[1].excel_values["Activity"] == "Construction"


def test_empty_xlsx_yields_no_chunks(empty_xlsx_path: Path) -> None:
    assert load_xlsx(empty_xlsx_path) == []


def test_unsupported_files_are_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "notes.txt").write_text("ignore me", encoding="utf-8")
    (folder / "legacy.xls").write_bytes(b"not-a-real-xls")

    assert load_documents(folder) == []


def test_corrupt_file_does_not_stop_other_files(
    tmp_path: Path, sample_pdf_path: Path, sample_xlsx_path: Path
) -> None:
    folder = tmp_path / "mixed"
    folder.mkdir()
    (folder / "broken.pdf").write_bytes(b"this is not a pdf")
    (folder / "broken.xlsx").write_bytes(b"this is not excel")
    (folder / sample_pdf_path.name).write_bytes(sample_pdf_path.read_bytes())
    (folder / sample_xlsx_path.name).write_bytes(sample_xlsx_path.read_bytes())

    chunks = load_documents(folder)
    filenames = {chunk.filename for chunk in chunks}

    assert "sample.pdf" in filenames
    assert "sample.xlsx" in filenames
    assert "broken.pdf" not in filenames
    assert "broken.xlsx" not in filenames


def test_load_documents_reads_pdf_and_xlsx_from_folder(
    tmp_path: Path, sample_pdf_path: Path, sample_xlsx_path: Path
) -> None:
    folder = tmp_path / "documents"
    folder.mkdir()
    (folder / sample_pdf_path.name).write_bytes(sample_pdf_path.read_bytes())
    (folder / sample_xlsx_path.name).write_bytes(sample_xlsx_path.read_bytes())

    chunks = load_documents(folder)

    assert any(chunk.source_type == "pdf" for chunk in chunks)
    assert any(chunk.source_type == "xlsx" for chunk in chunks)
    assert all(chunk.text.strip() for chunk in chunks)


def test_uploaded_pdf_preserves_filename_and_page_references(
    sample_pdf_path: Path,
) -> None:
    chunks = load_uploaded_document(sample_pdf_path.name, sample_pdf_path.read_bytes())

    assert [chunk.page_number for chunk in chunks] == [1, 2]
    assert {chunk.filename for chunk in chunks} == {"sample.pdf"}
    assert chunks[0].text == "Alpha page content"


def test_uploaded_excel_preserves_sheet_and_row_references(
    sample_xlsx_path: Path,
) -> None:
    chunks = load_uploaded_document(
        sample_xlsx_path.name,
        sample_xlsx_path.read_bytes(),
    )

    inventory = [chunk for chunk in chunks if chunk.sheet_name == "Inventory"]
    assert len(inventory) == 1
    assert inventory[0].filename == "sample.xlsx"
    assert inventory[0].row_number == 2
    assert inventory[0].excel_values == {"Item": "Widget", "Qty": 3}


def test_multiple_uploads_can_be_processed_together(
    sample_pdf_path: Path,
    sample_xlsx_path: Path,
) -> None:
    chunks = [
        *load_uploaded_document(sample_pdf_path.name, sample_pdf_path.read_bytes()),
        *load_uploaded_document(sample_xlsx_path.name, sample_xlsx_path.read_bytes()),
    ]

    assert {chunk.filename for chunk in chunks} == {"sample.pdf", "sample.xlsx"}
    assert any(chunk.page_number == 2 for chunk in chunks if chunk.source_type == "pdf")
    assert any(
        chunk.sheet_name == "Inventory" and chunk.row_number == 2
        for chunk in chunks
        if chunk.source_type == "xlsx"
    )


def test_uploaded_invalid_extension_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_uploaded_document("notes.txt", b"not a document")


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("bad.pdf", b"%PDF- but invalid"),
        ("bad.xlsx", b"not a zip workbook"),
        ("bad.xls", bytes.fromhex("D0CF11E0A1B11AE1") + b"not an xls workbook"),
    ],
)
def test_uploaded_corrupt_documents_are_rejected(
    filename: str,
    content: bytes,
) -> None:
    with pytest.raises(ValueError):
        load_uploaded_document(filename, content)


def test_oversized_upload_is_rejected_before_parsing() -> None:
    with pytest.raises(ValueError, match="per-file limit"):
        load_uploaded_document(
            "large.pdf",
            b"%PDF-" + b"0" * MAX_UPLOAD_FILE_BYTES,
        )


def test_uploaded_filename_is_sanitized(sample_pdf_path: Path) -> None:
    chunks = load_uploaded_document(
        r"..\..\sample.pdf",
        sample_pdf_path.read_bytes(),
    )
    assert chunks[0].filename == "sample.pdf"
