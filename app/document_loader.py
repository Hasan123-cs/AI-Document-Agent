"""Load PDF and Excel files from a local documents folder.

Extracted text is returned as chunks with source references so later
retrieval steps can cite filename, page, sheet, and row. Nothing in this
module calls Gemini or any other remote API.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import pymupdf

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".xlsx"}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "documents"


@dataclass(frozen=True)
class DocumentChunk:
    """A unit of extracted text plus the location it came from."""

    text: str
    filename: str
    source_type: str
    page_number: int | None = None
    sheet_name: str | None = None
    row_number: int | None = None
    excel_values: dict[str, object] | None = None


def default_documents_dir() -> Path:
    """Return the project-level documents/ directory."""
    return DEFAULT_DOCUMENTS_DIR


def load_documents(directory: str | Path | None = None) -> list[DocumentChunk]:
    """Read all supported files in *directory* and return text chunks.

    Missing or empty folders yield an empty list. Unsupported files are
    skipped. A corrupt or unreadable file is logged and processing
    continues with the remaining files.
    """
    folder = Path(directory) if directory is not None else default_documents_dir()
    started = time.perf_counter()

    if not folder.exists():
        logger.warning("Documents folder does not exist: %s", folder)
        return []

    if not folder.is_dir():
        logger.warning("Documents path is not a directory: %s", folder)
        return []

    chunks: list[DocumentChunk] = []
    loaded_files = 0
    for path in _iter_files(folder):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.info("Skipping unsupported file: %s", path.name)
            continue

        try:
            file_chunks = _load_file(path)
        except Exception:
            logger.exception("Failed to load %s; continuing with remaining files", path.name)
            continue

        if not file_chunks:
            logger.info("No extractable content in %s", path.name)
            continue

        loaded_files += 1
        chunks.extend(file_chunks)

    logger.info(
        "timing.load_documents files=%s chunks=%s seconds=%.3f",
        loaded_files,
        len(chunks),
        time.perf_counter() - started,
    )
    return chunks


def load_pdf(path: str | Path) -> list[DocumentChunk]:
    """Extract non-empty text from each page of a PDF using PyMuPDF."""
    pdf_path = Path(path)
    chunks: list[DocumentChunk] = []

    with pymupdf.open(pdf_path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            text = (page.get_text("text") or "").strip()
            if not text:
                continue
            chunks.append(
                DocumentChunk(
                    text=text,
                    filename=pdf_path.name,
                    source_type="pdf",
                    page_number=page_index + 1,
                )
            )

    return chunks


def load_xlsx(path: str | Path) -> list[DocumentChunk]:
    """Extract non-empty rows from every sheet of an Excel workbook."""
    xlsx_path = Path(path)
    sheets = pd.read_excel(
        xlsx_path,
        sheet_name=None,
        engine="openpyxl",
        header=None,
    )
    chunks: list[DocumentChunk] = []

    for sheet_name, raw_frame in sheets.items():
        header_row = _find_header_row(raw_frame)
        frame = _frame_with_headers(raw_frame, header_row)
        chunks.extend(
            _chunks_from_sheet(
                xlsx_path.name,
                str(sheet_name),
                frame,
                row_number_offset=header_row + 2,
            )
        )

    return chunks


def _load_file(path: Path) -> list[DocumentChunk]:
    """Dispatch a supported file to the matching loader."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path)
    if suffix == ".xlsx":
        return load_xlsx(path)
    return []


def _iter_files(folder: Path) -> Iterable[Path]:
    """Yield files in *folder* in a stable order, ignoring subdirectories."""
    return sorted(path for path in folder.iterdir() if path.is_file())


def _chunks_from_sheet(
    filename: str,
    sheet_name: str,
    frame: pd.DataFrame,
    row_number_offset: int = 2,
) -> list[DocumentChunk]:
    """Turn a worksheet into row chunks with original Excel row references."""
    if frame.empty:
        return []

    chunks: list[DocumentChunk] = []
    for pandas_index, row in frame.iterrows():
        text = _format_excel_row(row)
        if not text:
            continue
        chunks.append(
            DocumentChunk(
                text=text,
                filename=filename,
                source_type="xlsx",
                sheet_name=sheet_name,
                row_number=_excel_row_number(pandas_index, row_number_offset),
                excel_values=_excel_values(row),
            )
        )
    return chunks


def _find_header_row(frame: pd.DataFrame, scan_rows: int = 10) -> int:
    """Find the most header-like row near the top of a worksheet."""
    if frame.empty:
        return 0
    candidates = range(min(len(frame), scan_rows))
    return max(
        candidates,
        key=lambda row_index: sum(
            not _is_blank(value) for value in frame.iloc[row_index].tolist()
        ),
    )


def _frame_with_headers(frame: pd.DataFrame, header_row: int) -> pd.DataFrame:
    """Use the detected header row and retain only subsequent data rows."""
    if frame.empty or header_row >= len(frame) - 1:
        return pd.DataFrame()

    headers: list[str] = []
    used: set[str] = set()
    for index, value in enumerate(frame.iloc[header_row].tolist(), start=1):
        header = "" if _is_blank(value) else str(value).strip()
        if not header or header.lower().startswith("unnamed:"):
            header = f"Column {index}"
        base = header
        suffix = 2
        while header in used:
            header = f"{base} ({suffix})"
            suffix += 1
        used.add(header)
        headers.append(header)

    data = frame.iloc[header_row + 1 :].copy().reset_index(drop=True)
    data.columns = headers
    return data


def _excel_row_number(pandas_index: object, offset: int = 2) -> int:
    """Map a zero-based data index to its original 1-based Excel row."""
    try:
        return int(pandas_index) + offset
    except (TypeError, ValueError):
        return offset


def _format_excel_row(row: pd.Series) -> str:
    """Serialize one spreadsheet row as 'column: value' pairs."""
    parts: list[str] = []
    for column, value in row.items():
        if _is_blank(value):
            continue
        parts.append(f"{column}: {value}")
    return " | ".join(parts)


def _excel_values(row: pd.Series) -> dict[str, object]:
    """Keep typed worksheet cells for accurate local numeric analysis."""
    values: dict[str, object] = {}
    for column, value in row.items():
        key = str(column).strip()
        if not key:
            continue
        if _is_blank(value):
            values[key] = None
        elif hasattr(value, "item"):
            values[key] = value.item()
        elif hasattr(value, "isoformat"):
            values[key] = value.isoformat()
        else:
            values[key] = value
    return values


def _is_blank(value: object) -> bool:
    """Return True when a cell should not be included in extracted text."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""
