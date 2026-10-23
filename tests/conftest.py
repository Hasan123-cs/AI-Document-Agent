"""Shared pytest fixtures for building tiny synthetic documents."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pymupdf
import pytest


@pytest.fixture
def empty_documents_dir(tmp_path: Path) -> Path:
    """An empty folder that mimics documents/ with no files."""
    folder = tmp_path / "documents"
    folder.mkdir()
    return folder


@pytest.fixture
def sample_pdf_path(tmp_path: Path) -> Path:
    """A two-page PDF with distinct text on each page."""
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page_one = document.new_page()
    page_one.insert_text((72, 72), "Alpha page content")
    page_two = document.new_page()
    page_two.insert_text((72, 72), "Beta page content")
    document.save(path)
    document.close()
    return path


@pytest.fixture
def blank_pdf_path(tmp_path: Path) -> Path:
    """A PDF that has a page but no extractable text."""
    path = tmp_path / "blank.pdf"
    document = pymupdf.open()
    document.new_page()
    document.save(path)
    document.close()
    return path


@pytest.fixture
def sample_xlsx_path(tmp_path: Path) -> Path:
    """An Excel workbook with two sheets and a blank row."""
    path = tmp_path / "sample.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"Name": "Ada", "Role": "Engineer"},
                {"Name": "", "Role": ""},
                {"Name": "Grace", "Role": "Admiral"},
            ]
        ).to_excel(writer, sheet_name="People", index=False)
        pd.DataFrame([{"Item": "Widget", "Qty": 3}]).to_excel(
            writer, sheet_name="Inventory", index=False
        )
    return path


@pytest.fixture
def xlsx_with_title_and_blank_header_path(tmp_path: Path) -> Path:
    """Workbook with a title row followed by meaningful data headers."""
    path = tmp_path / "unfamiliar.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                ["Monthly project controls", None, None],
                ["Activity", "Actual Progress %", "Budget (SAR)"],
                ["Design", 40, 1000],
                ["Construction", 100, 2500],
            ]
        ).to_excel(writer, sheet_name="Controls", index=False, header=False)
    return path


@pytest.fixture
def empty_xlsx_path(tmp_path: Path) -> Path:
    """An Excel workbook whose only sheet has headers and no data rows."""
    path = tmp_path / "empty.xlsx"
    pd.DataFrame(columns=["Name", "Role"]).to_excel(path, sheet_name="Empty", index=False)
    return path
