"""Shared data models for document chunks and agent answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools import SearchResult


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


@dataclass
class AgentAnswer:
    """Final answer with citations from executed document tools."""

    text: str
    sources: list[SearchResult] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float | None = None
    reference_labels: list[str] = field(default_factory=list)
    is_fallback: bool = False
    model: str | None = None

    def source_labels(self) -> list[str]:
        """Human-readable citations for UI display."""
        from app.agent import format_source

        labels = [format_source(source) for source in self.sources]
        return list(dict.fromkeys(labels + self.reference_labels))
