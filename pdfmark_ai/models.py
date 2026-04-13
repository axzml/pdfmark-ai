"""Data models for the PDF parser pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pydantic import BaseModel


@dataclass(frozen=True)
class PageImage:
    """A single PDF page rendered as a PNG image."""

    page_number: int  # 1-indexed
    image_bytes: bytes  # PNG raw bytes

    @property
    def size_bytes(self) -> int:
        return len(self.image_bytes)


@dataclass
class Section:
    """A logical section of the document."""

    title: str
    start_page: int  # inclusive, 1-indexed
    end_page: int  # inclusive, 1-indexed


class DocumentStructure(BaseModel):
    """Document structure detected from TOC, LLM scan, or sliding window fallback."""

    sections: list[Section]
    doc_type: str  # "paper", "book", "report", "slides", "other"
    language: str  # "zh", "en", "auto"
    source: str  # "outline", "llm_scan", "sliding_window"


class Chunk:
    """A unit of work for the LLM extraction stage."""

    def __init__(
        self,
        chunk_id: int,
        section_title: str,
        pages: list[PageImage],
        start_page: int,
        end_page: int,
        context: str = "",
    ):
        self.chunk_id = chunk_id
        self.section_title = section_title
        self.pages = pages
        self.start_page = start_page
        self.end_page = end_page
        self.context = context


class ExtractionResult:
    """Result of extracting a single chunk via LLM."""

    def __init__(
        self,
        chunk_id: int,
        start_page: int,
        end_page: int,
        section_title: str,
        markdown: str,
        tail_summary: str = "",
    ):
        self.chunk_id = chunk_id
        self.start_page = start_page
        self.end_page = end_page
        self.section_title = section_title
        self.markdown = markdown
        self.tail_summary = tail_summary

    @property
    def is_empty(self) -> bool:
        return not self.markdown.strip()
