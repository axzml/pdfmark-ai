"""Tests for pdfmark_ai.models."""

import pytest
from pdfmark_ai.models import PageImage, Section, DocumentStructure, Chunk, ExtractionResult


class TestPageImage:
    def test_creation(self):
        img = PageImage(page_number=1, image_bytes=b"fake-png")
        assert img.page_number == 1
        assert img.image_bytes == b"fake-png"

    def test_frozen(self):
        img = PageImage(page_number=1, image_bytes=b"fake-png")
        with pytest.raises(AttributeError):
            img.page_number = 2

    def test_size_bytes(self):
        img = PageImage(page_number=1, image_bytes=b"\x89PNG\r\n")
        assert img.size_bytes == 6


class TestSection:
    def test_creation(self):
        s = Section(title="Introduction", start_page=1, end_page=5)
        assert s.title == "Introduction"
        assert s.start_page == 1
        assert s.end_page == 5


class TestDocumentStructure:
    def test_creation_with_sections(self):
        sections = [
            Section(title="Intro", start_page=1, end_page=5),
            Section(title="Methods", start_page=6, end_page=10),
        ]
        ds = DocumentStructure(
            sections=sections, doc_type="paper", language="en", source="outline"
        )
        assert len(ds.sections) == 2
        assert ds.source == "outline"

    def test_from_dict(self):
        data = {
            "sections": [
                {"title": "A", "start_page": 1, "end_page": 3},
                {"title": "B", "start_page": 4, "end_page": 5},
            ],
            "doc_type": "report",
            "language": "zh",
            "source": "llm_scan",
        }
        ds = DocumentStructure.model_validate(data)
        assert len(ds.sections) == 2
        assert ds.sections[1].title == "B"

    def test_sliding_window_fallback(self):
        ds = DocumentStructure(
            sections=[], doc_type="other", language="auto", source="sliding_window"
        )
        assert ds.source == "sliding_window"
        assert ds.sections == []


class TestChunk:
    def test_creation(self):
        pages = [PageImage(page_number=i, image_bytes=b"x") for i in range(1, 4)]
        chunk = Chunk(
            chunk_id=0, section_title="Intro",
            pages=pages, start_page=1, end_page=3,
        )
        assert chunk.chunk_id == 0
        assert len(chunk.pages) == 3
        assert chunk.context == ""

    def test_with_context(self):
        chunk = Chunk(
            chunk_id=1, section_title="Methods",
            pages=[], start_page=4, end_page=5,
            context="previous chunk tail...",
        )
        assert chunk.context == "previous chunk tail..."


class TestExtractionResult:
    def test_creation(self):
        r = ExtractionResult(
            chunk_id=0, start_page=1, end_page=3,
            section_title="Intro", markdown="# Hello\n\nWorld",
            tail_summary="World",
        )
        assert r.markdown == "# Hello\n\nWorld"
        assert r.tail_summary == "World"

    def test_is_empty_true(self):
        r = ExtractionResult(
            chunk_id=0, start_page=1, end_page=1,
            section_title="", markdown="  \n\n  ", tail_summary="",
        )
        assert r.is_empty

    def test_is_empty_false(self):
        r = ExtractionResult(
            chunk_id=0, start_page=1, end_page=1,
            section_title="", markdown="content", tail_summary="",
        )
        assert not r.is_empty
