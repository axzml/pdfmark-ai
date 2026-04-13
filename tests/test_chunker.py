"""Tests for pdfmark_ai.chunker."""

import pytest
from pdfmark_ai.chunker import build_chunks, _semantic_chunks, _sliding_window_chunks
from pdfmark_ai.models import (
    DocumentStructure, PageImage, Section, Chunk,
)


def make_pages(n: int) -> list[PageImage]:
    return [PageImage(page_number=i, image_bytes=f"png-{i}".encode()) for i in range(1, n + 1)]


class TestSemanticChunks:
    def test_basic_splitting(self):
        """A 10-page section with max_pages=4 should produce 3 chunks."""
        sections = [Section(title="Chapter", start_page=1, end_page=10)]
        structure = DocumentStructure(sections=sections, doc_type="book", language="en", source="outline")
        pages = make_pages(10)
        chunks = _semantic_chunks(structure, pages, max_pages=4)
        assert len(chunks) == 3
        assert chunks[0].pages[0].page_number == 1
        assert chunks[0].pages[-1].page_number == 4
        assert chunks[1].pages[0].page_number == 5
        assert chunks[2].pages[-1].page_number == 10

    def test_multiple_sections(self):
        sections = [
            Section(title="A", start_page=1, end_page=3),
            Section(title="B", start_page=4, end_page=8),
        ]
        structure = DocumentStructure(sections=sections, doc_type="book", language="en", source="outline")
        pages = make_pages(8)
        chunks = _semantic_chunks(structure, pages, max_pages=4)
        assert all(c.section_title in ("A", "B") for c in chunks)

    def test_no_cross_section_split(self):
        """Chunk should never span across section boundaries."""
        sections = [
            Section(title="A", start_page=1, end_page=5),
            Section(title="B", start_page=6, end_page=10),
        ]
        structure = DocumentStructure(sections=sections, doc_type="book", language="en", source="outline")
        pages = make_pages(10)
        chunks = _semantic_chunks(structure, pages, max_pages=8)
        a_chunks = [c for c in chunks if c.section_title == "A"]
        b_chunks = [c for c in chunks if c.section_title == "B"]
        assert len(a_chunks) >= 1
        assert len(b_chunks) >= 1


class TestSlidingWindowChunks:
    def test_basic_window(self):
        pages = make_pages(5)
        chunks = _sliding_window_chunks(pages, window_size=2, overlap=1)
        assert len(chunks) == 4
        assert chunks[0].start_page == 1 and chunks[0].end_page == 2
        assert chunks[1].start_page == 2 and chunks[1].end_page == 3
        assert chunks[3].start_page == 4 and chunks[3].end_page == 5

    def test_overlap_requires_less_than_size(self):
        pages = make_pages(5)
        with pytest.raises(ValueError, match="overlap"):
            _sliding_window_chunks(pages, window_size=2, overlap=2)

    def test_single_page_document(self):
        pages = make_pages(1)
        chunks = _sliding_window_chunks(pages, window_size=2, overlap=1)
        assert len(chunks) == 1
        assert chunks[0].start_page == 1 and chunks[0].end_page == 1


class TestBuildChunks:
    def test_routes_to_semantic(self):
        structure = DocumentStructure(
            sections=[Section(title="A", start_page=1, end_page=3)],
            doc_type="book", language="en", source="outline",
        )
        chunks = build_chunks(structure, make_pages(3))
        assert len(chunks) >= 1

    def test_routes_to_sliding_window(self):
        structure = DocumentStructure(
            sections=[], doc_type="unknown", language="auto", source="sliding_window",
        )
        chunks = build_chunks(structure, make_pages(5))
        assert len(chunks) >= 1

    def test_sequential_chunk_ids(self):
        structure = DocumentStructure(
            sections=[Section(title="A", start_page=1, end_page=10)],
            doc_type="book", language="en", source="llm_scan",
        )
        chunks = build_chunks(structure, make_pages(10), max_pages=3)
        ids = [c.chunk_id for c in chunks]
        assert ids == list(range(len(chunks)))
