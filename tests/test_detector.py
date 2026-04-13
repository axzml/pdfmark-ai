"""Tests for pdfmark_ai.detector."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from pdfmark_ai.detector import detect_structure, _parse_outline, _parse_structure_response
from pdfmark_ai.models import DocumentStructure, PageImage, Section


class TestParseOutline:
    def test_basic_toc(self):
        """PyMuPDF get_toc() returns [[level, title, page_number], ...]."""
        toc = [
            [1, "Introduction", 1],
            [1, "Methods", 6],
            [2, "Data Collection", 8],
            [1, "Results", 15],
        ]
        sections = _parse_outline(toc, total_pages=20)
        assert len(sections) == 4
        assert sections[0].title == "Introduction"
        assert sections[0].start_page == 1
        assert sections[0].end_page == 5  # before next section starts

    def test_filter_orphans(self):
        """Single-page sections are filtered out."""
        toc = [
            [1, "Introduction", 1],
            [1, "Methods", 3],
            [1, "Orphan", 5],  # only 1 page before next section at 6
            [1, "Results", 6],
        ]
        sections = _parse_outline(toc, total_pages=10)
        titles = [s.title for s in sections]
        assert "Orphan" not in titles

    def test_empty_toc(self):
        sections = _parse_outline([], total_pages=10)
        assert sections == []

    def test_coverage_check(self):
        """TOC covering < 50% of pages returns None (unreliable)."""
        # Only 2 short sections starting late in a long document
        toc = [
            [1, "Appendix A", 45],
            [1, "Appendix B", 48],
        ]
        # Appendix A: 45-47, Appendix B: 48-50 => 6 pages out of 50 = 12%
        # But last section extends to total_pages=50, so covered = 45-50 = 6 pages / 50 = 12%
        result = _parse_outline(toc, total_pages=50)
        assert result is None


class TestParseStructureResponse:
    def test_valid_json(self):
        raw = json.dumps({
            "sections": [{"title": "A", "start_page": 1, "end_page": 5}],
            "doc_type": "paper",
            "language": "en",
            "source": "llm_scan",
        })
        result = _parse_structure_response(raw, total_pages=10)
        assert result is not None
        assert len(result.sections) == 1

    def test_json_in_code_fences(self):
        raw = '```json\n{"sections": [{"title": "A", "start_page": 1, "end_page": 5}], "doc_type": "other", "language": "auto", "source": "llm_scan"}\n```'
        result = _parse_structure_response(raw, total_pages=5)
        assert result is not None
        assert result.doc_type == "other"
        assert len(result.sections) == 1

    def test_invalid_json_falls_back(self):
        result = _parse_structure_response("not json at all", total_pages=5)
        assert result is None

    def test_empty_sections_falls_back(self):
        raw = json.dumps({
            "sections": [], "doc_type": "other", "language": "auto", "source": "llm_scan"
        })
        result = _parse_structure_response(raw, total_pages=5)
        assert result is None

    def test_camel_case_fields_normalized(self):
        """LLM returns startPage/endPage instead of start_page/end_page."""
        raw = json.dumps({
            "sections": [{"title": "A", "startPage": 1, "endPage": 5}],
            "doc_type": "paper", "language": "en",
        })
        result = _parse_structure_response(raw, total_pages=5)
        assert result is not None
        assert result.sections[0].start_page == 1
        assert result.sections[0].end_page == 5

    def test_alternative_field_names(self):
        """LLM uses 'name' and 'heading' instead of 'title'."""
        raw = json.dumps({
            "sections": [{"name": "Intro", "page_start": 1, "page_end": 3}],
        })
        result = _parse_structure_response(raw, total_pages=3)
        assert result is not None
        assert result.sections[0].title == "Intro"
        assert result.sections[0].start_page == 1

    def test_json_embedded_in_text(self):
        """LLM wraps JSON with explanatory text."""
        raw = 'Here is the structure I found:\n{"sections": [{"title": "A", "start_page": 1, "end_page": 5}], "doc_type": "report", "language": "auto"}\nDone.'
        result = _parse_structure_response(raw, total_pages=5)
        assert result is not None
        assert len(result.sections) == 1

    def test_missing_doc_type_defaults(self):
        """Missing doc_type/language fields get sensible defaults."""
        raw = json.dumps({
            "sections": [{"title": "X", "start_page": 1, "end_page": 10}],
        })
        result = _parse_structure_response(raw, total_pages=10)
        assert result is not None
        assert result.doc_type == "unknown"
        assert result.language == "auto"
        assert result.source == "llm_scan"


class TestDetectStructure:
    @pytest.mark.asyncio
    async def test_tier1_outline_success(self, tmp_path):
        """Tier 1: valid TOC with good coverage -> use outline."""
        toc = [[1, "Ch1", 1], [1, "Ch2", 10], [1, "Ch3", 20]]
        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = toc
        mock_doc.page_count = 30
        mock_doc.close = MagicMock()

        with patch("pdfmark_ai.detector.fitz.open", return_value=mock_doc):
            result = await detect_structure(tmp_path / "test.pdf", [], None, "auto")
            assert result.source == "outline"
            assert len(result.sections) == 3

    @pytest.mark.asyncio
    async def test_tier3_fallback_no_client(self, tmp_path):
        """When no client provided and no TOC, fall back to sliding window."""
        mock_doc = MagicMock()
        mock_doc.get_toc.return_value = []
        mock_doc.close = MagicMock()

        with patch("pdfmark_ai.detector.fitz.open", return_value=mock_doc):
            result = await detect_structure(tmp_path / "test.pdf", [], None, "auto")
            assert result.source == "sliding_window"
