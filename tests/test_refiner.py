"""Tests for pdfmark_ai.refiner."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from pdfmark_ai.refiner import refine, assemble_fragments
from pdfmark_ai.models import DocumentStructure, ExtractionResult, Section


class TestAssembleFragments:
    def test_joins_with_separator(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown="# A\nContent", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=3, end_page=4,
                             section_title="B", markdown="# B\nMore", tail_summary=""),
        ]
        joined = assemble_fragments(results)
        assert "---" in joined
        assert "# A" in joined
        assert "# B" in joined

    def test_empty_results(self):
        assert assemble_fragments([]) == ""


class TestRefine:
    @pytest.mark.asyncio
    async def test_successful_refine(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown="# A\nDup content", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="A", markdown="Dup content\n# B\nNew", tail_summary=""),
        ]
        structure = DocumentStructure(
            sections=[Section(title="A", start_page=1, end_page=3)],
            doc_type="paper", language="en", source="outline",
        )
        client = MagicMock()
        client.refine = AsyncMock(return_value="# A\nDup content\n# B\nNew")

        result = await refine(results, client, structure)
        assert client.refine.called
        assert "# A" in result

    @pytest.mark.asyncio
    async def test_empty_results_skip_api(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=1,
                             section_title="", markdown="", tail_summary=""),
        ]
        structure = DocumentStructure(
            sections=[], doc_type="unknown", language="auto", source="sliding_window",
        )
        client = MagicMock()
        result = await refine(results, client, structure)
        assert result == ""
        client.refine.assert_not_called()
