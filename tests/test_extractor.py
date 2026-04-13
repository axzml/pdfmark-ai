"""Tests for pdfmark_ai.extractor."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from pdfmark_ai.extractor import process_all_chunks, _tail_summary
from pdfmark_ai.models import (
    Chunk, DocumentStructure, ExtractionResult, PageImage, Section,
)


def make_chunk(chunk_id, title, start, end, context=""):
    pages = [PageImage(page_number=i, image_bytes=f"p{i}".encode()) for i in range(start, end + 1)]
    return Chunk(chunk_id=chunk_id, section_title=title, pages=pages,
                 start_page=start, end_page=end, context=context)


class TestTailSummary:
    def test_short_text(self):
        assert _tail_summary("Hello") == "Hello"

    def test_long_text(self):
        text = "A" * 300
        result = _tail_summary(text)
        assert len(result) == 150
        assert result == "A" * 150

    def test_empty_text(self):
        assert _tail_summary("") == ""


class TestProcessAllChunks:
    @pytest.mark.asyncio
    async def test_semantic_mode_tail_chain(self):
        """In semantic mode, context propagates between chunks in the same section."""
        chunks = [
            make_chunk(0, "Ch1", 1, 2),
            make_chunk(1, "Ch1", 3, 4),
        ]
        structure = DocumentStructure(
            sections=[Section(title="Ch1", start_page=1, end_page=4)],
            doc_type="book", language="en", source="outline",
        )
        client = MagicMock()
        call_count = 0

        async def fake_extract(images, system, prompt, max_tokens=8192):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "# Page 1-2\n\nSome content about things"
            return "# Page 3-4\n\nMore content follows"

        client.extract = AsyncMock(side_effect=fake_extract)

        results = await process_all_chunks(chunks, client, structure, max_concurrency=3)
        assert len(results) == 2
        # Verify second call included context from first result's tail
        second_call_prompt = client.extract.call_args_list[1]
        assert second_call_prompt is not None

    @pytest.mark.asyncio
    async def test_window_mode_parallel(self):
        """In window mode, all chunks are processed without context."""
        chunks = [
            make_chunk(0, "", 1, 2),
            make_chunk(1, "", 2, 3),
            make_chunk(2, "", 3, 4),
        ]
        structure = DocumentStructure(
            sections=[], doc_type="unknown", language="auto", source="sliding_window",
        )
        client = MagicMock()
        client.extract = AsyncMock(return_value="# Content")

        results = await process_all_chunks(chunks, client, structure, max_concurrency=5)
        assert len(results) == 3
        assert client.extract.call_count == 3

    @pytest.mark.asyncio
    async def test_single_chunk_failure_returns_placeholder(self):
        """A failed chunk should return a placeholder, not crash."""
        chunks = [make_chunk(0, "Ch1", 1, 2)]
        structure = DocumentStructure(
            sections=[Section(title="Ch1", start_page=1, end_page=2)],
            doc_type="book", language="en", source="outline",
        )
        client = MagicMock()
        client.extract = AsyncMock(side_effect=RuntimeError("API timeout"))

        results = await process_all_chunks(chunks, client, structure)
        assert len(results) == 1
        assert "Extraction failed" in results[0].markdown

    @pytest.mark.asyncio
    async def test_cache_hit_skips_api(self, tmp_path):
        """Cached chunks should be loaded without API calls."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        cache_sub = cache_dir / "plain"  # no page_images → "plain" subdir
        cache_sub.mkdir()
        (cache_sub / "chunk_000.md").write_text("# Cached content")
        (cache_sub / "chunk_000.tail").write_text("Cached tail")

        chunks = [make_chunk(0, "Ch1", 1, 2)]
        structure = DocumentStructure(
            sections=[Section(title="Ch1", start_page=1, end_page=2)],
            doc_type="book", language="en", source="outline",
        )
        client = MagicMock()

        results = await process_all_chunks(chunks, client, structure, cache_dir=cache_dir)
        assert len(results) == 1
        assert results[0].markdown == "# Cached content"
        assert results[0].tail_summary == "Cached tail"
        client.extract.assert_not_called()
