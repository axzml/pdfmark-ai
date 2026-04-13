"""Tests for pdfmark_ai.renderer."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from pdfmark_ai.renderer import get_pdf_hash, render_pdf, get_sample_pages


class TestGetPdfHash:
    def test_deterministic(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"fake-pdf-content")
        h1 = get_pdf_hash(pdf)
        h2 = get_pdf_hash(pdf)
        assert h1 == h2
        assert len(h1) == 16  # first 16 hex chars of SHA-256

    def test_different_files_different_hash(self, tmp_path):
        pdf1 = tmp_path / "a.pdf"
        pdf2 = tmp_path / "b.pdf"
        pdf1.write_bytes(b"content-a")
        pdf2.write_bytes(b"content-b")
        assert get_pdf_hash(pdf1) != get_pdf_hash(pdf2)


class TestGetSamplePages:
    def test_small_pdf(self):
        pages = get_sample_pages(5)
        assert pages == [1, 2, 3, 4, 5]

    def test_medium_pdf(self):
        pages = get_sample_pages(20)
        assert pages == [1, 2, 3, 4, 5, 10, 20]

    def test_large_pdf(self):
        pages = get_sample_pages(100)
        assert 1 in pages
        assert 5 in pages
        assert 10 in pages
        assert 50 in pages
        assert 100 in pages

    def test_no_duplicates(self):
        pages = get_sample_pages(10)
        assert len(pages) == len(set(pages))


class TestRenderPdf:
    def test_no_cache_renders_all_pages(self, tmp_path):
        """When cache_dir is None, render all pages without caching."""
        pages = [
            MagicMock(page_number=i, image_bytes=b"png-data")
            for i in range(1, 4)
        ]
        with patch("pdfmark_ai.renderer._render_pages", return_value=pages) as mock_render:
            result = render_pdf(tmp_path / "test.pdf", cache_dir=None, dpi=150)
            assert len(result) == 3
            mock_render.assert_called_once()

    def test_cache_hit_skips_render(self, tmp_path):
        """When all cached pages exist, skip rendering."""
        # Determine the hash for the fake PDF content so cache dir matches
        pdf_content = b"fake-pdf"
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()[:16]

        cache_dir = tmp_path / "cache" / pdf_hash
        cache_dir.mkdir(parents=True)
        meta = {"page_count": 3, "dpi": 150}
        (cache_dir / "meta.json").write_text(json.dumps(meta))
        for i in range(1, 4):
            (cache_dir / f"page_{i:03d}.png").write_bytes(b"cached-png")

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_content)

        with patch("pdfmark_ai.renderer._render_pages") as mock_render:
            result = render_pdf(pdf_path, cache_dir=tmp_path / "cache", dpi=150)
            mock_render.assert_not_called()
            assert len(result) == 3

    def test_cache_miss_renders_only_missing(self, tmp_path):
        """When some pages are cached, render only missing ones."""
        pdf_content = b"fake-pdf"
        pdf_hash = hashlib.sha256(pdf_content).hexdigest()[:16]

        cache_dir = tmp_path / "cache" / pdf_hash
        cache_dir.mkdir(parents=True)
        meta = {"page_count": 3, "dpi": 150}
        (cache_dir / "meta.json").write_text(json.dumps(meta))
        # Only page 1 is cached
        (cache_dir / "page_001.png").write_bytes(b"cached")

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(pdf_content)

        new_pages = [
            MagicMock(page_number=2, image_bytes=b"new-2"),
            MagicMock(page_number=3, image_bytes=b"new-3"),
        ]
        with (
            patch("pdfmark_ai.renderer._render_pages", return_value=new_pages),
            patch("pdfmark_ai.renderer._render_page_numbers", return_value=new_pages) as mock_partial,
        ):
            result = render_pdf(pdf_path, cache_dir=tmp_path / "cache", dpi=150)
            # _render_page_numbers should be called for the 2 missing pages
            mock_partial.assert_called_once()
            assert len(result) == 3
