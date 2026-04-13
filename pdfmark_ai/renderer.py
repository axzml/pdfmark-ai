"""PDF rendering with PyMuPDF and SHA-256 progressive cache."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import fitz  # PyMuPDF

from pdfmark_ai.models import PageImage

logger = logging.getLogger(__name__)


def get_pdf_hash(pdf_path: Path) -> str:
    """Return first 16 hex chars of SHA-256 hash of the PDF file."""
    return hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]


def get_sample_pages(total_pages: int, first_n: int = 5, every_n: int = 10) -> list[int]:
    """Return 1-indexed page numbers for structure scanning samples."""
    pages = set(range(1, min(first_n, total_pages) + 1))
    for i in range(every_n, total_pages + 1, every_n):
        pages.add(i)
    pages.add(total_pages)
    return sorted(pages)


def render_pdf(
    pdf_path: Path,
    cache_dir: Path | None = None,
    dpi: int = 150,
) -> list[PageImage]:
    """Render PDF pages to PNG images, with optional SHA-256 progressive cache.

    When cache_dir is provided:
      - Full cache hit: loads all pages from disk, skips rendering entirely.
      - Partial cache hit: loads cached pages, renders only missing ones,
        merges results, and fills in the cache.
      - Full miss or DPI change: clears stale cache, renders all, saves to cache.
    """
    if cache_dir is None:
        return _render_pages(pdf_path, dpi)

    pdf_hash = get_pdf_hash(pdf_path)
    page_cache_dir = cache_dir / pdf_hash
    meta_path = page_cache_dir / "meta.json"

    # Try to load from cache
    if page_cache_dir.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            cached_dpi = meta["dpi"]
            total_pages = meta["page_count"]

            if cached_dpi == dpi:
                # Load whatever cached pages exist
                cached_pages = _load_cached_pages(page_cache_dir, total_pages)

                if len(cached_pages) == total_pages:
                    # Full cache hit
                    return cached_pages

                # Partial cache hit — render missing pages and merge
                missing_numbers = _missing_page_numbers(cached_pages, total_pages)
                logger.info(
                    "Partial cache hit: %d/%d pages cached, rendering %d missing",
                    len(cached_pages),
                    total_pages,
                    len(missing_numbers),
                )
                new_pages = _render_page_numbers(pdf_path, dpi, missing_numbers)
                all_pages = cached_pages + new_pages
                all_pages.sort(key=lambda p: p.page_number)

                # Update cache with newly rendered pages
                _save_to_cache(page_cache_dir, all_pages, dpi)
                return all_pages
            else:
                # DPI changed — clear and re-render
                import shutil

                shutil.rmtree(page_cache_dir, ignore_errors=True)

        except (json.JSONDecodeError, KeyError):
            import shutil

            shutil.rmtree(page_cache_dir, ignore_errors=True)

    # Full miss — render everything and save to cache
    pages = _render_pages(pdf_path, dpi)
    page_cache_dir.mkdir(parents=True, exist_ok=True)
    _save_to_cache(page_cache_dir, pages, dpi)
    return pages


def _render_pages(pdf_path: Path, dpi: int) -> list[PageImage]:
    """Render all pages of a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for i, page in enumerate(doc):
        try:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pages.append(PageImage(page_number=i + 1, image_bytes=pix.tobytes("png")))
        except Exception as e:
            logger.warning(f"Failed to render page {i + 1}: {e}")

    doc.close()
    return pages


def _render_page_numbers(
    pdf_path: Path, dpi: int, page_numbers: list[int]
) -> list[PageImage]:
    """Render only the specified page numbers (1-indexed) from a PDF."""
    if not page_numbers:
        return []

    doc = fitz.open(str(pdf_path))
    pages = []
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    number_set = set(page_numbers)

    for i, page in enumerate(doc):
        page_num = i + 1
        if page_num in number_set:
            try:
                pix = page.get_pixmap(matrix=mat, alpha=False)
                pages.append(
                    PageImage(page_number=page_num, image_bytes=pix.tobytes("png"))
                )
            except Exception as e:
                logger.warning(f"Failed to render page {page_num}: {e}")

    doc.close()
    return pages


def _missing_page_numbers(
    cached_pages: list[PageImage], total_pages: int
) -> list[int]:
    """Return sorted list of page numbers not present in cached_pages."""
    cached_numbers = {p.page_number for p in cached_pages}
    return sorted(set(range(1, total_pages + 1)) - cached_numbers)


def _load_cached_pages(cache_dir: Path, total_pages: int) -> list[PageImage]:
    """Load previously rendered pages from cache."""
    pages = []
    for i in range(1, total_pages + 1):
        path = cache_dir / f"page_{i:03d}.png"
        if path.exists():
            pages.append(PageImage(page_number=i, image_bytes=path.read_bytes()))
    return pages


def _save_to_cache(cache_dir: Path, pages: list[PageImage], dpi: int) -> None:
    """Save rendered pages and metadata to cache."""
    meta = {"page_count": len(pages), "dpi": dpi}
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    for page in pages:
        (cache_dir / f"page_{page.page_number:03d}.png").write_bytes(page.image_bytes)
