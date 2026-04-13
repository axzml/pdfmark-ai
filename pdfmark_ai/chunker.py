"""Chunking strategies: semantic (section-aligned) and sliding window."""

from __future__ import annotations

from pdfmark_ai.models import Chunk, DocumentStructure, PageImage, Section


def build_chunks(
    structure: DocumentStructure,
    pages: list[PageImage],
    max_pages: int = 4,
    window_size: int = 2,
    window_overlap: int = 1,
) -> list[Chunk]:
    """Route to the appropriate chunking strategy based on structure source."""
    if structure.source in ("outline", "llm_scan") and structure.sections:
        return _semantic_chunks(structure, pages, max_pages)
    return _sliding_window_chunks(pages, window_size, window_overlap)


def _page_map(pages: list[PageImage]) -> dict[int, PageImage]:
    """Create a lookup from page_number to PageImage."""
    return {p.page_number: p for p in pages}


def _semantic_chunks(
    structure: DocumentStructure,
    pages: list[PageImage],
    max_pages: int,
) -> list[Chunk]:
    """Split pages along section boundaries, max max_pages per chunk."""
    pm = _page_map(pages)
    chunks: list[Chunk] = []
    chunk_id = 0

    for section in structure.sections:
        section_pages = [
            pm[p] for p in range(section.start_page, section.end_page + 1) if p in pm
        ]
        if not section_pages:
            continue

        for i in range(0, len(section_pages), max_pages):
            group = section_pages[i : i + max_pages]
            chunks.append(Chunk(
                chunk_id=chunk_id,
                section_title=section.title,
                pages=group,
                start_page=group[0].page_number,
                end_page=group[-1].page_number,
            ))
            chunk_id += 1

    return chunks


def _sliding_window_chunks(
    pages: list[PageImage],
    window_size: int,
    overlap: int,
) -> list[Chunk]:
    """Create overlapping windows of pages."""
    if overlap >= window_size:
        raise ValueError(f"overlap ({overlap}) must be less than window_size ({window_size})")

    step = window_size - overlap
    chunks: list[Chunk] = []
    chunk_id = 0

    i = 0
    while i < len(pages):
        group = pages[i : i + window_size]
        chunks.append(Chunk(
            chunk_id=chunk_id,
            section_title="",
            pages=group,
            start_page=group[0].page_number,
            end_page=group[-1].page_number,
        ))
        chunk_id += 1
        i += step

    # Merge tiny tail chunk into the last chunk
    if len(chunks) > 1 and len(chunks[-1].pages) < window_size:
        last = chunks[-1]
        prev = chunks[-2]
        prev.pages = prev.pages + last.pages
        prev.end_page = last.end_page
        chunks.pop()

    return chunks
