"""Parallel chunk extraction with tail_summary context chain and cache awareness."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pdfmark_ai.client import LLMClient
from pdfmark_ai.models import Chunk, DocumentStructure, ExtractionResult
from pdfmark_ai.prompts import EXTRACTION_SYSTEM, build_extraction_prompt

logger = logging.getLogger(__name__)

TAIL_LENGTH = 150


def _tail_summary(text: str, length: int = TAIL_LENGTH) -> str:
    return text[-length:] if len(text) > length else text


async def _process_chunk(
    chunk: Chunk,
    client: LLMClient,
    structure: DocumentStructure,
) -> ExtractionResult:
    """Process a single chunk: call LLM, return result."""
    prompt = build_extraction_prompt(chunk, structure)
    markdown = await client.extract(
        images=[p.image_bytes for p in chunk.pages],
        system=EXTRACTION_SYSTEM,
        prompt=prompt,
    )
    tail = _tail_summary(markdown)
    return ExtractionResult(
        chunk_id=chunk.chunk_id,
        start_page=chunk.start_page,
        end_page=chunk.end_page,
        section_title=chunk.section_title,
        markdown=markdown,
        tail_summary=tail,
    )


def _cache_subdir(cache_dir: Path, crop_mode: bool = False) -> Path:
    """Return a subdirectory for cache entries, separating crop modes."""
    tag = "crop" if crop_mode else "plain"
    return cache_dir / tag


def _load_chunk_cache(cache_dir: Path, chunk: Chunk, crop_mode: bool = False) -> ExtractionResult | None:
    """Load a chunk result from cache. Returns None if not cached."""
    sub = _cache_subdir(cache_dir, crop_mode)
    md_path = sub / f"chunk_{chunk.chunk_id:03d}.md"
    tail_path = sub / f"chunk_{chunk.chunk_id:03d}.tail"
    if not md_path.exists():
        return None
    try:
        markdown = md_path.read_text(encoding="utf-8")
        tail = tail_path.read_text(encoding="utf-8") if tail_path.exists() else ""
        return ExtractionResult(
            chunk_id=chunk.chunk_id,
            start_page=chunk.start_page,
            end_page=chunk.end_page,
            section_title=chunk.section_title,
            markdown=markdown,
            tail_summary=tail,
        )
    except Exception:
        return None


def _save_chunk_cache(cache_dir: Path, result: ExtractionResult, crop_mode: bool = False) -> None:
    """Save a chunk result to cache."""
    sub = _cache_subdir(cache_dir, crop_mode)
    sub.mkdir(parents=True, exist_ok=True)
    (sub / f"chunk_{result.chunk_id:03d}.md").write_text(
        result.markdown, encoding="utf-8"
    )
    (sub / f"chunk_{result.chunk_id:03d}.tail").write_text(
        result.tail_summary, encoding="utf-8"
    )


async def _process_chapter(
    chunks: list[Chunk],
    client: LLMClient,
    structure: DocumentStructure,
    cache_dir: Path | None = None,
    crop_mode: bool = False,
) -> list[ExtractionResult]:
    """Process chunks in a chapter serially, propagating tail_summary."""
    results = []
    for chunk in chunks:
        if cache_dir is not None:
            cached = _load_chunk_cache(cache_dir, chunk, crop_mode)
            if cached is not None:
                results.append(cached)
                chunk.context = cached.tail_summary
                continue

        try:
            result = await _process_chunk(chunk, client, structure)
            results.append(result)
            if cache_dir is not None:
                _save_chunk_cache(cache_dir, result, crop_mode)
            chunk.context = result.tail_summary
        except Exception as e:
            logger.error(f"Chunk {chunk.chunk_id} (pages {chunk.start_page}-{chunk.end_page}) failed: {e}")
            placeholder = ExtractionResult(
                chunk_id=chunk.chunk_id,
                start_page=chunk.start_page,
                end_page=chunk.end_page,
                section_title=chunk.section_title,
                markdown=f"<!-- pages: {chunk.start_page}-{chunk.end_page} -->\n[Extraction failed: {e}]",
                tail_summary="",
            )
            results.append(placeholder)
    return results


def _group_by_section(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Group consecutive chunks by section_title."""
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_title = None

    for chunk in chunks:
        if chunk.section_title != current_title:
            if current:
                groups.append(current)
            current = [chunk]
            current_title = chunk.section_title
        else:
            current.append(chunk)
    if current:
        groups.append(current)
    return groups


async def process_all_chunks(
    chunks: list[Chunk],
    client: LLMClient,
    structure: DocumentStructure,
    max_concurrency: int = 3,
    cache_dir: Path | None = None,
    crop_mode: bool = False,
) -> list[ExtractionResult]:
    """Process all chunks with mode-dependent concurrency."""
    if not chunks:
        return []

    if structure.source == "sliding_window":
        # Window mode: all chunks fully parallel
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded(chunk):
            async with semaphore:
                if cache_dir is not None:
                    cached = _load_chunk_cache(cache_dir, chunk, crop_mode)
                    if cached is not None:
                        return cached
                try:
                    result = await _process_chunk(chunk, client, structure)
                    if cache_dir is not None:
                        _save_chunk_cache(cache_dir, result, crop_mode)
                    return result
                except Exception as e:
                    logger.error(f"Chunk {chunk.chunk_id} failed: {e}")
                    return ExtractionResult(
                        chunk_id=chunk.chunk_id, start_page=chunk.start_page,
                        end_page=chunk.end_page, section_title=chunk.section_title,
                        markdown=f"<!-- pages: {chunk.start_page}-{chunk.end_page} -->\n[Extraction failed: {e}]",
                    )

        results = await asyncio.gather(*[_guarded(c) for c in chunks])
        return list(results)
    else:
        # Semantic mode: inter-chapter parallel, intra-chapter serial
        groups = _group_by_section(chunks)
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _guarded_group(group):
            async with semaphore:
                return await _process_chapter(group, client, structure, cache_dir, crop_mode)

        group_results = await asyncio.gather(*[_guarded_group(g) for g in groups])
        return [r for group in group_results for r in group]
