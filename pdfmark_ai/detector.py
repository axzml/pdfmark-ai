"""3-tier structure detection: TOC -> LLM scan -> sliding window fallback."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

from pdfmark_ai.models import DocumentStructure, PageImage, Section
from pdfmark_ai.prompts import SCAN_SYSTEM, build_scan_prompt
from pdfmark_ai.renderer import get_sample_pages

logger = logging.getLogger(__name__)

COVERAGE_THRESHOLD = 0.5


def _parse_outline(toc: list[list], total_pages: int) -> list[Section] | None:
    """Convert PyMuPDF TOC to Section list. Returns None if coverage < threshold."""
    if not toc:
        return []

    sections = []
    for i, (level, title, page) in enumerate(toc):
        start = page
        if i + 1 < len(toc):
            end = toc[i + 1][2] - 1
        else:
            end = total_pages

        # Filter orphan sections (only 1 page)
        if end - start + 1 < 2:
            continue

        sections.append(Section(title=title.strip(), start_page=start, end_page=end))

    if not sections:
        return []

    # Coverage check
    covered = set()
    for s in sections:
        covered.update(range(s.start_page, s.end_page + 1))
    coverage = len(covered) / total_pages
    if coverage < COVERAGE_THRESHOLD:
        logger.info(f"TOC covers {coverage:.0%} of pages (< {COVERAGE_THRESHOLD:.0%}), falling back")
        return None

    return sections


def _normalize_field_names(data: dict) -> dict:
    """Normalize common field name variants to our expected schema."""
    # Normalize top-level keys
    for alt in ("chapters", "parts", "structure"):
        if alt in data and "sections" not in data:
            data["sections"] = data.pop(alt)

    # Normalize section fields
    if "sections" in data and isinstance(data["sections"], list):
        normalized = []
        for s in data["sections"]:
            if not isinstance(s, dict):
                continue
            title = s.get("title") or s.get("name") or s.get("heading") or ""
            start = s.get("start_page") or s.get("startPage") or s.get("page_start") or s.get("from")
            end = s.get("end_page") or s.get("endPage") or s.get("page_end") or s.get("to")

            if not title or start is None or end is None:
                continue

            normalized.append({
                "title": str(title).strip(),
                "start_page": int(start),
                "end_page": int(end),
            })
        data["sections"] = normalized

    return data


def _extract_json(text: str) -> dict | None:
    """Try multiple strategies to extract a JSON dict from text."""
    text = text.strip()

    # Strategy 1: strip markdown code fences then parse directly
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", text)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()

    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: find first { ... } block in the text
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        try:
            obj = json.loads(match.group())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _parse_structure_response(
    raw: str, total_pages: int
) -> DocumentStructure | None:
    """Parse LLM JSON response into DocumentStructure. Returns None on failure."""
    data = _extract_json(raw)
    if data is None:
        logger.debug("LLM scan response: no valid JSON found in %s", raw[:500])
        return None

    data = _normalize_field_names(data)

    if "sections" not in data or not data["sections"]:
        logger.debug("LLM scan response: no sections found, keys=%s", list(data.keys()))
        return None

    data.setdefault("source", "llm_scan")
    data.setdefault("doc_type", "unknown")
    data.setdefault("language", "auto")

    try:
        ds = DocumentStructure.model_validate(data)
        if not ds.sections:
            return None
        return ds
    except Exception as e:
        logger.debug("LLM scan validation failed: %s", e)
        return None


def _fill_coverage_gaps(sections: list[Section], total_pages: int) -> list[Section]:
    """Fill gaps between detected sections so no pages are skipped."""
    if not sections:
        return sections

    # Collect all covered pages
    covered = set()
    for s in sections:
        covered.update(range(s.start_page, s.end_page + 1))

    # Find gaps
    filled: list[Section] = []
    current = 1

    for section in sorted(sections, key=lambda s: s.start_page):
        # Gap before this section
        if section.start_page > current:
            gap_end = section.start_page - 1
            filled.append(Section(title="(continued)", start_page=current, end_page=gap_end))
            covered.update(range(current, gap_end + 1))
        filled.append(section)
        current = max(current, section.end_page + 1)

    # Gap after last section
    if current <= total_pages:
        filled.append(Section(title="(continued)", start_page=current, end_page=total_pages))

    if len(filled) != len(sections):
        logger.info(f"Filled coverage gaps: {len(sections)} -> {len(filled)} sections")

    return filled


async def detect_structure(
    pdf_path: Path,
    pages: list[PageImage],
    client: object | None,
    language: str = "auto",
) -> DocumentStructure:
    """3-tier structure detection.

    Tier 1: PyMuPDF TOC extraction (zero cost)
    Tier 2: LLM structure scan on sampled pages
    Tier 3: Sliding window fallback
    """
    total_pages = len(pages) if pages else 0

    # Tier 1: TOC
    try:
        doc = fitz.open(str(pdf_path))
        toc = doc.get_toc()
        doc_total = doc.page_count
        doc.close()

        # Use actual PDF page count if pages list is empty
        if total_pages == 0:
            total_pages = doc_total

        if toc:
            sections = _parse_outline(toc, total_pages)
            if sections is not None:
                sections = _fill_coverage_gaps(sections, total_pages)
                logger.info(f"Structure detected from TOC: {len(sections)} sections")
                return DocumentStructure(
                    sections=sections, doc_type="unknown", language=language,
                    source="outline",
                )
            logger.info("TOC present but coverage too low, trying LLM scan")
    except Exception as e:
        logger.warning(f"TOC extraction failed: {e}")

    # Tier 2: LLM scan
    if client and total_pages > 0:
        try:
            sample_indices = get_sample_pages(total_pages)
            sample_images = [
                p.image_bytes for p in pages if p.page_number in sample_indices
            ]

            prompt = build_scan_prompt(total_pages, language)
            raw = await client.extract(sample_images, SCAN_SYSTEM, prompt, max_tokens=16384)

            ds = _parse_structure_response(raw, total_pages)
            if ds is not None:
                ds.sections = _fill_coverage_gaps(ds.sections, total_pages)
                logger.info(f"Structure detected via LLM scan: {len(ds.sections)} sections")
                return ds
            logger.info("LLM scan produced no valid structure, falling back")
        except Exception as e:
            logger.warning(f"LLM structure scan failed: {e}")

    # Tier 3: sliding window fallback
    logger.info("Using sliding window chunking (no structure detected)")
    return DocumentStructure(
        sections=[], doc_type="unknown", language=language,
        source="sliding_window",
    )
