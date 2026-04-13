"""LLM prompt templates for all pipeline stages."""

from __future__ import annotations

from pdfmark_ai.models import Chunk, DocumentStructure, ExtractionResult


LANG_MAP: dict[str, str] = {
    "zh": "请用中文输出所有内容。",
    "en": "Output all content in English.",
    "auto": "Maintain the original language of the document.",
}

SCAN_SYSTEM = """You are a document structure analyst. Analyze the provided PDF page images and identify the document structure.

Return ONLY a JSON object (no markdown, no explanation) with this exact schema:
{"sections": [{"title": "Section Title", "start_page": 1, "end_page": 5}], "doc_type": "paper|book|report|slides|other", "language": "zh|en|auto"}

Rules:
- Field names MUST be exactly: "title", "start_page", "end_page", "sections", "doc_type", "language"
- page numbers are 1-indexed
- sections should cover all pages without gaps
- merge consecutive pages into logical sections
- doc_type should reflect the dominant format
- language should be "zh" or "en" or "auto"
- Do NOT wrap in ```json``` code fences"""


def build_scan_prompt(total_pages: int, language: str = "auto") -> str:
    """Build the user prompt for structure scanning."""
    lang_instruction = LANG_MAP.get(language, LANG_MAP["auto"])
    return f"""Analyze these sampled pages from a {total_pages}-page document.

{lang_instruction}

Identify all major sections with their page ranges. Return ONLY valid JSON matching the schema, no other text."""


EXTRACTION_SYSTEM = """You are a precise document converter. Convert PDF page images to high-quality Markdown with exact fidelity.

Rules:
- Preserve heading hierarchy (# ## ### etc.)
- Use Markdown tables for tabular data
- Use fenced code blocks with language tags
- Use LaTeX for math: $inline$ and $$block$$
- Strip headers, footers, page numbers, watermarks
- Linearize dual-column layouts (left column first)
- Include <!-- pages: x-y --> annotation at the start
- Do NOT wrap output in markdown code fences
- Output Markdown ONLY, no explanations

Image handling:
- If a figure manifest is provided (between "--- Available Figures ---" markers), use the EXACT placeholder format shown, e.g. ![brief description]({{IMG:3:0}})
- Place the image placeholder at the appropriate location in the document flow
- If no manifest is provided, use ![description](figure-N) as a generic placeholder"""


def build_extraction_prompt(
    chunk: Chunk,
    structure: DocumentStructure,
    image_manifest: str = "",
) -> str:
    """Build the extraction prompt for a single chunk."""
    parts = [
        f"Document type: {structure.doc_type}",
        f"Section: {chunk.section_title}",
        f"Pages: {chunk.start_page}-{chunk.end_page}",
    ]

    if chunk.context:
        parts.append(
            "\n--- Context from previous chunk (for continuity) ---\n"
            f"{chunk.context}\n"
            "--- End context ---"
        )

    if image_manifest:
        parts.append(image_manifest)

    parts.append(LANG_MAP.get(structure.language, LANG_MAP["auto"]))
    parts.append("\nConvert these page images to Markdown:")

    return "\n".join(parts)


REFINE_SYSTEM = """You are a document quality editor. Your task is to refine assembled Markdown fragments into a cohesive, publication-quality document."""


def build_refine_prompt(results: list[ExtractionResult]) -> str:
    """Build the refinement prompt from assembled extraction results."""
    fragments = []
    for r in results:
        fragments.append(f"<!-- pages: {r.start_page}-{r.end_page} -->\n{r.markdown}")

    joined = "\n\n---\n\n".join(fragments)

    return f"""Below are Markdown fragments extracted from a PDF. Some content may be duplicated due to overlapping page ranges, and some content may be broken across fragment boundaries.

Your tasks:
1. Remove duplicate content caused by overlapping regions
2. Fix cross-fragment breaks (merge split paragraphs, lists, tables)
3. Unify heading levels across the document
4. Do NOT add content not present in the original
5. Preserve <!-- pages: x-y --> annotations
6. Output Markdown ONLY, no explanations

Fragments:
{joined}"""
