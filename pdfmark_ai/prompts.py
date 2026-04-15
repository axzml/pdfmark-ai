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
- Use fenced code blocks with language tags
- Use LaTeX for math: $inline$ and $$block$$
- Strip headers, footers, page numbers, watermarks
- Linearize dual-column body text (left column first)
- For author/affiliation blocks, render as a compact Markdown table with no empty cells: group authors by row as they appear in the paper, with name (with markers), affiliation, and email as columns. Do NOT use empty cells for spacing between rows or to align columns.
- Include <!-- pages: x-y --> annotation at the start
- Do NOT wrap output in markdown code fences
- Output Markdown ONLY, no explanations

Table handling:
- ALWAYS render tables as Markdown tables using | syntax, regardless of complexity
- Include the table caption above the table, e.g.: **Table 1:** Description here.
- For empty cells in complex tables, leave them blank (empty space between | delimiters)
- Do NOT use {{FIGURE:N}} for tables — that placeholder is reserved for images and diagrams only

Image handling:
- When you see a figure, chart, diagram, or illustration in the page, insert: ![description]({{FIGURE:N}})
- Replace N with the actual 1-indexed page number (e.g. {{FIGURE:3}} for page 3)
- Place it at the position where the figure appears in the document flow
- Include a brief description in the alt text (e.g. the figure caption or title)
- IMPORTANT: Also output the figure caption as plain text on the line below the image, e.g.:
  ![The Transformer]({{FIGURE:3}})
  **Figure 1:** The Transformer - model architecture.
- Do NOT omit the caption text — it must appear in the markdown output"""


def build_extraction_prompt(
    chunk: Chunk,
    structure: DocumentStructure,
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
