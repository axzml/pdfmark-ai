"""Rule-based post-processing: dedup overlaps, clean boundaries, normalize headings, add frontmatter."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from pdfmark_ai.models import DocumentStructure, ExtractionResult


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for paragraph-level dedup comparison.

    Strips formatting artifacts (bold markers, backticks, table pipes,
    superscript LaTeX) and collapses whitespace so that the same content
    in different formats produces the same normalized string.
    """
    # Remove bold markers: **text** → text
    s = re.sub(r"\*\*([^*]+?)\*\*", r"\1", text)
    # Remove inline code backticks
    s = re.sub(r"`([^`]+?)`", r"\1", s)
    # Remove markdown table pipes and separator rows
    s = re.sub(r"\|", " ", s)
    s = re.sub(r"[-:]+", " ", s)
    # Remove LaTeX superscript markers at end of paragraph: $^\dagger$, ^\dagger, \dagger
    s = re.sub(r"\s*\$?\^?\\[a-zA-Z]+\s*\$?\s*$", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def _split_paragraphs(text: str) -> list[str]:
    """Split markdown text into paragraphs (blocks separated by blank lines).

    Returns list of raw (non-normalized) paragraph strings.
    Code blocks (``` ... ```) are kept as single units.
    HTML comments and page annotations are kept as separate paragraphs.
    Consecutive table rows (lines starting with |) separated by blank lines
    are merged into a single paragraph, since tables are often rendered with
    internal blank lines by LLMs.
    """
    raw_paragraphs: list[str] = []
    current: list[str] = []
    in_code = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            current.append(line)
            continue

        if in_code:
            current.append(line)
            continue

        if not stripped:
            if current:
                raw_paragraphs.append("\n".join(current))
                current = []
            continue

        current.append(line)

    if current:
        raw_paragraphs.append("\n".join(current))

    # Merge consecutive table paragraphs
    merged: list[str] = []
    table_buffer: list[str] = []

    for para in raw_paragraphs:
        is_table = para.strip().startswith("|")
        if is_table:
            table_buffer.append(para)
        else:
            if table_buffer:
                merged.append("\n\n".join(table_buffer))
                table_buffer = []
            merged.append(para)

    if table_buffer:
        merged.append("\n\n".join(table_buffer))

    return merged


def unwrap_nested_images(markdown: str) -> str:
    """Fix LLM-generated nested image syntax: ![desc](![inner](url)) -> ![desc](url).

    The LLM sometimes wraps the image placeholder in another markdown image
    syntax, producing ![desc](![image](url)) instead of ![desc](url).
    """
    return re.sub(
        r"(!\[[^\]]*\])\(\s*!\[[^\]]*\]\(([^)]*)\)\)",
        r"\1(\2)",
        markdown,
    )


# Pattern for LaTeX commands commonly used in display math
_LATEX_CMD_RE = re.compile(
    r"\\(?:mathrm|text|mathbf|mathit|mathcal|mathfrak|bar|hat|vec|dot|tilde|"
    r"overline|underline|boldsymbol|Omega|frac|sum|int|prod|sqrt|binom|"
    r"left|right|quad|qquad|cdots|ldots|vdots|ddots|infty|partial|prime)\b"
)

# Lines that are purely LaTeX (no Chinese/prose mixed in) and not already in math delimiters
_BARE_LATEX_LINE_RE = re.compile(
    r"^\s*([A-Za-z0-9_\^{}\[\].,;:=+\-*/\\ \t\n'\"]+\\[A-Za-z]+"
    r"[A-Za-z0-9_\^{}\[\].,;:=+\-*/\\ \t\n'\"]*)\s*\\?\s*$"
)


def fix_math_blocks(markdown: str) -> str:
    """Fix unclosed $$ blocks and wrap bare LaTeX lines in $$ delimiters.

    When LLM chunks split across display-math regions, the opening $$ may
    land in one chunk and the closing $$ in another.  The dedup pass can
    then drop one or both delimiters, leaving LaTeX commands as plain text.
    """
    lines = markdown.split("\n")
    out: list[str] = []
    in_display = False
    consecutive_bare = 0

    def _is_bare_latex(s: str) -> bool:
        """Check if a stripped line is bare LaTeX (no $ wrapping, no prose)."""
        return (
            bool(_LATEX_CMD_RE.search(s))
            and "$" not in s
            and not s.startswith("```")
            and not s.startswith("<!--")
            and len(s) > 0
        )

    for line in lines:
        stripped = line.strip()

        # Count $$ markers (but ignore $$ that appear an even number of times
        # on the same line, i.e. self-closing)
        dollar_count = stripped.count("$$")
        if dollar_count >= 2:
            # Self-closing or multiple on one line — pass through
            consecutive_bare = 0
            in_display = False
            out.append(line)
            continue
        if dollar_count == 1:
            in_display = not in_display
            consecutive_bare = 0
            out.append(line)
            continue

        # Inside a $$ block: if we hit prose, close the block
        if in_display and stripped and not _is_bare_latex(stripped):
            out.append("$$")
            in_display = False
            consecutive_bare = 0

        # Detect bare LaTeX outside math blocks
        if not in_display and _is_bare_latex(stripped):
            if consecutive_bare == 0:
                out.append("$$")
            out.append(stripped)
            consecutive_bare += 1
            continue

        # Non-LaTeX line: close any open bare-LaTeX streak
        if consecutive_bare > 0 and stripped:  # blank lines don't break the streak
            out.append("$$")
            consecutive_bare = 0

        out.append(line)

    # Close any trailing bare-LaTeX streak
    if consecutive_bare > 0:
        out.append("$$")
    if in_display:
        out.append("$$")

    return "\n".join(out)


def dedup_overlap(results: list[ExtractionResult]) -> list[ExtractionResult]:
    """Remove duplicate content at chunk boundaries.

    Compares each chunk against BOTH the previous deduped chunk AND the
    previous original chunk.  This handles:
    - Immediate boundary overlap (deduped prev catches same-text duplicates)
    - Transitive duplication across 3+ chunks (original prev catches paragraphs
      that were removed from the deduped version)
    - Non-overlapping sections with LLM-generated boundary duplication

    Code blocks, HTML comments, page annotations, and figure placeholders
    are always preserved.
    """
    if len(results) < 2:
        return results

    deduped = [results[0]]

    for i in range(1, len(results)):
        curr_paras = _split_paragraphs(results[i].markdown)

        # Build seen set from BOTH the deduped previous chunk AND the
        # original previous chunk.  The deduped chunk catches immediate
        # boundary overlap; the original chunk catches transitive duplicates
        # that were removed during the previous dedup pass.
        seen_norms: set[str] = set()
        seen_norm_list: list[str] = []
        for source in [deduped[-1].markdown, results[i - 1].markdown]:
            for p in _split_paragraphs(source):
                norm = _normalize_for_dedup(p)
                if norm:
                    seen_norms.add(norm)
                    seen_norm_list.append(norm)

        kept_paras: list[str] = []
        skipped = 0

        for para in curr_paras:
            stripped = para.strip()
            if stripped.startswith("```") or stripped.startswith("<!--"):
                kept_paras.append(para)
                continue

            if "{{FIGURE:" in para:
                kept_paras.append(para)
                continue

            norm = _normalize_for_dedup(para)
            if not norm:
                kept_paras.append(para)
                continue

            if norm in seen_norms:
                skipped += 1
                continue

            norm_tokens = set(norm.split())
            is_fuzzy_dup = False
            if len(norm_tokens) >= 5:
                for prev_norm in seen_norm_list:
                    prev_tokens = set(prev_norm.split())
                    shorter, longer = (norm_tokens, prev_tokens) if len(norm_tokens) <= len(prev_tokens) else (prev_tokens, norm_tokens)
                    overlap = len(shorter & longer)
                    if overlap / len(shorter) >= 0.85:
                        is_fuzzy_dup = True
                        break

            if is_fuzzy_dup:
                skipped += 1
            else:
                kept_paras.append(para)
                seen_norms.add(norm)
                seen_norm_list.append(norm)

        if kept_paras and skipped > 0:
            new_md = "\n\n".join(kept_paras).strip()
            if new_md:
                deduped.append(ExtractionResult(
                    chunk_id=results[i].chunk_id,
                    markdown=new_md,
                    start_page=results[i].start_page,
                    end_page=results[i].end_page,
                    section_title=results[i].section_title,
                    tail_summary=results[i].tail_summary,
                ))
        else:
            deduped.append(results[i])

    return deduped


def clean_boundaries(results: list[ExtractionResult]) -> str:
    """Join chunk results with clean spacing."""
    if not results:
        return ""
    sorted_results = sorted(results, key=lambda r: r.chunk_id)
    joined = "\n\n".join(r.markdown.strip() for r in sorted_results if r.markdown.strip())
    joined = re.sub(r"\n{3,}", "\n\n", joined)
    return joined.strip()


def normalize_headings(markdown: str) -> str:
    """Normalize heading levels across the document.

    1. For each unique heading text, find the most prominent level (lowest #)
       across all its occurrences (handles LLM per-chunk level inconsistency).
    2. Shift all headings so the global minimum becomes #, clamped to [1,6].
    """
    heading_matches = list(re.finditer(r"^(#+)\s(.+)$", markdown, re.MULTILINE))
    if not heading_matches:
        return markdown

    # Step 1: find the canonical (most prominent) level for each heading text
    text_to_min: dict[str, int] = {}
    for m in heading_matches:
        level = len(m.group(1))
        norm_text = re.sub(r"\s+", " ", m.group(2).strip().lower())
        if norm_text not in text_to_min or level < text_to_min[norm_text]:
            text_to_min[norm_text] = level

    # Step 2: global shift so the minimum canonical level becomes h1
    global_min = min(text_to_min.values())
    shift = global_min - 1

    # Build replacements (reverse order to preserve offsets)
    replacements: list[tuple[int, int, str]] = []
    for m in heading_matches:
        norm_text = re.sub(r"\s+", " ", m.group(2).strip().lower())
        canonical = text_to_min.get(norm_text, len(m.group(1)))
        new_level = max(1, min(6, canonical - shift))
        new_heading = "#" * new_level + " " + m.group(2)
        if m.group(0) != new_heading:
            replacements.append((m.start(), m.end(), new_heading))

    for start, end, replacement in reversed(replacements):
        markdown = markdown[:start] + replacement + markdown[end:]

    return markdown


def _build_frontmatter(
    structure: DocumentStructure,
    source_pdf: Path,
    total_pages: int,
) -> str:
    """Build YAML frontmatter string."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        "---",
        f"title: \"{source_pdf.stem}\"",
        f"source: \"{source_pdf.name}\"",
        f"pages: {total_pages}",
        f"doc_type: {structure.doc_type}",
        f"language: {structure.language}",
        f"structure_source: {structure.source}",
        f"parsed_at: {now}",
        "---",
    ]
    return "\n".join(lines)


def merge_results(
    results: list[ExtractionResult],
    structure: DocumentStructure,
    source_pdf: Path,
    frontmatter: bool = True,
) -> str:
    """Full merge pipeline: dedup -> clean -> normalize -> frontmatter."""
    if not results:
        return ""

    results = dedup_overlap(results)
    markdown = clean_boundaries(results)
    markdown = normalize_headings(markdown)
    markdown = unwrap_nested_images(markdown)
    markdown = fix_math_blocks(markdown)

    if frontmatter:
        total_pages = max(r.end_page for r in results)
        fm = _build_frontmatter(structure, source_pdf, total_pages)
        return f"{fm}\n\n{markdown}"

    return markdown
