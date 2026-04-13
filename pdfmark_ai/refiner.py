"""Optional LLM global refinement for deduplication and coherence."""

from __future__ import annotations

import logging

from pdfmark_ai.client import LLMClient
from pdfmark_ai.models import DocumentStructure, ExtractionResult
from pdfmark_ai.prompts import REFINE_SYSTEM, build_refine_prompt

logger = logging.getLogger(__name__)


def assemble_fragments(results: list[ExtractionResult]) -> str:
    """Join all extraction results with separator for refinement."""
    if not results:
        return ""
    sorted_results = sorted(results, key=lambda r: r.chunk_id)
    return "\n\n---\n\n".join(
        f"<!-- pages: {r.start_page}-{r.end_page} -->\n{r.markdown}"
        for r in sorted_results
        if r.markdown.strip()
    )


async def refine(
    results: list[ExtractionResult],
    client: LLMClient,
    structure: DocumentStructure,
) -> str:
    """Run optional LLM global refinement."""
    fragments = assemble_fragments(results)
    if not fragments.strip():
        return ""

    prompt = build_refine_prompt(results)
    logger.info("Running LLM global refinement...")
    return await client.refine(fragments, REFINE_SYSTEM, prompt)
