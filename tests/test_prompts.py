"""Tests for pdfmark_ai.prompts."""

from pdfmark_ai.prompts import (
    SCAN_SYSTEM,
    build_scan_prompt,
    build_extraction_prompt,
    build_refine_prompt,
    LANG_MAP,
)
from pdfmark_ai.models import DocumentStructure, Section, Chunk, ExtractionResult


class TestScanPrompts:
    def test_scan_system_is_string(self):
        assert isinstance(SCAN_SYSTEM, str)
        assert len(SCAN_SYSTEM) > 50

    def test_build_scan_prompt(self):
        prompt = build_scan_prompt(total_pages=33, language="auto")
        assert "33" in prompt
        assert "sections" in prompt.lower()


class TestExtractionPrompt:
    def test_basic_prompt(self):
        chunk = Chunk(
            chunk_id=0, section_title="Intro",
            pages=[], start_page=1, end_page=3,
        )
        structure = DocumentStructure(
            sections=[Section(title="Intro", start_page=1, end_page=5)],
            doc_type="paper", language="en", source="outline",
        )
        prompt = build_extraction_prompt(chunk, structure)
        assert "Intro" in prompt
        assert "1-3" in prompt

    def test_prompt_with_context(self):
        chunk = Chunk(
            chunk_id=1, section_title="Methods",
            pages=[], start_page=4, end_page=6,
            context="The previous section concluded that...",
        )
        structure = DocumentStructure(
            sections=[], doc_type="paper", language="en", source="sliding_window",
        )
        prompt = build_extraction_prompt(chunk, structure)
        assert "The previous section concluded" in prompt

    def test_prompt_language_zh(self):
        chunk = Chunk(
            chunk_id=0, section_title="摘要",
            pages=[], start_page=1, end_page=2,
        )
        structure = DocumentStructure(
            sections=[], doc_type="paper", language="zh", source="outline",
        )
        prompt = build_extraction_prompt(chunk, structure)
        assert "中文" in prompt


class TestRefinePrompt:
    def test_basic_prompt(self):
        results = [
            ExtractionResult(
                chunk_id=0, start_page=1, end_page=3,
                section_title="Intro", markdown="# Intro\n\nContent",
                tail_summary="Content",
            ),
        ]
        prompt = build_refine_prompt(results)
        assert "# Intro" in prompt
        assert "dedup" in prompt.lower() or "duplicate" in prompt.lower()


class TestLangMap:
    def test_has_required_keys(self):
        assert "zh" in LANG_MAP
        assert "en" in LANG_MAP
        assert "auto" in LANG_MAP
