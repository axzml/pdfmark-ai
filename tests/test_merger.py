"""Tests for pdfmark_ai.merger."""

import re
from pathlib import Path
from pdfmark_ai.merger import clean_boundaries, normalize_headings, merge_results, dedup_overlap
from pdfmark_ai.models import DocumentStructure, ExtractionResult, Section


class TestDedupOverlap:
    def test_removes_boundary_duplicates(self):
        """Consecutive chunks sharing a paragraph should dedup the second occurrence."""
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown="# A\n\nShared paragraph\n\nMore A", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="B", markdown="Shared paragraph\n\nUnique B", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        assert len(deduped) == 2
        # "Shared paragraph" should appear only once (in first chunk)
        full = "\n\n".join(r.markdown for r in deduped)
        count = full.count("Shared paragraph")
        assert count == 1

    def test_preserves_unique_content(self):
        """Content unique to each chunk should be preserved."""
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown="Only A\nOverlap\nEnd A", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="B", markdown="Overlap\nOnly B", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        full = "\n".join(r.markdown for r in deduped)
        assert "Only A" in full
        assert "Only B" in full
        assert "End A" in full

    def test_preserves_code_blocks(self):
        """Code blocks should never be deduped."""
        code = "```python\nprint('hello')\n```"
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown=f"{code}\nSome text", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="B", markdown=f"{code}\nNew text", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        full = "\n".join(r.markdown for r in deduped)
        assert full.count("print('hello')") == 2  # both preserved

    def test_preserves_annotations(self):
        """<!-- pages: x-y --> annotations should not be deduped."""
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="", markdown="<!-- pages: 1-2 -->\nIntro", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="", markdown="<!-- pages: 2-3 -->\nIntro\nMore", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        # Both annotations should remain
        full = "\n".join(r.markdown for r in deduped)
        assert full.count("<!-- pages:") == 2

    def test_single_chunk_passthrough(self):
        """With only one chunk, nothing should be removed."""
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=1,
                             section_title="", markdown="Hello world", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        assert len(deduped) == 1
        assert deduped[0].markdown == "Hello world"

    def test_case_insensitive_dedup(self):
        """Dedup should be case-insensitive."""
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="", markdown="Hello World\n\nContent A", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=2, end_page=3,
                             section_title="", markdown="hello world\n\nContent B", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        full = "\n\n".join(r.markdown for r in deduped)
        count = full.lower().count("hello world")
        assert count == 1

    def test_format_diff_dedup(self):
        """Same content in different formats should be deduped.

        Chunk 1 has plain text, chunk 2 has bold + backtick formatting.
        Paragraph-level normalization should catch this as a duplicate.
        """
        chunk1_content = (
            "Ashish Vaswani*\tNoam Shazeer*\tNiki Parmar*\tJakob Uszkoreit*\n"
            "Google Brain\tGoogle Brain\tGoogle Research\tGoogle Research\n"
            "avaswani@google.com\tnoam@google.com\tnikip@google.com\tusz@google.com"
        )
        chunk2_content = (
            "**Ashish Vaswani***\t**Noam Shazeer***\t**Niki Parmar***\t**Jakob Uszkoreit***\n"
            "Google Brain\tGoogle Brain\tGoogle Research\tGoogle Research\n"
            "`avaswani@google.com`\t`noam@google.com`\t`nikip@google.com`\t`usz@google.com`"
        )
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=1,
                             section_title="", markdown="# Title\n\n" + chunk1_content + "\n\n## Section", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=1, end_page=2,
                             section_title="", markdown=chunk2_content + "\n\nNew content", tail_summary=""),
        ]
        deduped = dedup_overlap(results)
        full = "\n".join(r.markdown for r in deduped)
        # "Ashish Vaswani" should appear only once (in chunk 1)
        assert full.count("Ashish Vaswani") == 1, f"Expected 1, got {full.count('Ashish Vaswani')}"
        assert full.count("Noam Shazeer") == 1
        assert "New content" in full


class TestCleanBoundaries:
    def test_joins_chunks(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=2,
                             section_title="A", markdown="# A\nContent 1", tail_summary=""),
            ExtractionResult(chunk_id=1, start_page=3, end_page=4,
                             section_title="B", markdown="# B\nContent 2", tail_summary=""),
        ]
        md = clean_boundaries(results)
        assert "# A\nContent 1" in md
        assert "# B\nContent 2" in md

    def test_collapses_whitespace(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=1,
                             section_title="", markdown="A\n\n\n\n\nB", tail_summary=""),
        ]
        md = clean_boundaries(results)
        assert "\n\n\n" not in md

    def test_empty_results(self):
        assert clean_boundaries([]) == ""


class TestNormalizeHeadings:
    def test_shift_to_h1(self):
        md = "## Title\n\n### Subtitle"
        result = normalize_headings(md)
        assert result.startswith("# Title")
        assert "## Subtitle" in result

    def test_already_h1(self):
        md = "# Title\n\n## Sub"
        result = normalize_headings(md)
        assert result == "# Title\n\n## Sub"

    def test_no_headings(self):
        assert normalize_headings("Just text") == "Just text"

    def test_clamp_to_h6(self):
        md = "## Title\n####### Too deep"
        result = normalize_headings(md)
        assert result.startswith("# Title")
        assert "###### Too deep" in result

    def test_canonicalizes_inconsistent_levels(self):
        """Same heading at different levels across chunks uses the most prominent."""
        md = "## Contents\n\nText A\n\n# Contents\n\nText B"
        result = normalize_headings(md)
        # Both "Contents" should be at the same level (the most prominent: h1)
        lines = [l for l in result.splitlines() if l.strip().startswith("#")]
        assert all(l.startswith("# ") for l in lines)
        # Only one "# Contents" should remain (dedup is separate, but levels must match)
        heading_levels = [len(l.split()[0]) for l in lines]
        assert len(set(heading_levels)) == 1  # all at same level

    def test_preserves_relative_hierarchy(self):
        """Relative heading hierarchy should be preserved after normalization."""
        md = "## Chapter 1\n\n### Section A\n\n#### Detail\n\n## Chapter 2\n\n### Section B"
        result = normalize_headings(md)
        lines = result.splitlines()
        # Find heading levels
        levels = []
        for l in lines:
            if l.startswith("#"):
                levels.append(len(l.split()[0]))
        assert levels == [1, 2, 3, 1, 2]


class TestMergeResults:
    def test_includes_frontmatter(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=3,
                             section_title="Intro", markdown="# Intro\n\nHello", tail_summary=""),
        ]
        structure = DocumentStructure(
            sections=[Section(title="Intro", start_page=1, end_page=3)],
            doc_type="paper", language="en", source="outline",
        )
        md = merge_results(results, structure, Path("test.pdf"))
        assert md.startswith("---")
        assert "source: \"test.pdf\"" in md
        assert "pages: 3" in md
        assert "# Intro" in md

    def test_no_frontmatter(self):
        results = [
            ExtractionResult(chunk_id=0, start_page=1, end_page=1,
                             section_title="", markdown="Hello", tail_summary=""),
        ]
        structure = DocumentStructure(
            sections=[], doc_type="unknown", language="auto", source="sliding_window",
        )
        md = merge_results(results, structure, Path("test.pdf"), frontmatter=False)
        assert not md.startswith("---")
        assert md == "Hello"
