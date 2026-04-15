"""Tests for post-extraction figure cropping and placeholder resolution."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import io

from pdfmark_ai.image_extractor import (
    CroppedImage,
    FIGURE_PATTERN,
    _is_blank,
    _merge_nearby_regions,
    _extend_regions_upward_for_titles,
    collect_figure_pages,
    replace_figure_placeholders,
)


def _make_page_png(width=800, height=600, bg_color=255):
    """Create a simple white PNG image."""
    img = Image.new("L", (width, height), bg_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- _is_blank ----


class TestIsBlank:
    def test_white_image_is_blank(self):
        png = _make_page_png(100, 100, 255)
        assert _is_blank(png)

    def test_dark_image_not_blank(self):
        # A uniform solid color is "blank" (std=0). Use a noisy dark image.
        img = Image.new("L", (100, 100), 50)
        for x in range(0, 100, 3):
            for y in range(0, 100, 3):
                img.putpixel((x, y), 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        assert not _is_blank(buf.getvalue())

    def test_mixed_image_not_blank(self):
        img = Image.new("L", (100, 100), 255)
        for x in range(50):
            img.putpixel((x, 50), 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        assert not _is_blank(buf.getvalue())


# ---- _merge_nearby_regions ----


class TestMergeNearbyRegions:
    def test_no_merge_single_region(self):
        regions = [(0, 100, 200, 300)]
        assert _merge_nearby_regions(regions) == [(0, 100, 200, 300)]

    def test_no_merge_far_apart(self):
        regions = [(0, 100, 200, 300), (600, 100, 800, 300)]
        merged = _merge_nearby_regions(regions, max_x_gap=250)
        assert len(merged) == 2

    def test_merge_adjacent_subfigures(self):
        """Two side-by-side figures with small gap should merge."""
        regions = [(100, 100, 400, 400), (500, 120, 750, 380)]
        merged = _merge_nearby_regions(regions, max_x_gap=250, y_overlap_ratio=0.5)
        assert len(merged) == 1
        assert merged[0] == (100, 100, 750, 400)

    def test_no_merge_vertical_gap(self):
        """Two vertically separated regions should not merge."""
        regions = [(100, 100, 400, 200), (100, 500, 400, 600)]
        merged = _merge_nearby_regions(regions, max_x_gap=250)
        assert len(merged) == 2


# ---- _extend_regions_upward_for_titles ----


class TestExtendRegionsUpward:
    def test_extends_to_include_title(self):
        regions = [(100, 200, 400, 500)]
        text_blocks = [(100, 150, 400, 190)]  # title just above the region
        extended = _extend_regions_upward_for_titles(regions, text_blocks, max_extend=60)
        assert extended[0] == (100, 150, 400, 500)

    def test_no_extend_title_too_far(self):
        regions = [(100, 200, 400, 500)]
        text_blocks = [(100, 100, 400, 130)]  # title 70px above — beyond max_extend
        extended = _extend_regions_upward_for_titles(regions, text_blocks, max_extend=60)
        assert extended[0] == (100, 200, 400, 500)

    def test_no_extend_no_overlap(self):
        regions = [(100, 200, 400, 500)]
        text_blocks = [(500, 150, 700, 190)]  # title doesn't overlap horizontally
        extended = _extend_regions_upward_for_titles(regions, text_blocks)
        assert extended[0] == (100, 200, 400, 500)


# ---- collect_figure_pages ----


class TestCollectFigurePages:
    def test_no_placeholders(self):
        assert collect_figure_pages("# Hello\n\nWorld") == set()

    def test_finds_pages(self):
        md = "text {{FIGURE:3}} more {{FIGURE:5}} text"
        assert collect_figure_pages(md) == {3, 5}

    def test_finds_pages_with_prefix(self):
        md = "text {{FIGURE:page_3}} more {{FIGURE:page_5}} text"
        assert collect_figure_pages(md) == {3, 5}

    def test_deduplicates(self):
        md = "{{FIGURE:3}} {{FIGURE:page_3}} {{FIGURE:3}}"
        assert collect_figure_pages(md) == {3}


# ---- replace_figure_placeholders ----


class TestReplaceFigurePlaceholders:
    def test_no_placeholders(self):
        md = "# Hello\n\nWorld"
        assert replace_figure_placeholders(md, {}) == md

    def test_replaces_placeholder(self):
        imgs = {
            3: [CroppedImage(3, 0, 200, 150, b"", "images/h/page_3_img_0.png")]
        }
        md = "![Transformer]({{FIGURE:3}})"
        result = replace_figure_placeholders(md, imgs)
        assert "{{FIGURE:3}}" not in result
        assert "./images/h/page_3_img_0.png" in result
        assert "![Transformer](" in result  # LLM's alt text preserved

    def test_preserves_unknown_placeholder(self):
        md = "![desc]({{FIGURE:99}})"
        assert replace_figure_placeholders(md, {}) == md

    def test_multiple_figures_same_page(self):
        imgs = {
            5: [
                CroppedImage(5, 0, 200, 150, b"", "a.png"),
                CroppedImage(5, 1, 200, 150, b"", "b.png"),
            ]
        }
        md = "![first]({{FIGURE:5}}) and ![second]({{FIGURE:5}})"
        result = replace_figure_placeholders(md, imgs)
        assert "./a.png" in result
        assert "./b.png" in result
