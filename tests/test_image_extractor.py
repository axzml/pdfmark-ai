"""Tests for pixel-scan image extraction and placeholder resolution."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from PIL import Image
import io

from pdfmark_ai.image_extractor import (
    CroppedImage,
    build_image_manifest,
    resolve_image_placeholders,
    _find_visual_regions,
    _is_blank,
    _build_text_mask,
)


def _make_page_png(width=800, height=600, bg_color=255):
    """Create a simple white PNG image."""
    img = Image.new("L", (width, height), bg_color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_page_with_dark_band(width=800, height=600, band_y=200, band_h=100):
    """Create a page with a dark horizontal band (simulating a figure)."""
    img = Image.new("L", (width, height), 255)
    for y in range(band_y, band_y + band_h):
        for x in range(width):
            img.putpixel((x, y), 50)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- _find_visual_regions ----


class TestFindVisualRegions:
    def test_no_regions_on_blank_page(self):
        png = _make_page_png(800, 600)
        blocks = [(50, 50, 750, 150)]  # one text block
        regions = _find_visual_regions(png, blocks)
        assert regions == []

    def test_detects_dark_band_below_text(self):
        png = _make_page_with_dark_band(800, 600, band_y=200, band_h=100)
        blocks = [(50, 50, 750, 150)]  # text block at top
        regions = _find_visual_regions(png, blocks)
        assert len(regions) >= 1
        # Region should be around y=200
        _, y0, _, y1 = regions[0]
        assert y0 <= 200
        # SCAN_STEP=2 means run_end is up to 298, not 300
        assert y1 >= 298

    def test_no_detection_inside_text_block(self):
        png = _make_page_with_dark_band(800, 600, band_y=60, band_h=80)
        blocks = [(50, 50, 750, 150)]  # text block covers the band
        regions = _find_visual_regions(png, blocks)
        # The band is inside the text block area, should not be detected
        for _, y0, _, y1 in regions:
            assert y0 >= 150 or y1 <= 50

    def test_multiple_bands(self):
        img = Image.new("L", (800, 800), 255)
        # Two dark bands
        for y in range(200, 280):
            for x in range(800):
                img.putpixel((x, y), 50)
        for y in range(500, 580):
            for x in range(800):
                img.putpixel((x, y), 50)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png = buf.getvalue()

        blocks = [(50, 50, 750, 150), (50, 600, 750, 750)]
        regions = _find_visual_regions(png, blocks)
        assert len(regions) >= 2


# ---- _build_text_mask ----


class TestBuildTextMask:
    def test_covers_text_rows(self):
        mask = _build_text_mask([(100, 50, 700, 150)], 200)
        assert all(mask[y] for y in range(50, 150))
        assert not any(mask[y] for y in range(0, 46))

    def test_buffer_around_blocks(self):
        mask = _build_text_mask([(100, 50, 700, 150)], 200)
        assert mask[46]  # buffer before (50 - TEXT_BUFFER=4 = 46)
        assert mask[153]  # buffer after (150 + TEXT_BUFFER-1 = 153)
        assert not mask[45]


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


# ---- build_image_manifest ----


class TestBuildImageManifest:
    def test_no_images(self):
        assert build_image_manifest({}, [1, 2, 3]) == ""

    def test_with_images(self):
        imgs = {
            3: [CroppedImage(3, 0, 200, 150, b"", "img.png")]
        }
        result = build_image_manifest(imgs, [3])
        assert "{{IMG:3:0}}" in result
        assert "Page 3 has 1 figure(s)" in result


# ---- resolve_image_placeholders ----


class TestResolveImagePlaceholders:
    def test_no_placeholders(self):
        md = "# Hello\n\nWorld"
        assert resolve_image_placeholders(md, {}) == md

    def test_replaces_placeholder(self):
        imgs = {
            3: [CroppedImage(3, 0, 200, 150, b"", "images/h/page_3_img_0.png")]
        }
        md = "See figure:\n\n{{IMG:3:0}}"
        result = resolve_image_placeholders(md, imgs)
        assert "{{IMG:3:0}}" not in result
        assert "./images/h/page_3_img_0.png" in result
        assert "![" not in result  # resolver returns URL only, no ![image] wrapper

    def test_preserves_unknown_placeholder(self):
        md = "{{IMG:99:0}}"
        assert resolve_image_placeholders(md, {}) == md

    def test_out_of_range_preserved(self):
        md = "{{IMG:3:5}}"
        assert resolve_image_placeholders(md, {3: []}) == md

    def test_bare_page_reference(self):
        imgs = {
            5: [
                CroppedImage(5, 0, 200, 150, b"", "a.png"),
                CroppedImage(5, 1, 200, 150, b"", "b.png"),
            ]
        }
        md = "{{IMG:5}}"
        result = resolve_image_placeholders(md, imgs)
        assert "./a.png" in result
        assert "./b.png" in result
