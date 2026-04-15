"""Post-extraction figure cropping using PyMuPDF embedded image detection."""

from __future__ import annotations

import io
import logging
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from pdfmark_ai.models import PageImage

logger = logging.getLogger(__name__)

# Minimum cropped image dimension (pixels) to keep.
MIN_IMAGE_DIM = 80


@dataclass
class CroppedImage:
    """An image cropped from a rendered PDF page."""

    page_number: int  # 1-indexed
    index: int  # 0-based index within the page
    width: int  # pixel width after crop
    height: int  # pixel height after crop
    image_bytes: bytes  # PNG bytes
    relative_path: str  # path relative to output markdown file


def _merge_nearby_regions(
    regions: list[tuple[int, int, int, int]],
    max_x_gap: int = 250,
    y_overlap_ratio: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """Merge horizontally adjacent regions that are likely parts of the same figure.

    Academic papers often store multi-part figures (e.g. Figure 2a and 2b) as
    separate embedded images. When two images are side-by-side with similar
    vertical spans and a small horizontal gap, they should be treated as one figure.
    """
    if len(regions) <= 1:
        return regions

    sorted_regions = sorted(regions, key=lambda r: r[0])
    merged: list[tuple[int, int, int, int]] = list(sorted_regions)

    changed = True
    while changed:
        changed = False
        new_merged: list[tuple[int, int, int, int]] = []
        used = [False] * len(merged)

        for i in range(len(merged)):
            if used[i]:
                continue
            rx0, ry0, rx1, ry1 = merged[i]

            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                sx0, sy0, sx1, sy1 = merged[j]

                x_gap = sx0 - rx1
                if x_gap > max_x_gap or x_gap < 0:
                    continue

                y_overlap = max(0, min(ry1, sy1) - max(ry0, sy0))
                shorter_h = min(ry1 - ry0, sy1 - sy0)
                if shorter_h == 0:
                    continue

                if y_overlap / shorter_h >= y_overlap_ratio:
                    rx0 = min(rx0, sx0)
                    ry0 = min(ry0, sy0)
                    rx1 = max(rx1, sx1)
                    ry1 = max(ry1, sy1)
                    used[j] = True
                    changed = True

            new_merged.append((rx0, ry0, rx1, ry1))
            used[i] = True

        merged = new_merged

    return merged


def _extend_regions_upward_for_titles(
    regions: list[tuple[int, int, int, int]],
    text_blocks_pixel: list[tuple[int, int, int, int]],
    max_extend: int = 60,
) -> list[tuple[int, int, int, int]]:
    """Extend embedded image regions upward to include nearby title text.

    In academic papers, figure titles (e.g. "Scaled Dot-Product Attention") are
    rendered as text blocks above the embedded image, not baked into the image.
    """
    if not text_blocks_pixel or not regions:
        return regions

    extended: list[tuple[int, int, int, int]] = []
    for rx0, ry0, rx1, ry1 in regions:
        for tx0, ty0, tx1, ty1 in text_blocks_pixel:
            if ty1 > ry0:
                continue
            if ry0 - ty1 > max_extend:
                continue
            if tx1 <= rx0 or tx0 >= rx1:
                continue
            ry0 = min(ry0, ty0)
            rx0 = min(rx0, tx0)
            rx1 = max(rx1, tx1)

        extended.append((rx0, ry0, rx1, ry1))
    return extended


def _find_embedded_image_regions(
    page: "fitz.Page",
    zoom: float,
    min_dim: int = MIN_IMAGE_DIM,
    text_blocks_pixel: list[tuple[int, int, int, int]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Get bounding boxes of embedded images from PyMuPDF, in pixel coordinates.

    Merges horizontally adjacent sub-figures and extends regions to include
    title text above images.
    """
    regions: list[tuple[int, int, int, int]] = []
    try:
        image_list = page.get_images()
        for img_info in image_list:
            xref = img_info[0]
            try:
                rects = page.get_image_rects(xref)
                for rect in rects:
                    x0 = int(rect.x0 * zoom)
                    y0 = int(rect.y0 * zoom)
                    x1 = int(rect.x1 * zoom)
                    y1 = int(rect.y1 * zoom)
                    pw = x1 - x0
                    ph = y1 - y0
                    if pw >= min_dim and ph >= min_dim:
                        regions.append((x0, y0, x1, y1))
            except Exception:
                pass
    except Exception:
        pass

    regions = _merge_nearby_regions(regions)
    if text_blocks_pixel:
        regions = _extend_regions_upward_for_titles(regions, text_blocks_pixel)

    return regions


def _crop_from_bytes(
    png_bytes: bytes,
    x0: int, y0: int,
    width: int, height: int,
) -> bytes | None:
    """Crop a region from PNG bytes. Returns PNG bytes or None on error."""
    try:
        img = Image.open(io.BytesIO(png_bytes))
        iw, ih = img.size
        x0 = max(0, min(x0, iw - 1))
        y0 = max(0, min(y0, ih - 1))
        x1 = min(x0 + width, iw)
        y1 = min(y0 + height, ih)
        if x1 <= x0 or y1 <= y0:
            return None
        cropped = img.crop((x0, y0, x1, y1))
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        logger.warning("Failed to crop image: %s", e)
        return None


def _is_blank(png_bytes: bytes) -> bool:
    """Check if an image is blank (uniform color / no visual content)."""
    try:
        img = Image.open(io.BytesIO(png_bytes)).convert("L")
        pixels = list(img.getdata())
        if not pixels:
            return True
        mean = sum(pixels) / len(pixels)
        variance = sum((p - mean) ** 2 for p in pixels) / len(pixels)
        std = math.sqrt(variance)
        return std < 5.0
    except Exception:
        return False


def crop_figures_for_pages(
    pdf_path: Path,
    pages: list[PageImage],
    page_numbers: set[int],
    output_dir: Path,
    pdf_hash: str,
    dpi: int = 150,
    min_dim: int = MIN_IMAGE_DIM,
) -> dict[int, list[CroppedImage]]:
    """Crop figures from specific pages using PyMuPDF embedded image detection.

    Uses PyMuPDF's ``page.get_images()`` and ``page.get_image_rects()`` for
    precise bounding boxes.  Nearby sub-figures are merged, and title text
    above images is included in the crop.

    Args:
        pdf_path: Path to the source PDF (for text block positions).
        pages: List of rendered PageImage objects.
        page_numbers: Set of 1-indexed page numbers to crop from.
        output_dir: Directory where the output markdown will be written.
            Images saved to {output_dir}/images/{pdf_hash}/.
        pdf_hash: Hash string for directory naming.
        dpi: DPI used for rendering (for coordinate conversion).
        min_dim: Minimum cropped image dimension in pixels.

    Returns:
        Dict mapping page_number (1-indexed) to list of CroppedImage.
    """
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    page_map = {p.page_number: p for p in pages}
    page_images: dict[int, list[CroppedImage]] = {}

    images_dir = output_dir / "images" / pdf_hash
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_num in sorted(page_numbers):
        page_img = page_map.get(page_num)
        if page_img is None:
            continue

        page = doc[page_num - 1]

        # Get text block bounding boxes for title extension
        blocks = page.get_text("blocks")
        text_blocks_pixel: list[tuple[int, int, int, int]] = []
        for b in blocks:
            if b[6] == 0:  # text block
                text_blocks_pixel.append((
                    int(b[0] * zoom), int(b[1] * zoom),
                    int(b[2] * zoom), int(b[3] * zoom),
                ))

        # Find and merge embedded image regions
        regions = _find_embedded_image_regions(page, zoom, min_dim, text_blocks_pixel)

        # Crop each region
        img_idx = 0
        for rx0, ry0, rx1, ry1 in regions:
            pw = rx1 - rx0
            ph = ry1 - ry0

            if pw < min_dim or ph < min_dim:
                continue

            cropped_bytes = _crop_from_bytes(page_img.image_bytes, rx0, ry0, pw, ph)
            if cropped_bytes is None:
                continue

            if _is_blank(cropped_bytes):
                logger.debug(
                    "Skipping blank region on page %d: %dx%d at (%d,%d)",
                    page_num, pw, ph, rx0, ry0,
                )
                continue

            filename = f"page_{page_num}_img_{img_idx}.png"
            filepath = images_dir / filename
            filepath.write_bytes(cropped_bytes)

            relative_path = f"images/{pdf_hash}/{filename}"

            cropped = CroppedImage(
                page_number=page_num,
                index=img_idx,
                width=pw,
                height=ph,
                image_bytes=cropped_bytes,
                relative_path=relative_path,
            )
            page_images.setdefault(page_num, []).append(cropped)
            img_idx += 1

    doc.close()

    total = sum(len(imgs) for imgs in page_images.values())
    logger.info("Cropped %d figure regions from %d pages", total, len(page_images))
    return page_images


FIGURE_PATTERN = re.compile(r"\{\{FIGURE:page_(\d+)\}\}|\{\{FIGURE:(\d+)\}\}")
FIGURE_IMG_PATTERN = re.compile(r"!\[[^\]]*\]\(\{\{FIGURE:page_\d+\}\}\)|!\[[^\]]*\]\(\{\{FIGURE:\d+\}\}\)")


def collect_figure_pages(markdown: str) -> set[int]:
    """Parse markdown for {{FIGURE:N}} placeholders and return unique page numbers."""
    return {int(g) for m in FIGURE_PATTERN.findall(markdown) for g in m if g}


def replace_figure_placeholders(
    markdown: str,
    page_images: dict[int, list[CroppedImage]],
) -> str:
    """Replace {{FIGURE:page_N}} placeholders with actual image paths.

    Uses a per-page counter so that if a page has multiple figures,
    the first placeholder gets img_0, the second gets img_1, etc.
    Returns only the URL path — the LLM already provides the ![desc]() wrapper.
    """
    page_counter: dict[int, int] = {}

    def _replace(match: re.Match) -> str:
        # group(1) = page_ prefix form, group(2) = bare number form
        page = int(match.group(1) or match.group(2))
        imgs = page_images.get(page, [])
        if not imgs:
            return match.group(0)
        idx = page_counter.get(page, 0)
        if idx < len(imgs):
            page_counter[page] = idx + 1
            return f"./{imgs[idx].relative_path}"
        return match.group(0)

    return FIGURE_PATTERN.sub(_replace, markdown)
