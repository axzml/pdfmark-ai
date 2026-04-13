"""Crop-based image extraction: detect visual (non-text) regions on rendered pages."""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from pdfmark_ai.models import PageImage

logger = logging.getLogger(__name__)

# Minimum cropped image dimension (pixels) to keep.
MIN_IMAGE_DIM = 80

# Minimum height (pixels) of a visual region to consider.
MIN_REGION_HEIGHT = 60

# Pixel brightness threshold: pixels above this are considered "white/blank".
WHITE_THRESHOLD = 245

# How many rows to skip between checks (performance).
# Every row is more accurate but slower; 2 is a good balance.
SCAN_STEP = 2

# Buffer (pixels) around text blocks to avoid catching ascenders/descenders.
TEXT_BUFFER = 4

# Standard deviation threshold for blank detection (grayscale).
BLANK_STD_THRESHOLD = 5.0


@dataclass
class CroppedImage:
    """An image cropped from a rendered PDF page."""

    page_number: int  # 1-indexed
    index: int  # 0-based index within the page
    width: int  # pixel width after crop
    height: int  # pixel height after crop
    image_bytes: bytes  # PNG bytes
    relative_path: str  # path relative to output markdown file


def _build_text_mask(
    text_blocks_pixel: list[tuple[int, int, int, int]],
    img_height: int,
) -> list[bool]:
    """Build a per-row boolean mask: True if the row overlaps with a text block."""
    mask = [False] * img_height
    for x0, y0, x1, y1 in text_blocks_pixel:
        row_start = max(0, y0 - TEXT_BUFFER)
        row_end = min(img_height, y1 + TEXT_BUFFER)
        for y in range(row_start, row_end):
            mask[y] = True
    return mask


def _find_visual_regions(
    png_bytes: bytes,
    text_blocks_pixel: list[tuple[int, int, int, int]],
    min_height: int = MIN_REGION_HEIGHT,
) -> list[tuple[int, int, int, int]]:
    """Scan rendered page rows to find non-text, non-white visual regions.

    Uses pixel-level analysis: rows that don't overlap text blocks AND
    contain non-white pixels are considered visual content (figures, charts,
    colored boxes, etc.).

    Returns list of (x0, y0, x1, y1) bounding boxes in pixel coordinates.
    """
    img = Image.open(io.BytesIO(png_bytes)).convert("L")
    width, height = img.size

    if not text_blocks_pixel:
        return []

    # Build text mask
    text_mask = _build_text_mask(text_blocks_pixel, height)

    # Get all pixel data at once for speed
    pixel_data = img.getdata()

    # Scan rows: find "content rows" (non-text AND non-white)
    content_rows: list[int] = []
    for y in range(0, height, SCAN_STEP):
        if text_mask[y]:
            continue
        # Sample pixels across the row (every 8th pixel for speed)
        row_start = y * width
        dark_count = 0
        total = 0
        for x in range(0, width, 8):
            brightness = pixel_data[row_start + x]
            if brightness < WHITE_THRESHOLD:
                dark_count += 1
            total += 1
        # If more than 5% of sampled pixels are non-white, it's content
        if total > 0 and dark_count / total > 0.05:
            content_rows.append(y)

    if not content_rows:
        return []

    # Find contiguous runs of content rows
    regions: list[tuple[int, int, int, int]] = []
    run_start = content_rows[0]
    run_end = content_rows[0]

    for y in content_rows[1:]:
        # Allow small gaps (≤ 6 pixels) within a run
        if y - run_end <= 6:
            run_end = y
        else:
            h = run_end - run_start
            if h >= min_height:
                regions.append((0, run_start, width, run_end))
            run_start = y
            run_end = y

    h = run_end - run_start
    if h >= min_height:
        regions.append((0, run_start, width, run_end))

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
        return std < BLANK_STD_THRESHOLD
    except Exception:
        return False


def extract_images(
    pdf_path: Path,
    pages: list[PageImage],
    output_dir: Path,
    pdf_hash: str,
    dpi: int = 150,
    min_dim: int = MIN_IMAGE_DIM,
) -> dict[int, list[CroppedImage]]:
    """Detect and crop visual (non-text) regions from rendered PDF pages.

    Uses pixel-level scanning: renders the page, builds a text block mask,
    then finds rows that are both outside text areas AND contain non-white
    pixels. These regions are cropped and saved as PNG images.

    This approach works with any visual content: embedded bitmaps, vector
    graphics, colored boxes, charts, screenshots, etc.

    Args:
        pdf_path: Path to the source PDF (for text block positions).
        pages: List of rendered PageImage objects.
        output_dir: Directory where output markdown will be written.
            Images saved to {output_dir}/images/{pdf_hash}/.
        pdf_hash: Hash string for directory naming.
        dpi: DPI used for rendering (for coordinate conversion).
        min_dim: Minimum cropped image dimension in pixels.

    Returns:
        Dict mapping page_number (1-indexed) to list of CroppedImage.
    """
    doc = fitz.open(str(pdf_path))
    zoom = dpi / 72.0
    page_images: dict[int, list[CroppedImage]] = {}

    images_dir = output_dir / "images" / pdf_hash
    images_dir.mkdir(parents=True, exist_ok=True)

    for page_img in pages:
        page_num = page_img.page_number
        page = doc[page_num - 1]

        # Get text block bounding boxes in pixel coordinates
        blocks = page.get_text("blocks")
        text_blocks_pixel = []
        for b in blocks:
            if b[6] == 0:  # text block
                x0 = int(b[0] * zoom)
                y0 = int(b[1] * zoom)
                x1 = int(b[2] * zoom)
                y1 = int(b[3] * zoom)
                text_blocks_pixel.append((x0, y0, x1, y1))

        if not text_blocks_pixel:
            continue

        # Find visual regions using pixel scanning
        regions = _find_visual_regions(
            page_img.image_bytes,
            text_blocks_pixel,
        )

        if not regions:
            continue

        # Crop and save each region
        img_idx = 0
        for rx0, ry0, rx1, ry1 in regions:
            pw = rx1 - rx0
            ph = ry1 - ry0

            if pw < min_dim or ph < min_dim:
                continue

            cropped_bytes = _crop_from_bytes(page_img.image_bytes, rx0, ry0, pw, ph)
            if cropped_bytes is None:
                continue

            # Skip blank regions
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


def build_image_manifest(
    page_images: dict[int, list[CroppedImage]], pages: list[int]
) -> str:
    """Build a text manifest of available cropped images for LLM prompts.

    Args:
        page_images: Per-page image index from extract_images().
        pages: List of page numbers (1-indexed) in the current chunk.

    Returns:
        Multi-line string describing available images, or empty string.
    """
    lines = []
    for page_num in pages:
        imgs = page_images.get(page_num, [])
        if imgs:
            placeholders = ", ".join(
                f"{{{{IMG:{page_num}:{img.index}}}}}" for img in imgs
            )
            lines.append(f"Page {page_num} has {len(imgs)} figure(s): {placeholders}")

    if not lines:
        return ""

    return (
        "\n--- Available Figures ---\n"
        "The following figures were extracted from the rendered pages.\n"
        "When you see a figure/chart/diagram in the page, use the "
        "corresponding placeholder: ![brief description]({{IMG:page_N:M}})\n"
        + "\n".join(lines)
        + "\n--- End Available Figures ---"
    )


def resolve_image_placeholders(
    markdown: str,
    page_images: dict[int, list[CroppedImage]],
) -> str:
    """Replace {{IMG:page_N:M}} placeholders with actual image paths.

    Args:
        markdown: Markdown text with placeholders.
        page_images: Per-page image index.

    Returns:
        Markdown with placeholders replaced by relative image paths.
    """
    import re

    def _replace(match: re.Match) -> str:
        page = int(match.group(1))
        idx_str = match.group(2)
        imgs = page_images.get(page, [])

        if idx_str is not None:
            idx = int(idx_str)
            if idx < len(imgs):
                return f"./{imgs[idx].relative_path}"
            return match.group(0)
        else:
            if not imgs:
                return match.group(0)
            return "\n".join(
                f"./{img.relative_path}" for img in imgs
            )

    pattern = r"\{\{IMG:(\d+)(?::(\d+))?\}\}"
    return re.sub(pattern, _replace, markdown)
