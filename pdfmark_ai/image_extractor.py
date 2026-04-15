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
    clean_dir: bool = True,
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
    if clean_dir and images_dir.exists():
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

# Pattern for figure captions (standalone, not inline references)
_FIGURE_CAPTION_RE = re.compile(
    r"^\s*\*\*(?:Figure|Fig\.?)\s*\d+[.:][^\n]*$",
    re.MULTILINE,
)

# Pattern for page range annotations
_PAGE_ANNOTATION_RE = re.compile(r"<!--\s*pages:\s*(\d+)-(\d+)\s*-->")


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


def _find_page_for_position(
    position: int,
    coverage: list[tuple[int, int, int, int]],
) -> tuple[int, int] | None:
    """Find (page_start, page_end) for a position using annotation coverage map."""
    for rng_start, rng_end, ps, pe in coverage:
        if rng_start <= position < rng_end:
            return (ps, pe)
    return None


_FIGURE_NUM_RE = re.compile(r"\*\*(?:Figure|Fig\.?)\s*(\d+)")

# Minimum gap (PDF points) between text blocks to consider as the
# body-text / figure boundary.  The largest gap above a caption is
# typically where body text ends and figure labels begin.
_BOUNDARY_GAP = 30


def _crop_vector_figure(
    page_num: int,
    page_image_bytes: bytes,
    pdf_path: Path,
    zoom: float,
    figure_num: int | None = None,
    min_dim: int = MIN_IMAGE_DIM,
) -> bytes | None:
    """Crop a vector-figure region from a rendered page using caption position.

    For pages where figures are vector graphics (not embedded bitmaps),
    locates the ``Figure N:`` caption text block via PyMuPDF, then finds
    the largest gap between text blocks above the caption to determine
    where body text ends and figure content begins.
    """
    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    blocks = page.get_text("blocks")

    # Find matching caption text blocks
    caption_rects: list[tuple[float, float, float, float]] = []
    for b in blocks:
        if b[6] != 0:
            continue
        text = b[4].strip()
        m = re.match(r"Figure\s+\d+", text)
        if not m:
            continue
        if figure_num is not None:
            found_num = int(re.search(r"\d+", text).group())
            if found_num != figure_num:
                continue
        caption_rects.append((b[0], b[1], b[2], b[3]))

    if not caption_rects:
        doc.close()
        return None

    # Collect all text blocks above the caption, sorted by position
    cap_y0 = min(r[1] for r in caption_rects)
    blocks_above = sorted(
        [(b[1], b[3], b[4]) for b in blocks if b[6] == 0 and b[3] <= cap_y0],
        key=lambda t: t[0],
    )

    # Find the top boundary of the figure using the largest gap between
    # consecutive text blocks above the caption.  The largest gap is
    # typically where body text ends and figure labels begin.
    fig_top_pdf = 0.0  # default: top of page content
    if blocks_above:
        gaps: list[tuple[float, float]] = []
        # Gap from page top to the first block
        gaps.append((0.0, blocks_above[0][0]))
        for i in range(1, len(blocks_above)):
            gap_top = blocks_above[i - 1][1]
            gap_bottom = blocks_above[i][0]
            if gap_bottom > gap_top:
                gaps.append((gap_top, gap_bottom))
        # Pick the position just below the largest gap
        max_gap_size = 0.0
        for g_top, g_bottom in gaps:
            if g_bottom - g_top > max_gap_size:
                max_gap_size = g_bottom - g_top
                fig_top_pdf = g_top

    # Bottom boundary: caption top
    fig_bottom_pdf = cap_y0

    doc.close()

    fig_top_px = int(fig_top_pdf * zoom)
    fig_bottom_px = int(fig_bottom_pdf * zoom)
    fig_height = fig_bottom_px - fig_top_px

    if fig_height < min_dim:
        return None

    img = Image.open(io.BytesIO(page_image_bytes))
    page_w = img.size[0]

    cropped = _crop_from_bytes(page_image_bytes, 0, fig_top_px, page_w, fig_height)
    if cropped and not _is_blank(cropped):
        return cropped

    return None


def scan_and_insert_missed_figures(
    markdown: str,
    pdf_path: Path,
    pages: list[PageImage],
    output_dir: Path,
    pdf_hash: str,
    dpi: int = 150,
    existing_page_images: dict[int, list[CroppedImage]] | None = None,
) -> tuple[str, dict[int, list[CroppedImage]]]:
    """Scan for figure captions without image links and insert cropped figures.

    Fallback for when the LLM misses {{FIGURE:N}} placeholders. Finds
    ``**Figure N:**`` captions, determines their PDF page ranges via
    ``<!-- pages: x-y -->`` annotations, crops embedded images, and inserts
    image links before each caption.
    """
    existing = existing_page_images or {}

    # Build page-range coverage from annotations
    annotations = sorted(
        (m.start(), int(m.group(1)), int(m.group(2)))
        for m in _PAGE_ANNOTATION_RE.finditer(markdown)
    )
    coverage: list[tuple[int, int, int, int]] = []
    for i, (pos, ps, pe) in enumerate(annotations):
        end = annotations[i + 1][0] if i + 1 < len(annotations) else len(markdown)
        coverage.append((pos, end, ps, pe))

    if not coverage:
        return markdown, existing

    # Find captions without a preceding image link
    missed: list[tuple[int, int, str, int, int]] = []
    for m in _FIGURE_CAPTION_RE.finditer(markdown):
        cap_start = m.start()
        cap_text = m.group(0).strip()
        preceding = markdown[max(0, cap_start - 500):cap_start]
        lines = [l.strip() for l in preceding.split("\n") if l.strip()]
        has_image = any(
            l.startswith("![") and "](" in l and "images/" in l
            for l in lines[-5:]
        )
        if not has_image:
            rng = _find_page_for_position(cap_start, coverage)
            if rng:
                missed.append((cap_start, m.end(), cap_text, rng[0], rng[1]))

    if not missed:
        return markdown, existing

    # Determine pages to crop (exclude already-cropped pages)
    pages_to_crop: set[int] = set()
    for _, _, _, ps, pe in missed:
        pages_to_crop.update(range(ps, pe + 1))
    pages_to_crop -= set(existing.keys())

    all_images = dict(existing)
    if pages_to_crop:
        new_images = crop_figures_for_pages(
            pdf_path=pdf_path,
            pages=pages,
            page_numbers=pages_to_crop,
            output_dir=output_dir,
            pdf_hash=pdf_hash,
            dpi=dpi,
            clean_dir=False,
        )
        for pn, imgs in new_images.items():
            all_images.setdefault(pn, []).extend(imgs)

    # Phase 2: Vector-graphics fallback — crop from rendered pages using
    # caption position for pages that still have no images
    zoom = dpi / 72.0
    page_byte_map = {p.page_number: p.image_bytes for p in pages}
    images_dir = output_dir / "images" / pdf_hash
    images_dir.mkdir(parents=True, exist_ok=True)

    for cap_start, cap_end, cap_text, ps, pe in missed:
        # Check if we already have an image for any page in range
        if any(all_images.get(p) for p in range(ps, pe + 1)):
            continue

        # Extract figure number for targeted search
        num_match = _FIGURE_NUM_RE.search(cap_text)
        figure_num = int(num_match.group(1)) if num_match else None

        cropped_bytes = None
        for p in range(ps, pe + 1):
            pb = page_byte_map.get(p)
            if pb is None:
                continue
            cropped_bytes = _crop_vector_figure(
                page_num=p,
                page_image_bytes=pb,
                pdf_path=pdf_path,
                zoom=zoom,
                figure_num=figure_num,
            )
            if cropped_bytes:
                save_page = p
                break

        if cropped_bytes:
            idx = len(all_images.get(save_page, []))
            filename = f"page_{save_page}_img_{idx}.png"
            filepath = images_dir / filename
            filepath.write_bytes(cropped_bytes)
            relative_path = f"images/{pdf_hash}/{filename}"
            cropped = CroppedImage(
                page_number=save_page,
                index=idx,
                width=0,
                height=0,
                image_bytes=cropped_bytes,
                relative_path=relative_path,
            )
            all_images.setdefault(save_page, []).append(cropped)

    # Insert image links before captions (reverse to preserve positions)
    page_img_counter: dict[int, int] = {}
    for cap_start, _, cap_text, ps, pe in reversed(missed):
        for p in range(ps, pe + 1):
            imgs = all_images.get(p, [])
            idx = page_img_counter.get(p, 0)
            if idx < len(imgs):
                alt = cap_text.replace("**", "").strip()
                if ". " in alt:
                    alt = alt[:alt.index(". ") + 1]
                img_path = f"./{imgs[idx].relative_path}"
                markdown = markdown[:cap_start] + f"![{alt}]({img_path})\n" + markdown[cap_start:]
                page_img_counter[p] = idx + 1
                break

    new_count = sum(len(v) for v in all_images.values()) - sum(len(v) for v in existing.values())
    logger.info(
        "Fallback scan: cropped %d images for %d missed captions",
        new_count, len(missed),
    )
    return markdown, all_images
