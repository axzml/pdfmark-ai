"""CLI entry point for pdfmark -- PDF to Markdown converter."""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

# Fix Unicode output on Windows — GBK console can't render Rich's braille chars
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetConsoleOutputCP(65001)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from pdfmark_ai.config import load_config
from pdfmark_ai.renderer import render_pdf, get_pdf_hash
from pdfmark_ai.detector import detect_structure
from pdfmark_ai.chunker import build_chunks
from pdfmark_ai.extractor import process_all_chunks
from pdfmark_ai.merger import merge_results
from pdfmark_ai.refiner import refine
from pdfmark_ai.client import LLMClient
from pdfmark_ai.image_extractor import (
    FIGURE_IMG_PATTERN,
    FIGURE_PATTERN,
    collect_figure_pages,
    crop_figures_for_pages,
    replace_figure_placeholders,
    scan_and_insert_missed_figures,
)

app = typer.Typer(
    name="pdfmark",
    help="Convert PDF files to high-quality Markdown using LLM vision models.",
    no_args_is_help=False,
)
console = Console(legacy_windows=False)


def _setup_logging(debug: bool) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _get_cache_subdir(
    cache_dir: str, pdf_path: Path, stage: str
) -> Path | None:
    """Return a stage-specific cache subdirectory for a given PDF."""
    base = Path(cache_dir)
    pdf_hash = get_pdf_hash(pdf_path)
    return base / pdf_hash / stage


_ENV_TEMPLATE = """\
# pdfmark - LLM Configuration
# ===================================
# Fill in your API key below. The active provider is set in pdfmark.toml.

# --- Kimi (kimi-for-coding) ---
# LLM_API_KEY=your-kimi-api-key

# --- Xiaomi (mimo-v2-omni) ---
# LLM_API_KEY=your-xiaomi-api-key

# --- Anthropic (Claude) ---
# LLM_API_KEY=your-anthropic-api-key

# --- Qwen ---
# LLM_API_KEY=your-qwen-api-key
# LLM_SDK_TYPE=openai

# Override provider selection from TOML (optional)
# LLM_BASE_URL=https://api.your-provider.com/v1
# LLM_MODEL=your-model-name
"""

_TOML_TEMPLATE = """\
# pdfmark Configuration
# =====================
# Non-sensitive defaults (safe to commit).
# Secrets (API keys) go in .env file.

# Active provider preset: "kimi" (default), "xiaomi", "qwen", or "anthropic"
# This selects [providers.<name>] as the default LLM settings.
# Environment variables (LLM_API_KEY, LLM_BASE_URL, LLM_MODEL) override these.
active_provider = "kimi"

# --- Provider presets ---

[providers.anthropic]
base_url = "https://api.anthropic.com"
model = "claude-sonnet-4-20250514"
auth_type = "api_key"
sdk_type = "anthropic"
timeout = 300
max_concurrent = 5

[providers.kimi]
base_url = "https://api.kimi.com/coding"
model = "kimi-for-coding"
auth_type = "api_key"
sdk_type = "anthropic"
timeout = 300
max_concurrent = 5

[providers.xiaomi]
base_url = "https://token-plan-cn.xiaomimimo.com/anthropic"
model = "mimo-v2-omni"
auth_type = "auth_token"
sdk_type = "anthropic"
timeout = 300
max_concurrent = 5

[providers.qwen]
base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
model = "qwen3.6-plus"
sdk_type = "openai"
timeout = 300
max_concurrent = 3

# --- Rendering ---

[render]
dpi = 150

# --- Chunking ---

[chunk]
max_pages = 4
window_size = 2
window_overlap = 1

# --- Output ---

[output]
language = "auto"
refine = false
frontmatter = true

# --- Cache ---

[cache]
enabled = true
dir = "~/.cache/pdfmark"
"""


@app.command()
def main(
    input: Optional[Path] = typer.Argument(
        None, help="Path to the PDF file to convert.",
    ),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", help="Output markdown file path.",
    ),
    lang: str = typer.Option(
        "auto", "--lang", help="Document language (e.g. 'en', 'zh', 'auto').",
    ),
    init: bool = typer.Option(
        False, "--init", help="Generate .env and pdfmark.toml config templates in the current directory.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite existing config files (use with --init).",
    ),
    refine: bool = typer.Option(
        False, "--refine", help="Run optional LLM global refinement pass.",
    ),
    no_cache: bool = typer.Option(
        False, "--no-cache", help="Disable caching of rendered pages and chunks.",
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Enable debug-level logging.",
    ),
    detect_only: bool = typer.Option(
        False, "--detect-only", help="Detect document structure and print sections, then exit.",
    ),
    config: Optional[str] = typer.Option(
        None, "--config", help="Path to a TOML configuration file.",
    ),
    dpi: Optional[int] = typer.Option(
        None, "--dpi", help="Rendering DPI for PDF pages.",
    ),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM model identifier.",
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", help="LLM API key (or set LLM_API_KEY env var).",
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", help="LLM API base URL.",
    ),
    max_concurrent: Optional[int] = typer.Option(
        None, "--max-concurrent", help="Maximum concurrent LLM requests.",
    ),
    no_frontmatter: bool = typer.Option(
        False, "--no-frontmatter", help="Omit YAML frontmatter from output.",
    ),
    crop_images: bool = typer.Option(
        False, "--crop-images", help="Extract visual (non-text) regions from pages as images.",
    ),
) -> None:
    """Convert PDF files to high-quality Markdown using LLM vision models."""

    # Handle --init: generate config templates and exit
    if init:
        cwd = Path.cwd()
        env_path = cwd / ".env"
        toml_path = cwd / "pdfmark.toml"
        env_created = toml_created = False

        if env_path.exists() and not force:
            console.print(f"  [yellow]Skipped[/yellow] .env (already exists)")
        else:
            env_path.write_text(_ENV_TEMPLATE, encoding="utf-8")
            console.print(f"  [green]Created[/green] .env")
            env_created = True

        if toml_path.exists() and not force:
            console.print(f"  [yellow]Skipped[/yellow] pdfmark.toml (already exists)")
        else:
            toml_path.write_text(_TOML_TEMPLATE, encoding="utf-8")
            console.print(f"  [green]Created[/green] pdfmark.toml")
            toml_created = True

        if env_created or toml_created:
            console.print(
                "\n[bold]Next steps:[/bold]\n"
                "  1. Edit [bold].env[/bold] and add your API key\n"
                "  2. Optionally edit [bold]pdfmark.toml[/bold] to change the provider\n"
                "  3. Run: [bold]pdfmark input.pdf -o output.md[/bold]"
            )
        else:
            console.print(
                "\nConfig files already exist. "
                "Use [bold]--force[/bold] to overwrite them."
            )
        return

    if input is None:
        console.print(
            "Usage: pdfmark [OPTIONS] INPUT\n"
            "Try 'pdfmark --help' for more information."
        )
        raise typer.Exit(1)
    _setup_logging(debug)

    # Build CLI args dict for config loading
    cli_args: dict = {
        "language": lang,
        "refine": refine,
        "no_cache": no_cache,
        "no_frontmatter": no_frontmatter,
        "crop_images": crop_images,
    }
    if config is not None:
        cli_args["config"] = config
    if dpi is not None:
        cli_args["dpi"] = dpi
    if model is not None:
        cli_args["model"] = model
    if api_key is not None:
        cli_args["api_key"] = api_key
    if base_url is not None:
        cli_args["base_url"] = base_url
    if max_concurrent is not None:
        cli_args["max_concurrent"] = max_concurrent

    try:
        asyncio.run(_run(input, output, cli_args, detect_only))
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Cancelled.[/bold yellow]")
        sys.exit(130)


async def _run(
    input_path: Path,
    output_path: Optional[Path],
    cli_args: dict,
    detect_only: bool,
) -> None:
    """Orchestrate the full PDF-to-Markdown pipeline."""
    # Load configuration
    cfg = load_config(cli_args)

    # Check API key is set (not needed for --help which exits earlier via typer)
    if not cfg["api_key"]:
        console.print(
            "[bold red]Error:[/bold red] No API key set. "
            "Use --api-key or set LLM_API_KEY environment variable."
        )
        raise typer.Exit(1)

    # Check input file exists
    if not input_path.exists():
        console.print(
            f"[bold red]Error:[/bold red] File not found: {input_path}"
        )
        raise typer.Exit(1)

    # Resolve output path
    if output_path is None:
        output_path = input_path.with_suffix(".md")

    # Cache setup
    cache_enabled = cfg["cache_enabled"]
    cache_dir: Path | None = None
    if cache_enabled:
        cache_dir = Path(cfg["cache_dir"])
        cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM client
    client = LLMClient(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        max_concurrent=cfg["max_concurrent"],
        timeout=cfg["timeout"],
        auth_type=cfg.get("auth_type", "api_key"),
        sdk_type=cfg.get("sdk_type", "anthropic"),
        request_delay=cfg.get("request_delay", 0),
    )

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with progress:
        # Stage 1: Render PDF pages
        task_render = progress.add_task("Rendering PDF pages...", total=None)
        pages = render_pdf(
            pdf_path=input_path,
            cache_dir=cache_dir,
            dpi=cfg["dpi"],
        )
        progress.update(task_render, completed=True, total=1)
        console.print(f"  Rendered [bold green]{len(pages)}[/bold green] pages.")

        # Stage 2: Detect structure
        progress.add_task("Detecting document structure...", total=None)
        structure = await detect_structure(
            pdf_path=input_path,
            pages=pages,
            client=client,
            language=cfg["language"],
        )
        console.print(
            f"  Structure: [bold]{structure.source}[/bold], "
            f"{len(structure.sections)} sections detected."
        )

        # If detect-only, print sections and exit
        if detect_only:
            console.print("\n[bold]Detected sections:[/bold]")
            if structure.sections:
                for sec in structure.sections:
                    console.print(
                        f"  - {sec.title} (pages {sec.start_page}-{sec.end_page})"
                    )
            else:
                console.print("  (no sections detected -- will use sliding window)")
            return

        # Stage 3: Build chunks
        progress.add_task("Building chunks...", total=None)
        chunks = build_chunks(
            structure=structure,
            pages=pages,
            max_pages=cfg["max_pages"],
            window_size=cfg["window_size"],
            window_overlap=cfg["window_overlap"],
        )
        console.print(f"  Created [bold green]{len(chunks)}[/bold green] chunks.")

        # Stage 4: Process chunks (LLM extraction)
        extraction_cache: Path | None = None
        if cache_dir is not None:
            extraction_cache = _get_cache_subdir(
                str(cache_dir), input_path, "chunks"
            )

        task_extract = progress.add_task(
            f"Extracting markdown (0/{len(chunks)})...", total=len(chunks)
        )

        # Wrap process_all_chunks to update progress
        results = await process_all_chunks(
            chunks=chunks,
            client=client,
            structure=structure,
            max_concurrency=cfg["max_concurrent"],
            cache_dir=extraction_cache,
            crop_mode=bool(cli_args.get("crop_images", False)),
        )

        progress.update(task_extract, completed=len(chunks))
        console.print(
            f"  Extracted [bold green]{len(results)}[/bold green] chunks."
        )

        # Stage 5: Merge results
        progress.add_task("Merging results...", total=None)
        markdown = merge_results(
            results=results,
            structure=structure,
            source_pdf=input_path,
            frontmatter=cfg["frontmatter"],
        )

        # Stage 5.5: Crop figures if --crop_images
        if cli_args.get("crop_images", False):
            output_dir = output_path.parent
            pdf_hash = get_pdf_hash(input_path)

            # Phase 1: LLM-marked figures
            figure_pages = collect_figure_pages(markdown)
            page_images: dict = {}
            if figure_pages:
                progress.add_task("Cropping LLM-marked figures...", total=None)
                page_images = crop_figures_for_pages(
                    pdf_path=input_path,
                    pages=pages,
                    page_numbers=figure_pages,
                    output_dir=output_dir,
                    pdf_hash=pdf_hash,
                    dpi=cfg["dpi"],
                )
                markdown = replace_figure_placeholders(markdown, page_images)

            # Strip leftover placeholder syntax
            markdown = FIGURE_IMG_PATTERN.sub("", markdown)

            # Phase 2: Fallback for missed figure captions
            progress.add_task("Scanning for missed figures...", total=None)
            markdown, all_images = scan_and_insert_missed_figures(
                markdown=markdown,
                pdf_path=input_path,
                pages=pages,
                output_dir=output_dir,
                pdf_hash=pdf_hash,
                dpi=cfg["dpi"],
                existing_page_images=page_images,
            )

            total_imgs = sum(len(imgs) for imgs in all_images.values())
            fallback_count = total_imgs - sum(len(imgs) for imgs in page_images.values())

            console.print(
                f"  Cropped [bold green]{total_imgs}[/bold green] figure regions."
                if total_imgs
                else "  No figures detected."
            )
            if fallback_count > 0:
                console.print(
                    f"  (fallback recovered [bold green]{fallback_count}[/bold green] missed)"
                )
        else:
            # Strip any ![...](\{\{FIGURE:N\}\}) placeholders when crop mode is off
            markdown = FIGURE_IMG_PATTERN.sub("", markdown)

        # Stage 6: Optional refinement
        if cfg["refine"]:
            progress.add_task("Refining with LLM...", total=None)
            markdown = await refine(results, client, structure)
            console.print("  [bold green]Refinement complete.[/bold green]")

        # Stage 7: Write output
        progress.add_task("Writing output...", total=None)
        output_path.write_text(markdown, encoding="utf-8")
        console.print(f"\n[bold green]Done![/bold green] Output: {output_path}")
