# pdfmark-ai

<p align="center">
  <strong>PDF to Markdown, powered by LLM vision.</strong>
</p>

<p align="center">
  Drop a PDF, get clean Markdown — tables, formulas, code, figures, all handled.
</p>

<p align="center">
  <img alt="PyPI version" src="https://img.shields.io/pypi/v/pdfmark-ai?color=blue&kill_cache=1"/>
  <img alt="Python" src="https://img.shields.io/pypi/pyversions/pdfmark-ai?kill_cache=1"/>
  <img alt="License" src="https://img.shields.io/pypi/l/pdfmark-ai?kill_cache=1"/>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/axzml/pdfmark-ai"/>
</p>

<p align="center">
  <a href="#demo">Demo</a> ·
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#configuration">Configuration</a> ·
  <a href="#cli-reference">CLI Reference</a>
</p>

<p align="center">
  English | <a href="README_ZH.md">简体中文</a>
</p>

pdfmark-ai doesn't parse PDFs the traditional way. Instead, it renders each page as an image and lets multimodal LLMs (Claude, Kimi, Qwen, etc.) "read" it — just like a human would. The result? Clean, structured Markdown that handles what other tools simply can't: complex tables with merged cells, inline math formulas, source code blocks, embedded diagrams, and even blurry scans.

## Demo

Real conversion results on academic papers and technical documents — no post-editing.

### Image Extraction, Tables & Code

<table>
  <tr>
    <td><img src="demo/demo0.png" alt="demo0" width="400"/></td>
    <td><img src="demo/demo3.png" alt="demo3" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>PDF original — mixed figures, tables & code</em></td>
    <td align="center"><em>Converted Markdown — images extracted, tables formatted</em></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="demo/demo1.png" alt="demo1" width="400"/></td>
    <td><img src="demo/demo4.png" alt="demo4" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>PDF original — tables & code blocks</em></td>
    <td align="center"><em>Converted Markdown — syntax-highlighted code</em></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="demo/demo2.png" alt="demo2" width="400"/></td>
    <td><img src="demo/demo5.png" alt="demo5" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>PDF original — charts & formulas</em></td>
    <td align="center"><em>Converted Markdown — chart images referenced</em></td>
  </tr>
</table>

### Math Formulas & Blurred Content

<table>
  <tr>
    <td><img src="demo/demo6.png" alt="demo6" width="400"/></td>
    <td><img src="demo/demo7.png" alt="demo7" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>PDF original — dense math formulas</em></td>
    <td align="center"><em>Converted Markdown — LaTeX `$...$` and `$$...$$` wrapping</em></td>
  </tr>
</table>

<table>
  <tr>
    <td><img src="demo/demo8.png" alt="demo8" width="400"/></td>
    <td><img src="demo/demo9.png" alt="demo9" width="400"/></td>
  </tr>
  <tr>
    <td align="center"><em>PDF original — blurred / low-quality scan</em></td>
    <td align="center"><em>Converted Markdown — content correctly recognized</em></td>
  </tr>
</table>

## Features

- 🖼️ **Vision-based extraction** — treats each page as an image, handles complex layouts that traditional parsers miss
- 🧮 **Math formulas** — LaTeX rendering with automatic `$...$` and `$$...$$` wrapping
- 📊 **Complex tables** — merged cells, multi-row headers, nested structures
- 💻 **Code blocks** — syntax-appropriate formatting for source code
- ✂️ **Image extraction** — `--crop-images` to crop figures and diagrams as separate files
- 🔍 **Blur tolerance** — handles low-quality and blurred scans with high recognition accuracy
- 🤖 **Multi-provider** — Claude, Kimi, Xiaomi, Qwen, and any OpenAI-compatible API
- ⚡ **Incremental caching** — SHA-256 progressive cache avoids re-processing unchanged pages

## Installation

```bash
pip install pdfmark-ai
```

## Requirements

- Python >= 3.10
- An LLM API key (Anthropic, Kimi, Qwen, or OpenAI-compatible)

## Quick Start

```bash
# Step 1: Generate config templates in your current directory
pdfmark --init

# Step 2: Edit .env — uncomment ONE provider and fill in your API key
#   e.g. LLM_API_KEY=your-xiaomi-api-key

# Step 3: Run
pdfmark input.pdf -o output.md
```

Default provider is **Xiaomi MiMo** (`mimo-v2.5`). To use a different provider, either edit `.env` to set `LLM_MODEL` / `LLM_BASE_URL`, or change `active_provider` in `pdfmark.toml`.

> 💡 **Tip:** Configuration files (`.env` and `pdfmark.toml`) are always read from your **current working directory** — not from the package installation directory. Place them alongside your PDF files or in your project root.

## Configuration

pdfmark-ai uses a 4-layer priority chain: **CLI args > env vars > TOML config > defaults**.

Config files live in your **working directory** (where you run `pdfmark`):

| File | Purpose | Contains |
|------|---------|----------|
| `.env` | API keys & overrides | `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` |
| `pdfmark.toml` | Provider presets & settings | providers, DPI, chunking, caching |

You can generate both files with `pdfmark --init`, or create them manually.

### .env (API keys)

```bash
# Uncomment ONE provider and add your key:
LLM_API_KEY=your-xiaomi-api-key
# LLM_AUTH_TYPE=auth_token
# LLM_API_KEY=your-kimi-api-key
# LLM_API_KEY=your-anthropic-api-key
# LLM_API_KEY=your-qwen-api-key

# Optional: override model or base URL
# LLM_MODEL=mimo-v2.5
# LLM_BASE_URL=https://token-plan-cn.xiaomimimo.com/anthropic
```

### pdfmark.toml (settings)

```toml
active_provider = "xiaomi"

[providers.xiaomi]
base_url = "https://token-plan-cn.xiaomimimo.com/anthropic"
model = "mimo-v2.5"

[render]
dpi = 150

[cache]
enabled = true
dir = "~/.cache/pdfmark"
```

### Supported Providers

| Provider | `active_provider` | Notes |
|----------|-------------------|-------|
| Anthropic Claude | `anthropic` | Supports Opus 4.6, Sonnet 4.6 and other Claude models. Uses Anthropic Messages API natively. |
| Kimi (Moonshot) | `kimi` | Anthropic-compatible API |
| Xiaomi (MiMo) | `xiaomi` | Auth token required. Default provider (mimo-v2.5). |
| Qwen (Alibaba) | `qwen` | OpenAI-compatible SDK |
| Any OpenAI-compatible | set `LLM_BASE_URL` | Set `LLM_SDK_TYPE=openai` |

## CLI Reference

```
Usage: pdfmark [OPTIONS] [INPUT]

Arguments:
  INPUT                   Path to the PDF file to convert

Options:
  --init                  Generate .env and pdfmark.toml config templates
  -f, --force             Overwrite existing config files (use with --init)
  -o, --output            Output markdown file path
  --lang                  Document language (e.g. 'en', 'zh', 'auto')
  --crop-images           Extract visual regions from pages as images
  --refine                (deprecated, ignored) Has no effect.
  --no-cache              Disable caching of rendered pages and chunks
  --no-frontmatter        Omit YAML frontmatter from output
  --detect-only           Detect document structure and print sections
  --config                Path to a TOML configuration file
  --dpi                   Rendering DPI for PDF pages
  --model                 LLM model identifier
  --api-key               LLM API key (or set LLM_API_KEY env var)
  --base-url              LLM API base URL
  --max-concurrent        Maximum concurrent LLM requests
```

## Image Extraction

Use `--crop-images` to extract figures and diagrams from the PDF as separate image files:

```bash
pdfmark input.pdf -o output.md --crop-images
```

Cropped images are saved alongside the output file (e.g., `images/page_003_fig_001.png`). Crop mode and plain mode use separate caches, so you can switch freely without needing `--no-cache`.

## License

MIT
