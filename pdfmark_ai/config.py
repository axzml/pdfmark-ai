"""Configuration loading with priority: CLI args > env vars > TOML config > defaults."""

from __future__ import annotations

import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# Auto-load .env from current working directory
try:
    from dotenv import load_dotenv
    _cwd = Path.cwd()
    if (_cwd / ".env").exists():
        load_dotenv(_cwd / ".env")
except ImportError:
    pass  # dotenv not installed, rely on explicit env vars


DEFAULT_CONFIG: dict = {
    "api_key": "",
    "base_url": "",
    "model": "",
    "auth_type": "api_key",
    "sdk_type": "anthropic",
    "request_delay": 0,
    "timeout": 300,
    "max_concurrent": 5,
    "dpi": 150,
    "max_pages": 4,
    "window_size": 2,
    "window_overlap": 1,
    "language": "auto",
    "refine": False,
    "frontmatter": True,
    "cache_enabled": True,
    "cache_dir": str(Path.home() / ".cache" / "pdfmark"),
}


def load_config(cli_args: dict, config_path: Path | None = None) -> dict:
    """Load configuration with priority: CLI args > env vars > TOML config > defaults.

    Args:
        cli_args: CLI argument dict (from typer).
        config_path: Explicit path to TOML config. If None, searches for
                     pdfmark.toml in the current directory, then project root.
    """
    config = dict(DEFAULT_CONFIG)

    # Resolve TOML path
    toml_path = cli_args.get("config")
    if toml_path:
        _merge_toml(config, Path(toml_path))
    elif config_path and config_path.exists():
        _merge_toml(config, config_path)
    else:
        # Auto-discover pdfmark.toml in current dir or project root
        _try_auto_toml(config)

    _merge_env(config)
    _merge_cli(config, cli_args)
    return config


def _try_auto_toml(config: dict) -> None:
    """Try to find and load pdfmark.toml from current working directory."""
    toml_path = Path.cwd() / "pdfmark.toml"
    if toml_path.exists():
        _merge_toml(config, toml_path)


def _merge_toml(config: dict, path: Path) -> None:
    """Merge settings from a TOML config file into the config dict."""
    if not path.exists():
        return
    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Provider preset support
    active = data.get("active_provider", "")
    if active and "providers" in data:
        providers = data["providers"]
        if active in providers:
            preset = providers[active]
            for key in ("base_url", "model", "timeout", "max_concurrent", "api_key", "auth_type", "sdk_type", "request_delay"):
                if key in preset and preset[key]:
                    config[key] = preset[key]

    # Direct [llm] section (legacy, takes priority over preset if present)
    if "llm" in data:
        llm = data["llm"]
        for key in ("api_key", "base_url", "model", "timeout", "max_concurrent"):
            if key in llm and llm[key]:
                config[key] = llm[key]

    if "render" in data:
        for key in ("dpi",):
            if key in data["render"]:
                config[key] = data["render"][key]
    if "chunk" in data:
        for key in ("max_pages", "window_size", "window_overlap"):
            if key in data["chunk"]:
                config[key] = data["chunk"][key]
    if "output" in data:
        for key in ("language", "refine", "frontmatter"):
            if key in data["output"]:
                config[key] = data["output"][key]
    if "cache" in data:
        if "enabled" in data["cache"]:
            config["cache_enabled"] = data["cache"]["enabled"]
        if "dir" in data["cache"]:
            config["cache_dir"] = data["cache"]["dir"].replace("~", str(Path.home()))


def _merge_env(config: dict) -> None:
    """Merge environment variables into the config dict."""
    env_map = {
        "LLM_API_KEY": "api_key",
        "LLM_BASE_URL": "base_url",
        "LLM_MODEL": "model",
        "LLM_AUTH_TYPE": "auth_type",
        "LLM_SDK_TYPE": "sdk_type",
    }
    for env_key, config_key in env_map.items():
        val = os.environ.get(env_key, "")
        if val:
            config[config_key] = val


def _merge_cli(config: dict, cli_args: dict) -> None:
    """Merge CLI arguments into the config dict (highest priority)."""
    if cli_args.get("no_cache"):
        config["cache_enabled"] = False
    if cli_args.get("refine"):
        config["refine"] = True
    if cli_args.get("no_frontmatter"):
        config["frontmatter"] = False
    value_map = {
        "api_key": "api_key", "base_url": "base_url", "model": "model",
        "timeout": "timeout", "max_concurrent": "max_concurrent",
        "dpi": "dpi", "max_pages": "max_pages",
        "window_size": "window_size", "window_overlap": "window_overlap",
        "language": "language", "cache_dir": "cache_dir",
    }
    for cli_key, config_key in value_map.items():
        val = cli_args.get(cli_key)
        if val is not None:
            config[config_key] = val
