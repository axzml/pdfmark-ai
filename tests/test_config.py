"""Tests for pdfmark_ai.config."""

import os
import pytest
from pathlib import Path
from pdfmark_ai.config import load_config, DEFAULT_CONFIG


class TestDefaultConfig:
    def test_has_required_keys(self):
        keys = ["api_key", "base_url", "model", "timeout",
                "max_concurrent", "dpi", "max_pages",
                "window_size", "window_overlap",
                "language", "refine", "frontmatter",
                "cache_enabled", "cache_dir"]
        for key in keys:
            assert key in DEFAULT_CONFIG, f"Missing key: {key}"

    def test_sensible_defaults(self):
        assert DEFAULT_CONFIG["dpi"] == 150
        assert DEFAULT_CONFIG["max_pages"] == 4
        assert DEFAULT_CONFIG["window_size"] == 2
        assert DEFAULT_CONFIG["window_overlap"] == 1
        assert DEFAULT_CONFIG["max_concurrent"] == 5
        assert DEFAULT_CONFIG["language"] == "auto"
        assert DEFAULT_CONFIG["refine"] is False


class TestLoadConfig:
    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "test-key")
        monkeypatch.setenv("LLM_MODEL", "test-model")
        config = load_config({})
        assert config["api_key"] == "test-key"
        assert config["model"] == "test-model"

    def test_cli_override_takes_priority(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "env-key")
        config = load_config({"api_key": "cli-key"})
        assert config["api_key"] == "cli-key"

    def test_toml_config_loading(self, tmp_path, monkeypatch):
        toml_file = tmp_path / "pdfmark.toml"
        toml_file.write_text("""
[llm]
model = "toml-model"
max_concurrent = 10

[render]
dpi = 200

[chunk]
max_pages = 6
""")
        config = load_config({"config": str(toml_file)})
        assert config["model"] == "toml-model"
        assert config["max_concurrent"] == 10
        assert config["dpi"] == 200
        assert config["max_pages"] == 6

    def test_no_cache_mode(self, monkeypatch):
        config = load_config({"no_cache": True})
        assert config["cache_enabled"] is False
