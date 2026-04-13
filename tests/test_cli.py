"""Tests for pdfmark_ai.cli."""

import pytest
from typer.testing import CliRunner
from pdfmark_ai.cli import app

runner = CliRunner()


class TestCLI:
    def test_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "pdf" in result.output.lower()

    def test_missing_input(self):
        result = runner.invoke(app, [])
        assert result.exit_code != 0

    def test_nonexistent_file(self):
        result = runner.invoke(app, ["nonexistent.pdf"])
        assert result.exit_code != 0

    def test_detect_only_flag(self):
        result = runner.invoke(app, ["--help"])
        assert "--detect-only" in result.output

    def test_config_flag(self):
        result = runner.invoke(app, ["--help"])
        assert "--config" in result.output

    def test_init_flag(self):
        result = runner.invoke(app, ["--help"])
        assert "--init" in result.output
