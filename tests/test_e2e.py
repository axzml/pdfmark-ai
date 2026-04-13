"""End-to-end test with mocked LLM calls."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from pdfmark_ai.models import PageImage, DocumentStructure, Section


@pytest.fixture
def cli_args():
    """Build cli_args dict that _run will pass to load_config."""
    return {
        "api_key": "test-key",
        "base_url": "http://fake-api",
        "model": "test-model",
        "no_cache": True,
        "refine": False,
        "no_frontmatter": True,
    }


class TestE2E:
    @pytest.mark.asyncio
    async def test_full_pipeline_sliding_window(self, tmp_path, cli_args):
        """Test the full pipeline with mocked renderer, detector, and LLM."""
        from pdfmark_ai.cli import _run

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        output_path = tmp_path / "output.md"

        pages = [
            PageImage(page_number=i, image_bytes=f"png-{i}".encode())
            for i in range(1, 4)
        ]

        structure = DocumentStructure(
            sections=[],
            doc_type="unknown",
            language="auto",
            source="sliding_window",
        )

        with patch("pdfmark_ai.cli.render_pdf", return_value=pages):
            with patch("pdfmark_ai.cli.extract_images", return_value={}):
                with patch("pdfmark_ai.cli.detect_structure", return_value=structure):
                    with patch("pdfmark_ai.cli.build_chunks") as mock_build:
                        from pdfmark_ai.models import Chunk
                        chunks = [
                            Chunk(
                                chunk_id=0,
                                section_title="",
                                pages=pages[:2],
                                start_page=1,
                                end_page=2,
                            )
                        ]
                        mock_build.return_value = chunks

                        with patch("pdfmark_ai.cli.process_all_chunks") as mock_extract:
                            from pdfmark_ai.models import ExtractionResult
                            mock_extract.return_value = [
                                ExtractionResult(
                                    chunk_id=0,
                                    start_page=1,
                                    end_page=2,
                                    section_title="",
                                    markdown="# Hello\n\nWorld",
                                    tail_summary="World",
                                )
                            ]

                            with patch("pdfmark_ai.cli.merge_results", return_value="# Hello\n\nWorld"):
                                await _run(pdf_path, output_path, cli_args, detect_only=False)

        assert output_path.exists()
        content = output_path.read_text(encoding="utf-8")
        assert "Hello" in content

    @pytest.mark.asyncio
    async def test_detect_only_mode(self, tmp_path, cli_args):
        """Test --detect-only mode: output file should NOT be created."""
        from pdfmark_ai.cli import _run

        pdf_path = tmp_path / "test.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake content")
        output_path = tmp_path / "output.md"

        pages = [PageImage(page_number=1, image_bytes=b"png")]
        structure = DocumentStructure(
            sections=[Section(title="Intro", start_page=1, end_page=1)],
            doc_type="paper",
            language="en",
            source="outline",
        )

        with patch("pdfmark_ai.cli.render_pdf", return_value=pages):
            with patch("pdfmark_ai.cli.extract_images", return_value={}):
                with patch("pdfmark_ai.cli.detect_structure", return_value=structure):
                    with patch("pdfmark_ai.cli.LLMClient") as MockClient:
                        mock_instance = MagicMock()
                        MockClient.return_value = mock_instance

                        await _run(pdf_path, output_path, cli_args, detect_only=True)

        assert not output_path.exists()
