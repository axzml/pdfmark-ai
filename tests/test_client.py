"""Tests for pdfmark_ai.client."""

import pytest
import base64
from unittest.mock import AsyncMock, MagicMock, patch
from pdfmark_ai.client import LLMClient, clean_response


class TestCleanResponse:
    def test_no_fence(self):
        assert clean_response("hello") == "hello"

    def test_markdown_fence(self):
        raw = "```markdown\n# Hello\n\nWorld\n```"
        assert clean_response(raw) == "# Hello\n\nWorld"

    def test_bare_fence(self):
        raw = "```\n# Hello\n```"
        assert clean_response(raw) == "# Hello"

    def test_fence_with_whitespace(self):
        raw = "  ```markdown  \n# Hello\n  ```  "
        assert clean_response(raw) == "# Hello"


class TestLLMClient:
    @pytest.mark.asyncio
    async def test_extract_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock(text="# Result", type="text")
        mock_response.content = [text_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("pdfmark_ai.client.AsyncAnthropic", return_value=mock_client):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.extract([b"img"], "system prompt", "user prompt")
            assert result == "# Result"
            mock_client.messages.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_thinking_model(self):
        """Thinking models return ThinkingBlock + TextBlock."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        thinking = MagicMock(type="thinking", thinking="let me think...")
        text_block = MagicMock(type="text", text="# Result from thinking")
        mock_response.content = [thinking, text_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("pdfmark_ai.client.AsyncAnthropic", return_value=mock_client):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.extract([b"img"], "system prompt", "user prompt")
            assert result == "# Result from thinking"

    @pytest.mark.asyncio
    async def test_extract_only_thinking_raises(self):
        """If response contains only ThinkingBlocks, raise ValueError."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        thinking = MagicMock(type="thinking", thinking="...")
        mock_response.content = [thinking]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("pdfmark_ai.client.AsyncAnthropic", return_value=mock_client):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            with pytest.raises(ValueError, match="No text block"):
                await client.extract([b"img"], "sys", "usr")

    @pytest.mark.asyncio
    async def test_extract_empty_response_raises(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = []
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("pdfmark_ai.client.AsyncAnthropic", return_value=mock_client):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            with pytest.raises(ValueError, match="empty content"):
                await client.extract([b"img"], "sys", "usr")

    @pytest.mark.asyncio
    async def test_refine_text_only(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        text_block = MagicMock(text="# Refined", type="text")
        mock_response.content = [text_block]
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch("pdfmark_ai.client.AsyncAnthropic", return_value=mock_client):
            client = LLMClient(api_key="test", base_url="http://test", model="test-model")
            result = await client.refine("fragments text", "system", "prompt")
            assert result == "# Refined"
            # Verify no images were sent
            call_kwargs = mock_client.messages.create.call_args
            content = call_kwargs.kwargs["messages"][0]["content"]
            assert all(item["type"] == "text" for item in content)
