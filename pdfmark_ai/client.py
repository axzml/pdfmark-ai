"""LLM client supporting Anthropic and OpenAI-compatible APIs with retry and concurrency control."""

from __future__ import annotations

import asyncio
import base64
import logging
import re

from anthropic import AsyncAnthropic, APITimeoutError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)

MAX_RETRIES = 3


def clean_response(raw: str) -> str:
    """Strip markdown code fences that LLM sometimes wraps around output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:markdown|md)?\s*\n?", "", raw)
    raw = re.sub(r"\n?```\s*$", "", raw)
    return raw.strip()


class LLMClient:
    """Async LLM client supporting Anthropic and OpenAI-compatible APIs.

    sdk_type:
        "anthropic" — Anthropic Messages API (Kimi, Xiaomi)
        "openai"   — OpenAI Chat Completions API (Zhipu, etc.)
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_concurrent: int = 5,
        timeout: int = 300,
        auth_type: str = "api_key",
        sdk_type: str = "anthropic",
        request_delay: float = 0,
    ):
        self._sdk_type = sdk_type
        self._model = model
        self._request_delay = request_delay
        self._semaphore = asyncio.Semaphore(max_concurrent)

        if sdk_type == "openai":
            from openai import AsyncOpenAI
            self._openai = AsyncOpenAI(
                api_key=api_key, base_url=base_url, timeout=timeout,
            )
            self._client = None
        else:
            if auth_type == "auth_token":
                self._client = AsyncAnthropic(
                    auth_token=api_key, base_url=base_url, timeout=timeout,
                )
            else:
                self._client = AsyncAnthropic(
                    api_key=api_key, base_url=base_url, timeout=timeout,
                )
            self._openai = None

    def _build_image_content(self, images: list[bytes]) -> list[dict]:
        """Build image blocks in the appropriate SDK format."""
        content = []
        if self._sdk_type == "openai":
            # OpenAI format: {"type": "image_url", "image_url": {"url": data:image/png;base64,...}}
            for img_bytes in images:
                b64 = base64.b64encode(img_bytes).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                })
        else:
            # Anthropic format
            for img_bytes in images:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(img_bytes).decode(),
                    },
                })
        return content

    async def _call_api(
        self, content: list[dict], system: str, max_tokens: int = 8192
    ) -> str:
        """Call the API with retry logic."""
        if self._request_delay > 0:
            await asyncio.sleep(self._request_delay)
        if self._sdk_type == "openai":
            return await self._call_openai(content, system, max_tokens)
        return await self._call_anthropic(content, system, max_tokens)

    async def _call_anthropic(
        self, content: list[dict], system: str, max_tokens: int
    ) -> str:
        """Call Anthropic Messages API with retry."""
        messages = [{"role": "user", "content": content}]

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    system=system,
                    messages=messages,
                )
                if not response.content:
                    raise ValueError("LLM returned empty content")
                return response.content[0].text
            except (APITimeoutError, APIConnectionError) as e:
                if attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    raise RuntimeError(
                        f"API call failed after {MAX_RETRIES} retries: {e}"
                    ) from e
            except APIStatusError:
                raise

        raise RuntimeError("Unreachable")

    async def _call_openai(
        self, content: list[dict], system: str, max_tokens: int
    ) -> str:
        """Call OpenAI Chat Completions API with retry."""
        messages = [{"role": "system", "content": system}]
        messages.append({"role": "user", "content": content})

        for attempt in range(MAX_RETRIES):
            try:
                response = await self._openai.chat.completions.create(
                    model=self._model,
                    max_tokens=max_tokens,
                    messages=messages,
                )
                if not response.choices:
                    raise ValueError("LLM returned empty response")
                text = response.choices[0].message.content or ""
                return text
            except Exception as e:
                err_str = str(e)
                is_retryable = any(kw in err_str for kw in ("timeout", "connection", "429"))
                if is_retryable and attempt < MAX_RETRIES - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        f"API error (attempt {attempt + 1}/{MAX_RETRIES}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    await asyncio.sleep(wait)
                elif attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** (attempt + 1))
                else:
                    raise RuntimeError(
                        f"API call failed after {MAX_RETRIES} retries: {e}"
                    ) from e

        raise RuntimeError("Unreachable")

    async def extract(
        self,
        images: list[bytes],
        system: str,
        prompt: str,
        max_tokens: int = 8192,
    ) -> str:
        """Send images + prompt, return markdown text."""
        async with self._semaphore:
            content = self._build_image_content(images)
            content.append({"type": "text", "text": prompt})
            raw = await self._call_api(content, system, max_tokens)
            return clean_response(raw)

    async def refine(
        self, fragments: str, system: str, prompt: str, max_tokens: int = 64000
    ) -> str:
        """Send text-only refinement request."""
        async with self._semaphore:
            content = [{"type": "text", "text": prompt}]
            raw = await self._call_api(content, system, max_tokens)
            return clean_response(raw)
