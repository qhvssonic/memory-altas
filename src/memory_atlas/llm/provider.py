"""LiteLLM wrapper for unified LLM access."""

from __future__ import annotations

from typing import Any

import litellm


class LLMProvider:
    """Thin wrapper around LiteLLM for MemoryAtlas internal LLM calls."""

    def __init__(self, model: str = "openai/gpt-4o-mini", api_key: str | None = None):
        self.model = model
        if api_key:
            litellm.api_key = api_key

    def complete(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.3,
        max_tokens: int = 512,
    ) -> str:
        """Single-turn completion."""
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = litellm.completion(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()

    def complete_json(
        self,
        prompt: str,
        system: str = "",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """Completion expecting JSON output."""
        import json

        raw = self.complete(prompt, system, temperature, max_tokens)
        # Try to extract JSON from markdown code blocks
        if "```" in raw:
            import re
            match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
            if match:
                raw = match.group(1)
        return json.loads(raw)
