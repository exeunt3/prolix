from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from typing import Protocol

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - optional in some test envs
    httpx = None


class LLMClient(Protocol):
    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate text from an LLM provider."""


@dataclass(slots=True)
class ChatCompletionLLMClient:
    api_key: str
    model: str
    endpoint: str = "https://api.openai.com/v1/chat/completions"
    timeout_s: float = 15.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> ChatCompletionLLMClient | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        model = os.getenv("NARRATION_MODEL", "gpt-4o-mini")
        timeout_s = float(os.getenv("NARRATION_TIMEOUT_S", "15"))
        max_retries = int(os.getenv("NARRATION_MAX_RETRIES", "3"))
        endpoint = os.getenv("NARRATION_ENDPOINT", "https://api.openai.com/v1/chat/completions")
        return cls(api_key=api_key, model=model, endpoint=endpoint, timeout_s=timeout_s, max_retries=max_retries)

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if httpx is None:
            raise RuntimeError("httpx is required for ChatCompletionLLMClient")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    response = client.post(self.endpoint, headers=headers, json=payload)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise RuntimeError(f"Transient provider error ({response.status_code})")
                response.raise_for_status()
                data = response.json()
                return data["choices"][0]["message"]["content"]
            except (httpx.TimeoutException, httpx.HTTPError, KeyError, RuntimeError) as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    break
                backoff = min(2 ** (attempt - 1), 4) + random.random() * 0.2
                time.sleep(backoff)

        raise RuntimeError("Chat completion failed after retries") from last_error
