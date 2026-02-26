from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI

from app.models import GroundingResult, Hop, RetrievalSnippet, VectorDomain
from app.settings import Settings


class VisionProvider(Protocol):
    def analyze(self, image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult | None:
        ...


class TextProvider(Protocol):
    def compose(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
    ) -> str | None:
        ...


@dataclass
class OpenAIVisionProvider:
    client: OpenAI
    model: str

    def analyze(self, image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult | None:
        if not image_b64:
            return None

        prompt = (
            "Return compact JSON with keys object_label (string), scene_descriptors (array of short strings), "
            "confidence (0-1 float), safety_face_or_plate (boolean)."
        )
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"tap_x={tap_x:.3f}, tap_y={tap_y:.3f}. {prompt}"},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
        )
        raw = response.output_text
        data = json.loads(raw)
        return GroundingResult(
            object_label=data.get("object_label", "object"),
            scene_descriptors=list(data.get("scene_descriptors", [])) or ["ambient manufactured light"],
            confidence=float(data.get("confidence", 0.5)),
            safety_face_or_plate=bool(data.get("safety_face_or_plate", False)),
        )


@dataclass
class OpenAITextProvider:
    client: OpenAI
    model: str

    def compose(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
    ) -> str | None:
        snippet = snippets[0].excerpt if snippets else ""
        prompt = {
            "object_label": object_label,
            "descriptors": descriptors,
            "vector_domain": vector_domain.value,
            "path": [hop.model_dump() for hop in path],
            "snippet": snippet,
            "safety_redirect": safety_redirect,
            "constraints": "Write one paragraph of 250-350 words.",
        }
        response = self.client.responses.create(
            model=self.model,
            input=f"Compose a single paragraph based on this JSON context: {json.dumps(prompt)}",
        )
        return response.output_text.strip() or None


def _build_openai_client(api_key: str, endpoint: str | None, settings: Settings) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=settings.ai_request_timeout_seconds,
        max_retries=settings.ai_max_retries,
    )


def build_vision_provider(settings: Settings) -> VisionProvider | None:
    provider = settings.vision_provider.lower()
    if provider == "mock":
        return None
    if provider == "openai":
        key = settings.vision_api_key.get_secret_value() if settings.vision_api_key else ""
        endpoint = str(settings.vision_endpoint) if settings.vision_endpoint else None
        return OpenAIVisionProvider(client=_build_openai_client(key, endpoint, settings), model=settings.vision_model)
    raise ValueError(f"Unsupported VISION_PROVIDER '{settings.vision_provider}'. Supported values: mock, openai")


def build_text_provider(settings: Settings) -> TextProvider | None:
    provider = settings.text_provider.lower()
    if provider == "mock":
        return None
    if provider == "openai":
        key = settings.text_api_key.get_secret_value() if settings.text_api_key else ""
        endpoint = str(settings.text_endpoint) if settings.text_endpoint else None
        return OpenAITextProvider(client=_build_openai_client(key, endpoint, settings), model=settings.text_model)
    raise ValueError(f"Unsupported TEXT_PROVIDER '{settings.text_provider}'. Supported values: mock, openai")
