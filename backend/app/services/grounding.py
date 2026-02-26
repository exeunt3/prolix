from __future__ import annotations

import base64
import imghdr
import json
import os
import time
from dataclasses import dataclass
from typing import Protocol
from urllib import error, request

from app.models import GroundingResult


@dataclass(frozen=True)
class ProviderGroundingResponse:
    object_label: str
    confidence: float
    scene_descriptors: list[str]
    safety_face_or_plate: bool = False


class VisionGroundingProvider(Protocol):
    def identify_object(self, image_bytes: bytes, tap_x: float, tap_y: float) -> ProviderGroundingResponse:
        """Identify an object near the tap target and return structured grounding metadata."""


@dataclass(frozen=True)
class VisionProviderSettings:
    api_key: str
    model: str
    endpoint: str
    timeout_seconds: float = 8.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "VisionProviderSettings":
        required = {
            "VISION_GROUNDING_API_KEY": os.getenv("VISION_GROUNDING_API_KEY"),
            "VISION_GROUNDING_MODEL": os.getenv("VISION_GROUNDING_MODEL"),
            "VISION_GROUNDING_ENDPOINT": os.getenv("VISION_GROUNDING_ENDPOINT"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise RuntimeError(f"Missing required grounding provider settings: {missing_str}")

        timeout_raw = os.getenv("VISION_GROUNDING_TIMEOUT_SECONDS", "8")
        retries_raw = os.getenv("VISION_GROUNDING_MAX_RETRIES", "2")
        return cls(
            api_key=required["VISION_GROUNDING_API_KEY"] or "",
            model=required["VISION_GROUNDING_MODEL"] or "",
            endpoint=required["VISION_GROUNDING_ENDPOINT"] or "",
            timeout_seconds=max(0.5, float(timeout_raw)),
            max_retries=max(0, int(retries_raw)),
        )


@dataclass
class OpenAIVisionGroundingProvider:
    settings: VisionProviderSettings

    def identify_object(self, image_bytes: bytes, tap_x: float, tap_y: float) -> ProviderGroundingResponse:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "You are grounding a user tap on an image. "
            f"Tap coordinates are normalized floats x={tap_x:.4f}, y={tap_y:.4f}. "
            "Return strict JSON with keys: object_label (string), confidence (0..1 float), "
            "scene_descriptors (array of short strings), safety_face_or_plate (boolean)."
        )
        payload = {
            "model": self.settings.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                    ],
                }
            ],
            "text": {"format": {"type": "json_object"}},
        }

        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.settings.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
        )

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.settings.timeout_seconds) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return self._parse_response(response_payload)
            except (error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt < self.settings.max_retries:
                    time.sleep(0.25 * (attempt + 1))

        raise RuntimeError(f"Grounding provider request failed: {last_error}")

    def _parse_response(self, payload: dict[str, object]) -> ProviderGroundingResponse:
        output_text = self._extract_output_text(payload)
        parsed = json.loads(output_text)
        object_label = str(parsed.get("object_label") or "object")
        confidence = float(parsed.get("confidence") or 0.0)
        descriptors = [str(item) for item in parsed.get("scene_descriptors") or []]
        safety = bool(parsed.get("safety_face_or_plate", False))
        return ProviderGroundingResponse(
            object_label=object_label,
            confidence=max(0.0, min(confidence, 1.0)),
            scene_descriptors=descriptors,
            safety_face_or_plate=safety,
        )

    def _extract_output_text(self, payload: dict[str, object]) -> str:
        output = payload.get("output")
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, dict):
                    continue
                for content in item.get("content", []):
                    if isinstance(content, dict) and isinstance(content.get("text"), str):
                        return content["text"]
        if isinstance(payload.get("output_text"), str):
            return str(payload["output_text"])
        raise ValueError("Provider response did not include parseable output text")


@dataclass
class GroundingService:
    provider: VisionGroundingProvider
    confidence_threshold: float = 0.45
    fallback_label: str = "object"

    def ground(self, image_bytes: bytes | None, tap_x: float, tap_y: float) -> GroundingResult:
        descriptors = [f"tap near x={tap_x:.2f}", f"tap near y={tap_y:.2f}"]
        if not image_bytes:
            return GroundingResult(
                object_label=self.fallback_label,
                scene_descriptors=descriptors,
                confidence=0.0,
                safety_face_or_plate=False,
            )

        try:
            provider_result = self.provider.identify_object(image_bytes=image_bytes, tap_x=tap_x, tap_y=tap_y)
            descriptors.extend(provider_result.scene_descriptors)
            safety = provider_result.safety_face_or_plate or self._policy_safety_marker(
                provider_result.object_label,
                provider_result.scene_descriptors,
            )
            label = provider_result.object_label
            if provider_result.confidence < self.confidence_threshold:
                label = self.fallback_label
            return GroundingResult(
                object_label=label,
                scene_descriptors=descriptors,
                confidence=provider_result.confidence,
                safety_face_or_plate=safety,
            )
        except Exception:
            kind = imghdr.what(None, image_bytes)
            fallback_confidence = 0.52 if kind else 0.0
            return GroundingResult(
                object_label=self.fallback_label,
                scene_descriptors=descriptors + ["provider_unavailable"],
                confidence=fallback_confidence,
                safety_face_or_plate=self._legacy_safety_heuristic(image_bytes),
            )

    def _policy_safety_marker(self, object_label: str, scene_descriptors: list[str]) -> bool:
        haystack = " ".join([object_label, *scene_descriptors]).lower()
        return "face" in haystack or "plate" in haystack

    def _legacy_safety_heuristic(self, image_bytes: bytes) -> bool:
        lowered = image_bytes.lower()
        return b"face" in lowered or b"plate" in lowered


def decode_image_b64(image_b64: str) -> bytes:
    candidate = image_b64.strip()
    missing_padding = len(candidate) % 4
    if missing_padding:
        candidate += "=" * (4 - missing_padding)
    return base64.b64decode(candidate, validate=False)
