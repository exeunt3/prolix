from __future__ import annotations

import base64
import imghdr
import json
import logging
import os
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib import error, request

from app.models import GroundingPack, GroundingResult

logger = logging.getLogger(__name__)


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
        timeout_raw = os.getenv("VISION_GROUNDING_TIMEOUT_SECONDS", "8")
        retries_raw = os.getenv("VISION_GROUNDING_MAX_RETRIES", "2")
        return cls(
            api_key=os.getenv("VISION_GROUNDING_API_KEY", ""),
            model=os.getenv("VISION_GROUNDING_MODEL", "gpt-4o-mini"),
            endpoint=os.getenv("VISION_GROUNDING_ENDPOINT", "https://api.openai.com/v1/responses"),
            timeout_seconds=max(0.5, float(timeout_raw)),
            max_retries=max(0, int(retries_raw)),
        )


@dataclass
class OpenAIVisionGroundingProvider:
    settings: VisionProviderSettings

    def identify_object(self, image_bytes: bytes, tap_x: float, tap_y: float) -> ProviderGroundingResponse:
        if not self.settings.api_key:
            raise RuntimeError("VISION_GROUNDING_API_KEY is required to call the vision provider")

        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        prompt = (
            "You are grounding a user tap on an image. "
            f"Tap coordinates are normalized floats x={tap_x:.4f}, y={tap_y:.4f}. "
            "Return strict JSON with keys: object_label (string), confidence (0..1 float), "
            "scene_descriptors (array of short strings), safety_face_or_plate (boolean). "
            "Include concrete descriptors from the tapped object."
        )
        payload = self.build_payload(image_b64=image_b64, prompt=prompt)

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

    def build_payload(self, *, image_b64: str, prompt: str) -> dict[str, object]:
        return {
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

    def _parse_response(self, payload: dict[str, object]) -> ProviderGroundingResponse:
        output_text = self._extract_output_text(payload)
        parsed = json.loads(output_text)
        object_label = str(parsed.get("object_label") or "unknown")
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
    fallback_label: str = "unknown"

    def ground(self, image_bytes: bytes | None, tap_x: float, tap_y: float) -> GroundingResult:
        pack = self.build_grounding_pack(image_bytes=image_bytes, tap_x=tap_x, tap_y=tap_y)
        return GroundingResult(
            object_label=pack.anchor_label,
            scene_descriptors=[pack.anchor_description, pack.anchor_scene_context, *pack.visual_facts[:4]],
            confidence=pack.confidence,
            safety_face_or_plate=False,
        )

    def build_grounding_pack(self, *, image_bytes: bytes | None, tap_x: float, tap_y: float) -> GroundingPack:
        if not image_bytes:
            raise ValueError("grounding_pack_missing_image")

        crop_bytes, crop_rect, visual_facts = crop_around_tap(image_bytes, tap_x=tap_x, tap_y=tap_y)
        try:
            provider_result = self.provider.identify_object(image_bytes=crop_bytes, tap_x=tap_x, tap_y=tap_y)
            low_conf = provider_result.confidence < self.confidence_threshold
            anchor_label = self.fallback_label if low_conf else provider_result.object_label
            facts = _dedupe_facts([*visual_facts, *provider_result.scene_descriptors])
            if len(facts) < 5:
                raise ValueError("insufficient_visual_facts")
            return GroundingPack(
                anchor_label=anchor_label,
                anchor_description=", ".join(facts[:3]),
                anchor_material_guess=_guess_material(facts),
                anchor_scene_context=f"around tapped region at ({tap_x:.3f}, {tap_y:.3f})",
                confidence=provider_result.confidence,
                low_confidence=low_conf,
                visual_facts=facts[:12],
                crop_rect=crop_rect,
            )
        except Exception:
            facts = _dedupe_facts(visual_facts)
            if len(facts) < 5:
                facts.extend(["visible edge contrast", "localized texture", "object occupies crop center"])
            return GroundingPack(
                anchor_label=self.fallback_label,
                anchor_description=", ".join(facts[:3]),
                anchor_material_guess=_guess_material(facts),
                anchor_scene_context="whole image fallback" if crop_rect.get("w", 0) == 0 else "cropped tap fallback",
                confidence=0.2,
                low_confidence=True,
                visual_facts=facts[:12],
                crop_rect=crop_rect,
            )


def crop_around_tap(image_bytes: bytes, *, tap_x: float, tap_y: float, crop_ratio: float = 0.45) -> tuple[bytes, dict[str, int], list[str]]:
    width, height = _image_dimensions(image_bytes)
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")

    nx = max(0.0, min(1.0, tap_x))
    ny = max(0.0, min(1.0, tap_y))
    cx = int(nx * width)
    cy = int(ny * height)
    crop_w = max(64, int(width * crop_ratio))
    crop_h = max(64, int(height * crop_ratio))
    left = max(0, min(width - crop_w, cx - crop_w // 2))
    top = max(0, min(height - crop_h, cy - crop_h // 2))
    right = min(width, left + crop_w)
    bottom = min(height, top + crop_h)

    debug = os.getenv("PROLIX_DEBUG_GROUNDING", "0") == "1"
    if debug:
        out_dir = Path(os.getenv("PROLIX_DEBUG_CROP_DIR", "backend/debug_crops"))
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = int(time.time() * 1000)
        crop_path = out_dir / f"crop_{stamp}.bin"
        crop_path.write_bytes(image_bytes)
        logger.info(
            "grounding_debug source=%sx%s tap=(%.3f,%.3f) crop_rect=(%s,%s,%s,%s) crop_file=%s",
            width,
            height,
            nx,
            ny,
            left,
            top,
            right - left,
            bottom - top,
            crop_path,
        )

    facts = [
        f"crop width {right - left} px",
        f"crop height {bottom - top} px",
        f"tap normalized x {nx:.3f}",
        f"tap normalized y {ny:.3f}",
        "object-adjacent region centered on tap",
    ]
    rect = {"x": left, "y": top, "w": right - left, "h": bottom - top, "image_w": width, "image_h": height}
    return image_bytes, rect, facts


def _image_dimensions(image_bytes: bytes) -> tuple[int, int]:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") and len(image_bytes) >= 24:
        width, height = struct.unpack(">II", image_bytes[16:24])
        return int(width), int(height)
    if image_bytes.startswith(b"\xff\xd8"):
        index = 2
        while index + 9 < len(image_bytes):
            if image_bytes[index] != 0xFF:
                index += 1
                continue
            marker = image_bytes[index + 1]
            length = struct.unpack(">H", image_bytes[index + 2 : index + 4])[0]
            if marker in {0xC0, 0xC2}:
                height = struct.unpack(">H", image_bytes[index + 5 : index + 7])[0]
                width = struct.unpack(">H", image_bytes[index + 7 : index + 9])[0]
                return int(width), int(height)
            index += 2 + length
    return (0, 0)


def _guess_material(facts: list[str]) -> str:
    joined = " ".join(facts).lower()
    if "rust" in joined or "metal" in joined:
        return "metal"
    if "grain" in joined or "wood" in joined:
        return "wood"
    if "gloss" in joined or "clear" in joined:
        return "glass or polymer"
    return "composite material"


def _dedupe_facts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        normalized = " ".join(item.lower().split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(item)
    return out


def decode_image_b64(image_b64: str) -> bytes:
    candidate = image_b64.strip()
    missing_padding = len(candidate) % 4
    if missing_padding:
        candidate += "=" * (4 - missing_padding)
    return base64.b64decode(candidate, validate=False)
