from __future__ import annotations

import base64
import binascii
import imghdr
from dataclasses import dataclass

from app.models import GroundingResult
from app.providers import VisionProvider


@dataclass
class GroundingService:
    confidence_threshold: float = 0.45
    provider: VisionProvider | None = None

    def _decode_image(self, image_b64: str) -> bytes:
        candidate = image_b64.strip()
        missing_padding = len(candidate) % 4
        if missing_padding:
            candidate += "=" * (4 - missing_padding)
        return base64.b64decode(candidate, validate=False)

    def ground(self, image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult:
        if self.provider is not None:
            try:
                provided = self.provider.analyze(image_b64, tap_x=tap_x, tap_y=tap_y)
            except Exception:
                provided = None
            if provided is not None:
                return provided

        # MVP deterministic placeholder for vision label extraction.
        descriptors = [
            f"tap near x={tap_x:.2f}",
            f"tap near y={tap_y:.2f}",
            "ambient manufactured light",
        ]
        label = "object"
        confidence = 0.40
        safety = False
        if image_b64:
            try:
                raw = self._decode_image(image_b64)
            except (ValueError, binascii.Error):
                raw = b""
            kind = imghdr.what(None, raw)
            if kind:
                confidence = 0.62
            if b"face" in raw.lower() or b"plate" in raw.lower():
                safety = True
            if b"leaf" in raw.lower():
                label = "leaf"
            elif b"wire" in raw.lower():
                label = "wire"
            elif b"tire" in raw.lower():
                label = "tire"
            elif b"bottle" in raw.lower():
                label = "bottle"
            else:
                label = "object"
        if confidence < self.confidence_threshold:
            label = "object"
        return GroundingResult(
            object_label=label,
            scene_descriptors=descriptors,
            confidence=confidence,
            safety_face_or_plate=safety,
        )
