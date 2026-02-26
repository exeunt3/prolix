import struct
import zlib

from app.services.grounding import OpenAIVisionGroundingProvider, VisionProviderSettings, crop_around_tap


def _png_bytes(w: int = 400, h: int = 300) -> bytes:
    raw = b"\x00" + (b"\x7f\x50\x28" * w)
    raw = raw * h

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def test_crop_around_tap_uses_normalized_coordinates() -> None:
    img = _png_bytes(400, 300)
    _, rect, facts = crop_around_tap(img, tap_x=0.9, tap_y=0.1, crop_ratio=0.5)

    assert rect["x"] > 100
    assert rect["y"] == 0
    assert rect["w"] == 200
    assert rect["h"] == 150
    assert any("tap normalized x" in item for item in facts)


def test_vision_payload_contains_input_image() -> None:
    provider = OpenAIVisionGroundingProvider(
        settings=VisionProviderSettings(api_key="k", model="m", endpoint="http://example.test")
    )
    payload = provider.build_payload(image_b64="abc", prompt="hello")
    content = payload["input"][0]["content"]
    assert any(item.get("type") == "input_image" for item in content)
