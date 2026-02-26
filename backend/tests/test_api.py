import base64
import os
import struct
import zlib

import httpx
from fastapi.testclient import TestClient

os.environ.setdefault("VISION_GROUNDING_API_KEY", "test-key")
os.environ.setdefault("VISION_GROUNDING_MODEL", "test-model")
os.environ.setdefault("VISION_GROUNDING_ENDPOINT", "http://127.0.0.1:9/v1/responses")
os.environ.setdefault("VISION_GROUNDING_TIMEOUT_SECONDS", "0.2")
os.environ.setdefault("VISION_GROUNDING_MAX_RETRIES", "0")

from app.main import app


client = TestClient(app)


def _png_bytes(w: int = 320, h: int = 240) -> bytes:
    raw = (b"\x00" + b"\x78\x5a\x46" * w) * h

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def _image_b64() -> str:
    return base64.b64encode(_png_bytes()).decode("ascii")


def test_generate_contract() -> None:
    resp = client.post(
        "/generate",
        data={"tap_x": 0.3, "tap_y": 0.7, "image_b64": _image_b64()},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "trace_id" in payload
    words = payload["paragraph_text"].split()
    assert 250 <= len(words) <= 350
    assert "\n" not in payload["paragraph_text"]


def test_deepen_contract() -> None:
    generated = client.post(
        "/generate",
        data={"tap_x": 0.4, "tap_y": 0.5, "image_b64": _image_b64()},
    ).json()
    resp = client.post("/deepen", json={"trace_id": generated["trace_id"]})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["trace_id"] != generated["trace_id"]


def test_generate_rejects_missing_grounding_pack() -> None:
    resp = client.post(
        "/generate",
        data={"tap_x": 0.1, "tap_y": 0.2, "image_b64": "not-an-image"},
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error_type"] == "missing_grounding_pack"


def test_web_app_shell() -> None:
    resp = client.get("/web")
    assert resp.status_code == 200
    assert "<title>Prolix Web</title>" in resp.text

    script_resp = client.get("/web-static/app.js")
    assert script_resp.status_code == 200
    assert "fetch('/api/ai/respond'" in script_resp.text


def test_ai_respond_missing_key() -> None:
    previous = os.environ.pop("OPENAI_API_KEY", None)
    try:
        resp = client.post("/api/ai/respond", json={"text": "hello"})
        assert resp.status_code == 500
        assert "OPENAI_API_KEY is missing" in resp.json()["detail"]
    finally:
        if previous is not None:
            os.environ["OPENAI_API_KEY"] = previous


def test_ai_respond_validation() -> None:
    resp = client.post("/api/ai/respond", json={"text": "   "})
    assert resp.status_code == 422


def test_ai_respond_success(monkeypatch) -> None:
    os.environ["OPENAI_API_KEY"] = "test-openai-key"

    class MockResponse:
        status_code = 200

        @staticmethod
        def json() -> dict[str, object]:
            return {"choices": [{"message": {"content": "Mock reply"}}]}

    async def mock_post(self, url, headers=None, json=None):
        return MockResponse()

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post)

    resp = client.post("/api/ai/respond", json={"text": "Say hi"})
    assert resp.status_code == 200
    assert resp.json() == {"reply": "Mock reply"}
