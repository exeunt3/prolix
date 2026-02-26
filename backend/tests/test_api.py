import os

from fastapi.testclient import TestClient

os.environ.setdefault("VISION_GROUNDING_API_KEY", "test-key")
os.environ.setdefault("VISION_GROUNDING_MODEL", "test-model")
os.environ.setdefault("VISION_GROUNDING_ENDPOINT", "http://127.0.0.1:9/v1/responses")
os.environ.setdefault("VISION_GROUNDING_TIMEOUT_SECONDS", "0.2")
os.environ.setdefault("VISION_GROUNDING_MAX_RETRIES", "0")

from app.main import app


client = TestClient(app)


def test_generate_contract() -> None:
    resp = client.post(
        "/generate",
        data={"tap_x": 0.3, "tap_y": 0.7, "image_b64": "dGlyZSB3aXJl"},
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
        data={"tap_x": 0.4, "tap_y": 0.5, "image_b64": "bGVhZiBvYmplY3Q="},
    ).json()
    resp = client.post("/deepen", json={"trace_id": generated["trace_id"]})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["trace_id"] != generated["trace_id"]


def test_safety_redirection() -> None:
    resp = client.post(
        "/generate",
        data={"tap_x": 0.1, "tap_y": 0.2, "image_b64": "ZmFjZSBwbGF0ZQ=="},
    )
    assert resp.status_code == 200
    text = resp.json()["paragraph_text"].lower()
    assert "dossier" in text or "anonymity" in text
