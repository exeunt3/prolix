from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.models import DriftPlan, GroundingResult, Hop, NarrationResult, RetrievalSnippet, VectorDomain


client = TestClient(app)


def _deterministic_paragraph(prefix: str) -> str:
    sentence = f"{prefix} output remains a single-paragraph systems narration for deterministic contract testing"
    return " ".join([sentence] * 24)


def _path(anchor: str) -> list[Hop]:
    return [
        Hop(node=anchor, rel="is_a"),
        Hop(node=f"{anchor} substrate", rel="made_of"),
        Hop(node="process chemistry", rel="derived_from"),
        Hop(node="feedback control dynamics", rel="enabled_by"),
        Hop(node="feedback control behavior", rel="embedded_in"),
        Hop(node="micro-to-macro inversion", rel="scaled_to"),
        Hop(node="observer entanglement", rel="resembles"),
        Hop(node="constraints", rel="constrained_by"),
        Hop(node="scene-level return signal", rel="feeds"),
    ]


def test_generate_contract(monkeypatch) -> None:
    import app.main as main

    def mock_ground(image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult:
        return GroundingResult(
            object_label="wire",
            scene_descriptors=["tap near x=0.30", "tap near y=0.70", "ambient manufactured light"],
            confidence=0.9,
            safety_face_or_plate=False,
        )

    def mock_choose_domain(previous: VectorDomain | None = None) -> VectorDomain:
        return VectorDomain.FEEDBACK_CONTROL

    def mock_plan(anchor: str, domain: VectorDomain, force_safe: bool = False) -> DriftPlan:
        return DriftPlan(vector_domain=domain, concept_path=_path(anchor), dark_flag=False)

    def mock_retrieve(object_label: str, domain: VectorDomain, k: int = 12) -> list[RetrievalSnippet]:
        return [
            RetrievalSnippet(
                title="Deterministic",
                excerpt="Deterministic retrieval snippet for API contracts.",
                domain_tag=domain.value,
                source_id="det-1",
            )
        ]

    def mock_narrate(**kwargs) -> NarrationResult:
        return NarrationResult(paragraph_text=_deterministic_paragraph("Generate"), path_used=kwargs["path"], ending_type="RETURN")

    monkeypatch.setattr(main.grounder, "ground", mock_ground)
    monkeypatch.setattr(main.drift, "choose_domain", mock_choose_domain)
    monkeypatch.setattr(main.drift, "plan", mock_plan)
    monkeypatch.setattr(main.retrieval, "retrieve", mock_retrieve)
    monkeypatch.setattr(main.narrator, "narrate", mock_narrate)

    resp = client.post("/generate", data={"tap_x": 0.3, "tap_y": 0.7, "image_b64": "irrelevant"})
    assert resp.status_code == 200
    payload = resp.json()

    assert "trace_id" in payload
    assert "\n" not in payload["paragraph_text"]
    words = payload["paragraph_text"].split()
    assert 250 <= len(words) <= 350


def test_deepen_contract(monkeypatch) -> None:
    import app.main as main

    def mock_ground(image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult:
        return GroundingResult(
            object_label="wire",
            scene_descriptors=["tap near x=0.40", "tap near y=0.50", "ambient manufactured light"],
            confidence=0.9,
            safety_face_or_plate=False,
        )

    def mock_choose_domain(previous: VectorDomain | None = None) -> VectorDomain:
        return VectorDomain.FEEDBACK_CONTROL

    def mock_plan(anchor: str, domain: VectorDomain, force_safe: bool = False) -> DriftPlan:
        return DriftPlan(vector_domain=domain, concept_path=_path(anchor), dark_flag=False)

    def mock_retrieve(object_label: str, domain: VectorDomain, k: int = 12) -> list[RetrievalSnippet]:
        return [
            RetrievalSnippet(
                title="Deterministic",
                excerpt="Deterministic retrieval snippet for deepen contracts.",
                domain_tag=domain.value,
                source_id="det-2",
            )
        ]

    def mock_narrate(**kwargs) -> NarrationResult:
        return NarrationResult(paragraph_text=_deterministic_paragraph("Deepen"), path_used=kwargs["path"], ending_type="RETURN")

    monkeypatch.setattr(main.grounder, "ground", mock_ground)
    monkeypatch.setattr(main.drift, "choose_domain", mock_choose_domain)
    monkeypatch.setattr(main.drift, "plan", mock_plan)
    monkeypatch.setattr(main.retrieval, "retrieve", mock_retrieve)
    monkeypatch.setattr(main.narrator, "narrate", mock_narrate)

    generated = client.post("/generate", data={"tap_x": 0.4, "tap_y": 0.5, "image_b64": "irrelevant"}).json()
    resp = client.post("/deepen", json={"trace_id": generated["trace_id"]})
    assert resp.status_code == 200
    payload = resp.json()

    assert payload["trace_id"] != generated["trace_id"]
    assert "\n" not in payload["paragraph_text"]
    words = payload["paragraph_text"].split()
    assert 250 <= len(words) <= 350


def test_safety_redirection_contract(monkeypatch) -> None:
    import app.main as main

    seen_safety_redirect: list[bool] = []

    def mock_ground(image_b64: str | None, tap_x: float, tap_y: float) -> GroundingResult:
        return GroundingResult(
            object_label="object",
            scene_descriptors=["tap near x=0.10", "tap near y=0.20", "ambient manufactured light"],
            confidence=0.7,
            safety_face_or_plate=True,
        )

    def mock_retrieve(object_label: str, domain: VectorDomain, k: int = 12) -> list[RetrievalSnippet]:
        return [
            RetrievalSnippet(
                title="Safety",
                excerpt="Safety mode retrieval snippet.",
                domain_tag=domain.value,
                source_id="det-3",
            )
        ]

    def mock_narrate(**kwargs) -> NarrationResult:
        seen_safety_redirect.append(kwargs["safety_redirect"])
        return NarrationResult(paragraph_text=_deterministic_paragraph("Safety"), path_used=kwargs["path"], ending_type="RETURN")

    monkeypatch.setattr(main.grounder, "ground", mock_ground)
    monkeypatch.setattr(main.retrieval, "retrieve", mock_retrieve)
    monkeypatch.setattr(main.narrator, "narrate", mock_narrate)

    resp = client.post("/generate", data={"tap_x": 0.1, "tap_y": 0.2, "image_b64": "contains-safety-signal"})
    assert resp.status_code == 200
    payload = resp.json()

    assert "trace_id" in payload
    assert seen_safety_redirect == [True]
    words = payload["paragraph_text"].split()
    assert 250 <= len(words) <= 350
    assert "\n" not in payload["paragraph_text"]
