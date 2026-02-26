from app.models import Hop, RetrievalSnippet, VectorDomain
from app.services.narration import NarrationService


class StubLLM:
    def __init__(self, outputs: list[str]) -> None:
        self.outputs = outputs
        self.calls = 0

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        value = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return value


def _path() -> list[Hop]:
    return [
        Hop(node="object", rel="is_a"),
        Hop(node="process", rel="enabled_by"),
        Hop(node="system", rel="embedded_in"),
    ]


def _snippets() -> list[RetrievalSnippet]:
    return [
        RetrievalSnippet(
            title="Entropy and prediction",
            excerpt="Information processing is paid for in heat, delay, and substrate.",
            domain_tag="THERMODYNAMICS_ENTROPY",
            source_id="core-entropy",
        )
    ]


def test_narration_regenerates_until_valid() -> None:
    valid = " ".join(["word"] * 260)
    llm = StubLLM(["short output", valid])
    service = NarrationService(llm_client=llm)

    result = service.narrate(
        object_label="wire",
        descriptors=["metallic", "industrial"],
        vector_domain=VectorDomain.THERMODYNAMICS_ENTROPY,
        path=_path(),
        snippets=_snippets(),
    )

    assert llm.calls == 2
    assert 250 <= len(result.paragraph_text.split()) <= 350
    assert "\n" not in result.paragraph_text


def test_safety_prompt_branch_uses_safety_language() -> None:
    service = NarrationService(llm_client=None)
    result = service.narrate(
        object_label="face",
        descriptors=["portrait"],
        vector_domain=VectorDomain.FEEDBACK_CONTROL,
        path=_path(),
        snippets=_snippets(),
        safety_redirect=True,
    )

    assert "dossier" in result.paragraph_text.lower() or "anonymity" in result.paragraph_text.lower()
    assert 250 <= len(result.paragraph_text.split()) <= 350
