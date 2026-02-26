from app.models import Hop, RetrievalSnippet, VectorDomain
from app.services.narration import NarrationService


class SuccessfulNarrationProvider:
    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
        timeout_s: float,
    ) -> dict:
        return {
            "paragraph_text": " ".join(["provider"] * 260),
            "ending_type": "RETURN",
        }


class TimeoutNarrationProvider:
    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
        timeout_s: float,
    ) -> dict:
        raise TimeoutError("provider timeout")


class MalformedNarrationProvider:
    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
        timeout_s: float,
    ) -> dict:
        return {"ending_type": "RETURN"}


def _path() -> list[Hop]:
    return [
        Hop(node="wire", rel="is_a"),
        Hop(node="wire substrate", rel="made_of"),
        Hop(node="process", rel="derived_from"),
        Hop(node="domain dynamics", rel="enabled_by"),
        Hop(node="domain behavior", rel="embedded_in"),
        Hop(node="scale inversion", rel="scaled_to"),
        Hop(node="observer entanglement", rel="resembles"),
        Hop(node="constraints", rel="constrained_by"),
        Hop(node="scene return", rel="feeds"),
    ]


def _snippets() -> list[RetrievalSnippet]:
    return [
        RetrievalSnippet(
            title="A",
            excerpt="Energy and perception are entangled in infrastructure.",
            domain_tag=VectorDomain.FEEDBACK_CONTROL.value,
            source_id="src-1",
        )
    ]


def test_narration_service_uses_provider_response_on_success() -> None:
    service = NarrationService(provider_client=SuccessfulNarrationProvider())

    result = service.narrate("wire", ["x", "y"], VectorDomain.FEEDBACK_CONTROL, _path(), _snippets())

    assert 250 <= len(result.paragraph_text.split()) <= 350
    assert result.ending_type == "RETURN"


def test_narration_service_falls_back_when_provider_times_out() -> None:
    service = NarrationService(provider_client=TimeoutNarrationProvider())

    result = service.narrate("wire", ["x", "y"], VectorDomain.FEEDBACK_CONTROL, _path(), _snippets())

    assert 250 <= len(result.paragraph_text.split()) <= 350
    assert "\n" not in result.paragraph_text


def test_narration_service_falls_back_on_malformed_provider_response() -> None:
    service = NarrationService(provider_client=MalformedNarrationProvider())

    result = service.narrate("wire", ["x", "y"], VectorDomain.FEEDBACK_CONTROL, _path(), _snippets())

    assert 250 <= len(result.paragraph_text.split()) <= 350
    assert result.path_used == _path()


def test_narration_service_uses_internal_fallback_without_provider() -> None:
    service = NarrationService(provider_client=None)

    result = service.narrate("wire", ["x", "y"], VectorDomain.FEEDBACK_CONTROL, _path(), _snippets(), safety_redirect=True)

    assert result.ending_type == "RETURN"
    assert len(result.paragraph_text.split()) > 80
