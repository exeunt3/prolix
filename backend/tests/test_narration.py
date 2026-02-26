import os

from app.models import GroundingPack, Hop, RetrievalSnippet, VectorDomain
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
    return [Hop(node="object", rel="is_a"), Hop(node="process", rel="enabled_by"), Hop(node="system", rel="embedded_in")]


def _snippets() -> list[RetrievalSnippet]:
    return [
        RetrievalSnippet(
            title="Entropy and prediction",
            excerpt="Information processing is paid for in heat, delay, and substrate.",
            domain_tag="THERMODYNAMICS_ENTROPY",
            source_id="core-entropy",
        )
    ]


def _grounding() -> GroundingPack:
    return GroundingPack(
        anchor_label="wire",
        anchor_description="coiled copper strand, nicked insulation, matte dust",
        anchor_material_guess="metal and polymer",
        anchor_scene_context="on a dark bench",
        confidence=0.8,
        visual_facts=[
            "coiled copper strand",
            "nicked insulation",
            "matte dust",
            "small scratch",
            "residue seam",
            "shadow under the bend",
            "oxidized spot",
        ],
    )


def test_narration_regenerates_until_valid() -> None:
    stage1 = '{"visual_facts_used":["coiled copper strand","nicked insulation","matte dust","small scratch","residue seam","shadow under the bend"],"hop_trace":[{"from_node":"Anchor","to_node":"Material","relation":"made_of","evidence":"frag"}],"micro_outline":["a","b","c","d","e","f"],"banned_words_triggered":[],"tone_checks":{"second_person":false,"academic_markers":0}}'
    sentence = (
        "The wire remains visible with coiled copper strand, nicked insulation, matte dust, small scratch, residue seam, and shadow under the bend, "
        "while micron pits, centuries of extraction, and global shipment timing keep oxidized spot and seam drift tied to measured maintenance."
    )
    valid = " ".join([sentence] * 8)
    llm = StubLLM([stage1, valid])
    service = NarrationService(llm_client=llm)

    result = service.narrate(
        object_label="wire",
        descriptors=["metallic", "industrial"],
        vector_domain=VectorDomain.THERMODYNAMICS_ENTROPY,
        path=_path(),
        snippets=_snippets(),
        grounding_pack=_grounding(),
    )

    assert llm.calls == 2
    assert 250 <= len(result.paragraph_text.split()) <= 350
    assert "\n" not in result.paragraph_text


def test_safety_prompt_branch_uses_safety_language() -> None:
    os.environ.pop("OPENAI_API_KEY", None)
    service = NarrationService(llm_client=None)
    result = service.narrate(
        object_label="face",
        descriptors=["portrait"],
        vector_domain=VectorDomain.FEEDBACK_CONTROL,
        path=_path(),
        snippets=_snippets(),
    )

    assert len(result.paragraph_text.split()) >= 250
