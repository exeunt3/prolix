from app.models import ConceptHop, ConceptPath, GroundingPack, OutlinePlan, RetrievalBundle
from app.services.prompts import build_stage1_user_prompt
from app.services.stage1_outline import generate_outline
from app.services.validator import validate_prose


class StubLLM:
    def __init__(self, output: str) -> None:
        self.output = output

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return self.output


def _grounding() -> GroundingPack:
    return GroundingPack(
        anchor_label="bolt",
        anchor_description="dull gray head, chipped rim, oily sheen",
        anchor_material_guess="metal",
        anchor_scene_context="on scratched painted panel",
        confidence=0.8,
        low_confidence=False,
        visual_facts=[
            "dull gray head",
            "hexagonal rim",
            "oily sheen",
            "chipped paint around edge",
            "rust freckles",
            "shadow to lower right",
            "thread marks",
        ],
    )


def test_stage1_prompt_contains_grounding_fields() -> None:
    prompt = build_stage1_user_prompt(
        grounding_pack=_grounding(),
        concept_path=ConceptPath(
            chosen_vector_domain="Physical Sciences",
            hop_trace=[ConceptHop(from_node="Anchor", to_node="Material", relation="made_of")],
        ),
        retrieval_bundle=RetrievalBundle(fragment_ids=["a"], fragments=["doc fragment"]),
    )
    assert "anchor_description" in prompt
    assert "visual_facts" in prompt


def test_outline_fallback_uses_visual_facts() -> None:
    outline = generate_outline(
        llm_client=StubLLM("not-json"),
        grounding_pack=_grounding(),
        concept_path=ConceptPath(
            chosen_vector_domain="Physical Sciences",
            hop_trace=[ConceptHop(from_node="Anchor", to_node="Material", relation="made_of")],
        ),
        retrieval_bundle=RetrievalBundle(fragment_ids=["a", "b"], fragments=["frag1", "frag2"]),
    )
    assert len(outline.visual_facts_used) >= 6


def test_validator_detects_generic_essay_failures() -> None:
    bad = "In essence this framework is complex and interconnected. " * 20
    violations = validate_prose(bad, _grounding())
    assert "word_count" in violations or "visual_facts" in violations
    assert "conclusion_markers" in violations
