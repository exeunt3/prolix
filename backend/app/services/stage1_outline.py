from __future__ import annotations

import json

from app.models import ConceptPath, GroundingPack, OutlinePlan, RetrievalBundle
from app.services.llm_client import LLMClient
from app.services.prompts import STAGE1_SYSTEM_PROMPT, build_stage1_user_prompt


def generate_outline(
    *,
    llm_client: LLMClient | None,
    grounding_pack: GroundingPack,
    concept_path: ConceptPath,
    retrieval_bundle: RetrievalBundle,
) -> OutlinePlan:
    if llm_client is None:
        return _fallback_outline(grounding_pack=grounding_pack, concept_path=concept_path, retrieval_bundle=retrieval_bundle)

    raw = llm_client.generate(
        system_prompt=STAGE1_SYSTEM_PROMPT,
        user_prompt=build_stage1_user_prompt(
            grounding_pack=grounding_pack,
            concept_path=concept_path,
            retrieval_bundle=retrieval_bundle,
        ),
    )
    try:
        payload = json.loads(raw)
        return OutlinePlan.model_validate(payload)
    except Exception:
        return _fallback_outline(grounding_pack=grounding_pack, concept_path=concept_path, retrieval_bundle=retrieval_bundle)


def _fallback_outline(*, grounding_pack: GroundingPack, concept_path: ConceptPath, retrieval_bundle: RetrievalBundle) -> OutlinePlan:
    hops = []
    for hop in concept_path.hop_trace[:6]:
        evidence = retrieval_bundle.fragments[min(len(hops), max(len(retrieval_bundle.fragments) - 1, 0))] if retrieval_bundle.fragments else grounding_pack.anchor_description
        hops.append({"from_node": hop.from_node, "to_node": hop.to_node, "relation": hop.relation, "evidence": evidence})
    return OutlinePlan(
        visual_facts_used=grounding_pack.visual_facts[:6],
        hop_trace=hops,
        micro_outline=[
            "Recognition sentence plan using visible anchor details in present tense.",
            "Material descent plan from visible texture to material composition.",
            "System expansion plan tied to retrieved industrial or ecological contexts.",
            "Scale destabilization plan with order-of-magnitude cue.",
            "Conceptual turn plan constrained by adjacency relation and evidence.",
            "Return/suspension plan back to the scene without moral closure.",
        ],
        banned_words_triggered=[],
        tone_checks={"second_person": False, "academic_markers": 0},
    )
