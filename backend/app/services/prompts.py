from __future__ import annotations

import json

from app.models import ConceptPath, GroundingPack, OutlinePlan, RetrievalBundle

STAGE1_SYSTEM_PROMPT = (
    "You create an internal plan for a single-paragraph object meditation. "
    "You must remain faithful to the provided visual facts and retrieved fragments. "
    "You must not write the final paragraph yet. Output valid JSON only matching the schema."
)

STAGE2_SYSTEM_PROMPT = (
    "You write exactly one paragraph of 250–350 words. No line breaks. No second-person. "
    "No summaries. Use the outline and hop trace. Keep the object present throughout. "
    "Weirdness only via honest adjacency. End with return or suspension."
)

BANNED_WORDS = [
    "interplay",
    "framework",
    "dynamics",
    "interconnected",
    "landscape",
    "complex",
    "systemic",
    "infrastructure",
    "paradigm",
    "urges us",
    "highlights our relationship",
    "in essence",
    "not merely",
]


def build_stage1_user_prompt(*, grounding_pack: GroundingPack, concept_path: ConceptPath, retrieval_bundle: RetrievalBundle) -> str:
    payload = {
        "schema": {
            "visual_facts_used": ["at least 6 items copied/adapted from grounding pack"],
            "hop_trace": [
                {"from": "Anchor Object", "to": "Material", "relation": "made_of", "evidence": "..."}
            ],
            "micro-outline": [
                "Recognition sentence plan...",
                "Material descent plan...",
                "System expansion plan...",
                "Scale destabilization plan...",
                "Conceptual turn plan...",
                "Return/suspension plan...",
            ],
            "banned_words_triggered": [],
            "tone_checks": {"second_person": False, "academic_markers": 0},
        },
        "grounding_pack": grounding_pack.model_dump(),
        "chosen_vector": concept_path.chosen_vector_domain,
        "hop_trace": [hop.model_dump() for hop in concept_path.hop_trace],
        "retrieved_fragments": retrieval_bundle.model_dump(),
        "requirements": [
            "Provide evidence strings that quote/paraphrase retrieved fragments for at least two hops.",
            "If low_confidence is true, do not assert identity beyond visual anchor descriptors.",
        ],
    }
    return json.dumps(payload)


def build_stage2_user_prompt(*, outline: OutlinePlan, grounding_pack: GroundingPack) -> str:
    payload = {
        "outline": outline.model_dump(),
        "grounding_pack": grounding_pack.model_dump(),
        "banned_or_avoid_words": BANNED_WORDS,
        "constraints": {
            "single_paragraph": True,
            "word_count": [250, 350],
            "no_second_person": True,
            "must_reference_min_visual_facts": 6,
            "must_include_measurement_like_specificity": True,
            "must_include_time_or_wear_marker": True,
        },
    }
    return json.dumps(payload)
