from __future__ import annotations

import random

from app.models import (
    ConceptHop,
    ConceptPath,
    GroundingPack,
    NarrationResult,
    OutlinePlan,
    RetrievalBundle,
    RetrievalSnippet,
    Hop,
    VectorDomain,
)
from app.services.llm_client import ChatCompletionLLMClient, LLMClient
from app.services.stage1_outline import generate_outline
from app.services.stage2_prose import generate_prose_stage2


class GroundingPackRequiredError(ValueError):
    pass


class NarrationService:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or ChatCompletionLLMClient.from_env()

    def generateProse(
        self,
        *,
        image: bytes | None,
        tap: tuple[float, float],
        groundingPack: GroundingPack,
        conceptPath: ConceptPath,
        retrievalBundle: RetrievalBundle,
    ) -> NarrationResult:
        if groundingPack is None or not groundingPack.visual_facts:
            raise GroundingPackRequiredError("grounding pack missing or empty")

        ending_suspend = random.random() < 0.4
        ending_type = "SUSPEND" if ending_suspend else "RETURN"

        outline: OutlinePlan = generate_outline(
            llm_client=self.llm_client,
            grounding_pack=groundingPack,
            concept_path=conceptPath,
            retrieval_bundle=retrievalBundle,
        )
        prose = generate_prose_stage2(llm_client=self.llm_client, outline=outline, grounding_pack=groundingPack)
        if not prose.validation_passed:
            raise GroundingPackRequiredError(f"prose validation failed: {', '.join(prose.violations)}")
        return NarrationResult(paragraph_text=prose.paragraph_text, path_used=_concept_to_hops(conceptPath), ending_type=ending_type)

    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool = False,
        grounding_pack: GroundingPack | None = None,
    ) -> NarrationResult:
        if grounding_pack is None:
            facts = descriptors + ["object in tapped region", "visible wear and residue", "localized texture"]
            grounding_pack = GroundingPack(
                anchor_label=object_label,
                anchor_description=", ".join(facts[:3]),
                anchor_material_guess="composite material",
                anchor_scene_context="image scene",
                confidence=0.5,
                low_confidence=False,
                visual_facts=facts[:8],
            )

        concept_path = ConceptPath(
            chosen_vector_domain=_map_vector(vector_domain),
            hop_trace=[
                ConceptHop(
                    from_node=path[i - 1].node if i > 0 else grounding_pack.anchor_label,
                    to_node=hop.node,
                    relation=hop.rel,
                )
                for i, hop in enumerate(path)
            ],
        )
        retrieval_bundle = RetrievalBundle(
            fragment_ids=[snippet.source_id for snippet in snippets[:10]],
            fragments=[f"{snippet.title}: {snippet.excerpt}" for snippet in snippets[:10]],
        )
        return self.generateProse(
            image=None,
            tap=(0.0, 0.0),
            groundingPack=grounding_pack,
            conceptPath=concept_path,
            retrievalBundle=retrieval_bundle,
        )


def _map_vector(domain: VectorDomain) -> str:
    mapping = {
        VectorDomain.THERMODYNAMICS_ENTROPY: "Physical Sciences",
        VectorDomain.CHEMISTRY_TRANSFORMATION: "Physical Sciences",
        VectorDomain.PHYSICS_PARADOXES: "Physical Sciences",
        VectorDomain.COSMOLOGY: "Physical Sciences",
        VectorDomain.DISTRIBUTED_INTELLIGENCE: "Cybernetic",
        VectorDomain.EVOLUTIONARY_SIGNALING: "Biological & Ecological",
        VectorDomain.PLANT_AGENCY: "Biological & Ecological",
        VectorDomain.FEEDBACK_CONTROL: "Cybernetic",
        VectorDomain.EMERGENCE_OBSERVER: "Cybernetic",
        VectorDomain.INFRASTRUCTURE_LOGISTICS: "Historical",
        VectorDomain.OBSCURE_HISTORY_AGENTS: "Historical",
        VectorDomain.HERESY_SECTS: "Intellectual & Anomalous",
        VectorDomain.MYSTERIOUS_EVENTS: "Intellectual & Anomalous",
        VectorDomain.COSMOLOGY_MYTH: "Intellectual & Anomalous",
        VectorDomain.CULT_PRACTICES: "Intellectual & Anomalous",
    }
    return mapping.get(domain, "Physical Sciences")


def _concept_to_hops(concept_path: ConceptPath) -> list[Hop]:
    return [Hop(node=hop.to_node, rel=hop.relation) for hop in concept_path.hop_trace]
