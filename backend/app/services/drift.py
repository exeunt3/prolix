from __future__ import annotations

import random
from collections import deque

from app.models import DriftPlan, Hop, VectorDomain


DARK_DOMAINS = {VectorDomain.MYSTERIOUS_EVENTS}

DOMAIN_WEIGHTS = {
    VectorDomain.THERMODYNAMICS_ENTROPY: 1.0,
    VectorDomain.CHEMISTRY_TRANSFORMATION: 1.0,
    VectorDomain.PHYSICS_PARADOXES: 0.9,
    VectorDomain.COSMOLOGY: 0.5,
    VectorDomain.DISTRIBUTED_INTELLIGENCE: 1.0,
    VectorDomain.EVOLUTIONARY_SIGNALING: 1.0,
    VectorDomain.PLANT_AGENCY: 0.8,
    VectorDomain.FEEDBACK_CONTROL: 1.0,
    VectorDomain.EMERGENCE_OBSERVER: 0.8,
    VectorDomain.INFRASTRUCTURE_LOGISTICS: 1.0,
    VectorDomain.OBSCURE_HISTORY_AGENTS: 0.8,
    VectorDomain.HERESY_SECTS: 0.7,
    VectorDomain.MYSTERIOUS_EVENTS: 0.35,
    VectorDomain.COSMOLOGY_MYTH: 0.7,
    VectorDomain.CULT_PRACTICES: 0.6,
}

ATTRACTORS = {
    VectorDomain.HERESY_SECTS: ["Bogomils"],
    VectorDomain.PHYSICS_PARADOXES: ["quantum suicide", "the number 137", "holographic principle"],
    VectorDomain.COSMOLOGY: ["holographic principle"],
    VectorDomain.PLANT_AGENCY: ["corn is growing us"],
    VectorDomain.COSMOLOGY_MYTH: ["Egyptian cosmology motifs"],
    VectorDomain.DISTRIBUTED_INTELLIGENCE: ["distributed brains of octopi"],
    VectorDomain.OBSCURE_HISTORY_AGENTS: ["Napoleonic-era triple-agent courier networks", "pseudo-Frederick"],
    VectorDomain.MYSTERIOUS_EVENTS: ["Pauli Effect"],
    VectorDomain.CULT_PRACTICES: ["pre-modern bear cults"],
}


class DriftEngine:
    def __init__(self) -> None:
        self.recent_dark: deque[bool] = deque(maxlen=20)
        self.recent_attractors: deque[str] = deque(maxlen=3)

    def choose_domain(self, previous: VectorDomain | None = None) -> VectorDomain:
        domains = [d for d in VectorDomain if d != previous]
        weights = [DOMAIN_WEIGHTS[d] for d in domains]
        dark_ratio = (sum(self.recent_dark) / len(self.recent_dark)) if self.recent_dark else 0
        if dark_ratio > 0.35:
            weights = [w * 0.3 if domains[i] in DARK_DOMAINS else w for i, w in enumerate(weights)]
        return random.choices(domains, weights=weights, k=1)[0]

    def plan(self, anchor: str, domain: VectorDomain, force_safe: bool = False) -> DriftPlan:
        candidates = [entry for entry in ATTRACTORS.get(domain, []) if entry not in self.recent_attractors]
        attractor = random.choice(candidates) if candidates else None
        dark_flag = domain in DARK_DOMAINS and not force_safe
        domain_signal = domain.value.lower().replace("_", " ")
        chain = [
            Hop(node=anchor, rel="is_a"),
            Hop(node=f"{anchor} substrate", rel="made_of"),
            Hop(node="extraction and process chemistry", rel="derived_from"),
            Hop(node=f"{domain_signal} dynamics", rel="enabled_by"),
            Hop(node=f"{domain_signal} system behavior", rel="embedded_in"),
            Hop(node="micro-to-macro scale inversion", rel="scaled_to"),
            Hop(node=f"{domain_signal} observer entanglement", rel="resembles"),
            Hop(node=f"{domain_signal} constraints", rel="constrained_by"),
        ]
        if attractor:
            chain.append(Hop(node=attractor, rel="historically_entangled_with"))
        chain.append(Hop(node="scene-level return signal", rel="feeds"))
        chain = chain[: random.randint(8, 10)]
        self._validate(chain, domain)
        self.recent_dark.append(dark_flag)
        if attractor:
            self.recent_attractors.append(attractor)
        return DriftPlan(vector_domain=domain, concept_path=chain, dark_flag=dark_flag, attractor=attractor)

    def _validate(self, chain: list[Hop], domain: VectorDomain) -> None:
        if not (6 <= len(chain) <= 12):
            raise ValueError("Invalid hop count")
        if not any(h.rel == "scaled_to" for h in chain):
            raise ValueError("Missing scale shift")
        domain_signal = domain.value.lower().replace("_", " ")
        hops_after_intro = chain[2:]
        domain_hits = sum(1 for hop in hops_after_intro if domain_signal[:10] in hop.node)
        commitment_ratio = domain_hits / max(1, len(hops_after_intro))
        if commitment_ratio < 0.6:
            for idx in range(3, len(chain) - 1):
                if chain[idx].rel != "scaled_to":
                    chain[idx] = Hop(node=f"{domain_signal} {chain[idx].node}", rel=chain[idx].rel)
