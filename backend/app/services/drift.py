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
    VectorDomain.HERESY_SECTS: "Bogomils",
    VectorDomain.PHYSICS_PARADOXES: "quantum suicide",
    VectorDomain.COSMOLOGY: "holographic principle",
    VectorDomain.PLANT_AGENCY: "corn is growing us",
    VectorDomain.COSMOLOGY_MYTH: "Egyptian cosmology motifs",
    VectorDomain.DISTRIBUTED_INTELLIGENCE: "distributed brains of octopi",
    VectorDomain.OBSCURE_HISTORY_AGENTS: "Napoleonic courier networks",
    VectorDomain.MYSTERIOUS_EVENTS: "Pauli Effect",
    VectorDomain.CULT_PRACTICES: "pre-modern bear cults",
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
        attractor = ATTRACTORS.get(domain)
        if attractor in self.recent_attractors:
            attractor = None
        dark_flag = domain in DARK_DOMAINS and not force_safe
        chain = [
            Hop(node=anchor, rel="is_a"),
            Hop(node=f"{anchor} substrate", rel="made_of"),
            Hop(node="extraction and process chemistry", rel="derived_from"),
            Hop(node=f"{domain.value.lower().replace('_', ' ')} dynamics", rel="enabled_by"),
            Hop(node="networked system behavior", rel="embedded_in"),
            Hop(node="micro-to-macro scale inversion", rel="scaled_to"),
            Hop(node="observer entanglement", rel="resembles"),
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
        domain_hits = sum(1 for hop in chain[2:] if domain.value.lower().replace("_", " ")[:8] in hop.node)
        if domain_hits < 1:
            # Ensure vector commitment on deterministic skeleton.
            chain[3] = Hop(node=f"{domain.value.lower().replace('_', ' ')} commitments", rel="enabled_by")
