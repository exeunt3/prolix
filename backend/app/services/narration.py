from __future__ import annotations

import random

from app.models import Hop, NarrationResult, RetrievalSnippet, VectorDomain


class NarrationService:
    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool = False,
    ) -> NarrationResult:
        if safety_redirect:
            text = (
                "At the tapped point, the scene refuses to become a dossier and remains an object among other objects: edge, glare, texture, motion, each one enough to trigger the machinery of recognition without granting it authority. "
                "The frame carries the old fantasy that naming is neutral, yet every naming system drags a registry behind it, and every registry quietly asks who gets to disappear. "
                "Industrial optics taught cameras to sort faces, vehicles, and habits as if identity were a surface property, but perception at scale behaves more like weather than law: it swirls, leaks, and reclassifies. "
                "What looks like a stable subject dissolves into signal processing, compression artifacts, transmission standards, and institutional appetite, then widens into the politics of anonymity where masks are not concealment but negotiated borders. "
                "In that widened system, certainty is the most theatrical output, because the apparatus is built from thresholds, not truths, from confidence scores that mimic judgment while avoiding responsibility. "
                "So the image settles back into atmosphere and grain, and the tapped region becomes a reminder that the ethics of looking begins before identification and continues after the database closes."
            )
            return NarrationResult(paragraph_text=text, path_used=path, ending_type="RETURN")

        lead = f"The tapped {object_label} sits in a field of {', '.join(descriptors[:2])}, ordinary enough to be ignored until its material lineage starts talking."
        process = (
            f" It descends through {path[1].node} and {path[2].node}, where manufacture is less a single act than a relay of pressures, catalysts, labor schedules, and delayed consequences."
        )
        system = (
            f" From there the chain enters {path[3].node}, then folds into {path[4].node}, and the object stops being an item and becomes a local interface for a larger machine."
        )
        scale = (
            " The scale shifts abruptly: microscopic frictions become metropolitan rhythms, and present handling touches deep-time deposits, evolutionary compromises, and institutional memory at once."
        )
        turn_target = path[-2].node if len(path) > 7 else vector_domain.value.lower().replace("_", " ")
        snippet = snippets[0].excerpt if snippets else "prediction burns energy and meaning in the same breath"
        turn = (
            f" That is where the conceptual turn appears, because {turn_target} is not trivia but a pressure point: {snippet.lower()}, making observation itself part of the circuit it describes."
        )
        ending_suspend = random.random() < 0.4
        ending = (
            " The scene returns with altered weight, the object still present yet no longer solitary, as if it had been carrying an invisible infrastructure in plain sight."
            if not ending_suspend
            else " The scene does not close; it hovers, with the object acting like a small aperture through which systems, myths, and measurements continue to leak."
        )
        paragraph = lead + process + system + scale + turn + ending
        paragraph = self._fit_word_range(paragraph)
        return NarrationResult(paragraph_text=paragraph, path_used=path, ending_type="SUSPEND" if ending_suspend else "RETURN")

    def _fit_word_range(self, text: str) -> str:
        words = text.split()
        filler = (
            " Around that aperture sit shipping ledgers, field ecologies, protocol committees, refinery leftovers, laboratory metaphors, and inherited rituals that keep translating matter into coordination while pretending coordination was already there."
        )
        while len(words) < 250:
            text += filler
            words = text.split()
        if len(words) > 350:
            text = " ".join(words[:350]).rstrip(" ,.;") + "."
        return " ".join(text.split())
