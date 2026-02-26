from __future__ import annotations

import random

from app.models import Hop, NarrationResult, RetrievalSnippet, VectorDomain
from app.services.llm_client import ChatCompletionLLMClient, LLMClient


class NarrationService:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or ChatCompletionLLMClient.from_env()

    def narrate(
        self,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool = False,
    ) -> NarrationResult:
        ending_suspend = random.random() < 0.4
        ending_type = "SUSPEND" if ending_suspend else "RETURN"
        system_prompt, user_prompt = self._build_prompts(
            object_label=object_label,
            descriptors=descriptors,
            vector_domain=vector_domain,
            path=path,
            snippets=snippets,
            safety_redirect=safety_redirect,
            ending_type=ending_type,
        )

        paragraph = self._generate_with_validation(system_prompt=system_prompt, user_prompt=user_prompt)
        return NarrationResult(paragraph_text=paragraph, path_used=path, ending_type=ending_type)

    def _build_prompts(
        self,
        *,
        object_label: str,
        descriptors: list[str],
        vector_domain: VectorDomain,
        path: list[Hop],
        snippets: list[RetrievalSnippet],
        safety_redirect: bool,
        ending_type: str,
    ) -> tuple[str, str]:
        constraints = (
            "Write exactly one paragraph between 250 and 350 words. "
            "No newlines. No bullet points. Do not include prefaces, titles, or markdown."
        )
        path_hops = " | ".join([f"{hop.node} ({hop.rel})" for hop in path])
        top_snippets = snippets[:3]
        snippets_block = " || ".join([f"{item.title}: {item.excerpt}" for item in top_snippets])

        if safety_redirect:
            system_prompt = (
                "You are a critical writing assistant for a visual analysis system. "
                "Safety mode is active: never identify or profile people or license plates. "
                "Keep the prose focused on ethics of classification, ambiguity, and non-identifying observation. "
                f"{constraints}"
            )
            user_prompt = (
                f"Object label: {object_label}. "
                f"Vector domain: {vector_domain.value}. "
                f"Concept path hops: {path_hops}. "
                f"Retrieval snippets: {snippets_block}. "
                "Include language that reframes naming systems and anonymity. "
                f"Ending style target: {ending_type}."
            )
            return system_prompt, user_prompt

        system_prompt = (
            "You are a prose generator for a conceptual camera essay pipeline. "
            "Ground your paragraph in concrete material/process/system transitions, then widen to broader implications. "
            f"{constraints}"
        )
        user_prompt = (
            f"Object label: {object_label}. "
            f"Scene descriptors: {', '.join(descriptors[:4])}. "
            f"Selected vector domain: {vector_domain.value}. "
            f"Concept path hops: {path_hops}. "
            f"Top retrieval snippets: {snippets_block}. "
            f"Ending style target: {ending_type}."
        )
        return system_prompt, user_prompt

    def _generate_with_validation(self, *, system_prompt: str, user_prompt: str) -> str:
        if self.llm_client is None:
            return self._local_fallback(system_prompt=system_prompt)

        for _ in range(4):
            raw = self.llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
            normalized = self._normalize_output(raw)
            if self._is_valid_output(normalized):
                return normalized
            user_prompt = (
                f"{user_prompt} Previous output violated constraints. "
                "Regenerate with one paragraph only and 250-350 words, no newlines."
            )

        return self._local_fallback(system_prompt=system_prompt)

    def _normalize_output(self, text: str) -> str:
        return " ".join(text.replace("\n", " ").split())

    def _is_valid_output(self, text: str) -> bool:
        words = text.split()
        if "\n" in text or "\r" in text:
            return False
        return 250 <= len(words) <= 350

    def _local_fallback(self, *, system_prompt: str) -> str:
        safety = "Safety mode is active" in system_prompt
        if safety:
            text = (
                "At the tapped point, the scene resists becoming a dossier and stays an object among other objects, where edge, glare, texture, and motion are enough to trigger recognition without granting recognition authority over a person. "
                "The frame carries the old fantasy that naming is neutral, but each naming system drags a registry behind it, and each registry quietly asks who can be tracked, misread, over-policed, or allowed to disappear from view. "
                "Industrial optics trained devices to sort faces, vehicles, and habits as if identity were a stable surface feature, yet perception at scale behaves more like weather than law, swirling through thresholds, lens noise, compression artifacts, transmission standards, and institutional appetite. "
                "What appears like certainty dissolves into confidence scores, dashboard defaults, procurement choices, and procedural habits that mimic judgment while deferring responsibility, then circulate through staffing policy, street design, court language, and media myth. "
                "Anonymity in this setting is not secrecy for its own sake but a negotiated border where people refuse forced legibility, and where care means reducing extractive attention rather than improving capture rates. "
                "The ethics of looking therefore begins before identification, continues through uncertainty, and returns attention to atmosphere, grain, and shared vulnerability, reminding us that a humane image can hold context without converting a life into searchable inventory, and keeps accountability centered on systems rather than on biometric shortcuts that pretend context can be reduced to a label, even when systems claim neutrality through polished interfaces and technical jargon that hides human stakes, lived histories, and unequal risk."
            )
            return self._normalize_output(text)

        text = (
            "The tapped object appears ordinary at first, yet the pipeline context makes it read like a junction where material history, social protocol, and computational attention meet at once, pulling the eye from surface detail into process and from process into consequence. "
            "Its conceptual path suggests a relay rather than a single origin, with each hop adding a different pressure, from extraction and fabrication to routing, maintenance, and interpretation, so the thing in frame becomes a local interface for distant infrastructures that rarely appear together. "
            "The selected vector domain amplifies that shift because domain language converts local texture into broader questions about thresholds, costs, and feedback loops, and those abstractions refuse to stay abstract when translated back into labor schedules, transit delays, procurement standards, and environmental residue. "
            "Top retrieval snippets reinforce the pattern by naming concrete frictions, energetic tolls, and institutional assumptions that sit behind simple acts of seeing and using, including what gets measured, what gets discarded, and which uncertainties are quietly normalized. "
            "As attention widens, the object stops behaving like an isolated noun and starts behaving like a verb in a long sentence about coordination, where chemistry, logistics, interface conventions, and inherited metaphors keep rewriting one another across scales. "
            "By the time the paragraph closes, observation has become participatory: the object remains present, but it now carries a visible infrastructure of decisions, constraints, and consequences that continues unfolding after the frame, asking the viewer to treat looking as involvement rather than distance, while the chain of interpretation keeps extending into governance, maintenance budgets, and the stories communities tell about what technology is allowed to notice."
        )
        return self._normalize_output(text)
