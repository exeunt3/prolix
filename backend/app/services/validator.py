from __future__ import annotations

import re

from app.models import GroundingPack

BANNED_CONCLUSION_MARKERS = ["in essence", "ultimately", "this perspective urges", "this shows that"]
SCALE_SHIFT_CUES = [
    "micron",
    "planetary",
    "centuries",
    "millions",
    "infrared",
    "global shipment",
    "kilometer",
    "millimeter",
]


def validate_prose(paragraph: str, grounding_pack: GroundingPack) -> list[str]:
    violations: list[str] = []
    words = paragraph.split()
    if not 250 <= len(words) <= 350:
        violations.append("word_count")
    if "\n" in paragraph:
        violations.append("single_paragraph")

    lowered = f" {paragraph.lower()} "
    if re.search(r"\b(you|your|we)\b", lowered):
        violations.append("no_second_person")

    visual_hits = 0
    for fact in grounding_pack.visual_facts:
        tokens = [tok for tok in re.findall(r"[a-z0-9]+", fact.lower()) if len(tok) > 2]
        if any(token in lowered for token in tokens[:4]):
            visual_hits += 1
    required_visual_hits = min(6, max(3, len(grounding_pack.visual_facts)))
    if visual_hits < required_visual_hits:
        violations.append("visual_facts")

    if any(marker in lowered for marker in BANNED_CONCLUSION_MARKERS):
        violations.append("conclusion_markers")

    if not any(cue in lowered for cue in SCALE_SHIFT_CUES):
        violations.append("scale_shift_cue")

    return violations
