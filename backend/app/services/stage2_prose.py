from __future__ import annotations

from app.models import GroundingPack, OutlinePlan, ProseResult
from app.services.llm_client import LLMClient
from app.services.prompts import STAGE2_SYSTEM_PROMPT, build_stage2_user_prompt
from app.services.validator import validate_prose


def generate_prose_stage2(*, llm_client: LLMClient | None, outline: OutlinePlan, grounding_pack: GroundingPack) -> ProseResult:
    if llm_client is None:
        paragraph = _fallback_paragraph(grounding_pack)
        violations = validate_prose(paragraph, grounding_pack)
        return ProseResult(paragraph_text=paragraph, validation_passed=not violations, violations=violations)

    user_prompt = build_stage2_user_prompt(outline=outline, grounding_pack=grounding_pack)
    raw = llm_client.generate(system_prompt=STAGE2_SYSTEM_PROMPT, user_prompt=user_prompt)
    paragraph = " ".join(raw.replace("\n", " ").split())
    violations = validate_prose(paragraph, grounding_pack)
    if not violations:
        return ProseResult(paragraph_text=paragraph, validation_passed=True, violations=[])

    repair_prompt = (
        f"Repair the paragraph. Violations: {', '.join(violations)}. "
        f"Grounding facts: {', '.join(grounding_pack.visual_facts)}. "
        "Return one paragraph only and satisfy all constraints."
    )
    raw_repair = llm_client.generate(system_prompt=STAGE2_SYSTEM_PROMPT, user_prompt=repair_prompt)
    paragraph_repair = " ".join(raw_repair.replace("\n", " ").split())
    repair_violations = validate_prose(paragraph_repair, grounding_pack)
    return ProseResult(
        paragraph_text=paragraph_repair,
        validation_passed=not repair_violations,
        violations=repair_violations,
    )


def _fallback_paragraph(grounding_pack: GroundingPack) -> str:
    facts = grounding_pack.visual_facts + ["a thin film of residue", "a few millimeters of worn edge", "patina from repeated handling"]
    facts = facts[:9]
    text = (
        f"The {grounding_pack.anchor_label} stays in frame as a concrete surface rather than an idea: {facts[0]}, {facts[1]}, and {facts[2]} hold in present light while {facts[3]} keeps the anchor pinned near {grounding_pack.anchor_scene_context}. "
        f"Its material behavior reads as {grounding_pack.anchor_material_guess}, where {facts[4]} and {facts[5]} show how pressure, heat, and handling leave a thin film and a few millimeters of softened edge over time, and where {facts[6]} records older contact as a visible timeline. "
        "What looks static is enabled by extraction, smelting, machining tolerances, adhesive chemistry, and maintenance rituals; global shipment schedules, port bottlenecks, and warehouse humidity decide whether this surface arrives matte or glossy, and whether the seam keeps its bite or flakes early under vibration. "
        "At micron scale the texture becomes ridges and pits with edge burrs, while across centuries the same pattern sits inside mining histories, labor shifts, refinery emissions, and disposal routes that recirculate residue into water and air before re-entering new commodity loops. "
        "That scale jump does not abandon the object; it keeps the anchor as an index of computed quality control, where scanners, confidence thresholds, and replacement rules translate scratches into pass/fail signals and feed purchasing cycles, repair tickets, and inventory forecasts. "
        f"The scene returns without closure: {grounding_pack.anchor_description} remains present, held between use and decay, still carrying patina, measurement, and unresolved agency in the same handspan of matter, with no clean moral and no final release from the object itself."
    )
    return " ".join(text.split())
