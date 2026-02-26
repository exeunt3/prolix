from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


ALLOWED_RELATIONS = {
    "is_a",
    "made_of",
    "derived_from",
    "enabled_by",
    "resembles",
    "computed_by",
    "constrained_by",
    "historically_entangled_with",
    "scaled_to",
    "embedded_in",
    "feeds",
    "mutates_into",
    "stabilized_by",
    "narrated_as",
}


class VectorDomain(str, Enum):
    THERMODYNAMICS_ENTROPY = "THERMODYNAMICS_ENTROPY"
    CHEMISTRY_TRANSFORMATION = "CHEMISTRY_TRANSFORMATION"
    PHYSICS_PARADOXES = "PHYSICS_PARADOXES"
    COSMOLOGY = "COSMOLOGY"
    DISTRIBUTED_INTELLIGENCE = "DISTRIBUTED_INTELLIGENCE"
    EVOLUTIONARY_SIGNALING = "EVOLUTIONARY_SIGNALING"
    PLANT_AGENCY = "PLANT_AGENCY"
    FEEDBACK_CONTROL = "FEEDBACK_CONTROL"
    EMERGENCE_OBSERVER = "EMERGENCE_OBSERVER"
    INFRASTRUCTURE_LOGISTICS = "INFRASTRUCTURE_LOGISTICS"
    OBSCURE_HISTORY_AGENTS = "OBSCURE_HISTORY_AGENTS"
    HERESY_SECTS = "HERESY_SECTS"
    MYSTERIOUS_EVENTS = "MYSTERIOUS_EVENTS"
    COSMOLOGY_MYTH = "COSMOLOGY_MYTH"
    CULT_PRACTICES = "CULT_PRACTICES"


class Hop(BaseModel):
    node: str
    rel: str

    @field_validator("rel")
    @classmethod
    def rel_allowed(cls, value: str) -> str:
        if value not in ALLOWED_RELATIONS:
            raise ValueError(f"Unsupported relation: {value}")
        return value


class GenerateResponse(BaseModel):
    paragraph_text: str
    trace_id: UUID


class DeepenRequest(BaseModel):
    trace_id: UUID


class TraceRecord(BaseModel):
    trace_id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    object_label: str
    vector_domain: VectorDomain
    concept_path: list[Hop]
    paragraph_text: str
    ending_type: str
    safety_flag: bool = False
    dark_flag: bool = False


class NarrationResult(BaseModel):
    paragraph_text: str
    path_used: list[Hop]
    ending_type: str


class GenerateInput(BaseModel):
    tap_x: float
    tap_y: float
    image_b64: str | None = None


class RetrievalSnippet(BaseModel):
    title: str
    excerpt: str
    domain_tag: str
    source_id: str


class GroundingResult(BaseModel):
    object_label: str
    scene_descriptors: list[str]
    confidence: float
    safety_face_or_plate: bool = False


class DriftPlan(BaseModel):
    vector_domain: VectorDomain
    concept_path: list[Hop]
    dark_flag: bool = False
    attractor: str | None = None


class TraceEnvelope(BaseModel):
    trace: TraceRecord
    retrieval: list[RetrievalSnippet]
    metadata: dict[str, Any] = Field(default_factory=dict)
