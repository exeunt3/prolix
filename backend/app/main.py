from __future__ import annotations

import binascii

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.db.storage import TraceStore
from app.models import DeepenRequest, GenerateResponse, Hop, TraceRecord, VectorDomain
from app.services.drift import DriftEngine
from app.services.grounding import (
    GroundingService,
    OpenAIVisionGroundingProvider,
    VisionProviderSettings,
    decode_image_b64,
)
from app.services.narration import NarrationService
from app.services.retrieval import RetrievalService

app = FastAPI(title="Prolix API", version="0.1.0")
app.mount("/web-static", StaticFiles(directory="app/static/web"), name="web-static")
store = TraceStore()
grounding_provider = OpenAIVisionGroundingProvider(settings=VisionProviderSettings.from_env())
grounder = GroundingService(provider=grounding_provider)
retrieval = RetrievalService(corpus_dir="corpus")
drift = DriftEngine()
narrator = NarrationService()


def _safety_path(anchor: str) -> list[Hop]:
    return [
        Hop(node=anchor, rel="is_a"),
        Hop(node="pattern recognition", rel="enabled_by"),
        Hop(node="classification thresholds", rel="constrained_by"),
        Hop(node="institutional anonymity", rel="embedded_in"),
        Hop(node="ethics of naming", rel="historically_entangled_with"),
        Hop(node="scene-level return signal", rel="feeds"),
    ]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/web")
def web_app() -> FileResponse:
    return FileResponse("app/static/web/index.html")


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    tap_x: float = Form(...),
    tap_y: float = Form(...),
    image: UploadFile | None = File(default=None),
    image_b64: str | None = Form(default=None),
) -> GenerateResponse:
    normalized_image_bytes: bytes | None = None
    if image is not None:
        normalized_image_bytes = await image.read()
    elif image_b64:
        try:
            normalized_image_bytes = decode_image_b64(image_b64)
        except (ValueError, binascii.Error):
            normalized_image_bytes = None

    grounding = grounder.ground(normalized_image_bytes, tap_x=tap_x, tap_y=tap_y)

    if grounding.safety_face_or_plate:
        domain = VectorDomain.FEEDBACK_CONTROL
        concept_path = _safety_path(grounding.object_label)
        dark_flag = False
    else:
        domain = drift.choose_domain()
        plan = drift.plan(grounding.object_label, domain, force_safe=False)
        concept_path = plan.concept_path
        dark_flag = plan.dark_flag

    snippets = retrieval.retrieve(grounding.object_label, domain, k=12)
    narration = narrator.narrate(
        object_label=grounding.object_label,
        descriptors=grounding.scene_descriptors,
        vector_domain=domain,
        path=concept_path,
        snippets=snippets,
        safety_redirect=grounding.safety_face_or_plate,
    )
    trace = TraceRecord(
        object_label=grounding.object_label,
        vector_domain=domain,
        concept_path=concept_path,
        paragraph_text=narration.paragraph_text,
        ending_type=narration.ending_type,
        safety_flag=grounding.safety_face_or_plate,
        dark_flag=dark_flag,
    )
    store.insert_trace(trace)
    return GenerateResponse(paragraph_text=narration.paragraph_text, trace_id=trace.trace_id)


@app.post("/deepen", response_model=GenerateResponse)
def deepen(request: DeepenRequest) -> GenerateResponse:
    previous = store.get_trace(request.trace_id)
    if previous is None:
        raise HTTPException(status_code=404, detail="trace_id not found")

    anchor_nodes = previous.concept_path[-2:] if len(previous.concept_path) > 1 else previous.concept_path
    anchor = " / ".join([h.node for h in anchor_nodes])

    if previous.safety_flag:
        domain = VectorDomain.FEEDBACK_CONTROL
        concept_path = _safety_path(anchor)
        dark_flag = False
    else:
        domain = drift.choose_domain(previous=previous.vector_domain)
        plan = drift.plan(anchor, domain, force_safe=False)
        concept_path = plan.concept_path
        dark_flag = plan.dark_flag

    snippets = retrieval.retrieve(previous.object_label, domain, k=12)
    narration = narrator.narrate(
        object_label=previous.object_label,
        descriptors=["carried-over endpoint", "continuation"],
        vector_domain=domain,
        path=concept_path,
        snippets=snippets,
        safety_redirect=previous.safety_flag,
    )
    trace = TraceRecord(
        object_label=previous.object_label,
        vector_domain=domain,
        concept_path=concept_path,
        paragraph_text=narration.paragraph_text,
        ending_type=narration.ending_type,
        safety_flag=previous.safety_flag,
        dark_flag=dark_flag,
    )
    store.insert_trace(trace)
    return GenerateResponse(paragraph_text=narration.paragraph_text, trace_id=trace.trace_id)
