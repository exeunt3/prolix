from app.services.grounding import GroundingService


class SuccessfulGroundingProvider:
    def ground(self, image_b64: str | None, tap_x: float, tap_y: float, timeout_s: float) -> dict:
        return {
            "object_label": "wire",
            "scene_descriptors": ["descriptor-a", "descriptor-b"],
            "confidence": 0.91,
            "safety_face_or_plate": False,
        }


class TimeoutGroundingProvider:
    def ground(self, image_b64: str | None, tap_x: float, tap_y: float, timeout_s: float) -> dict:
        raise TimeoutError("provider timeout")


class MalformedGroundingProvider:
    def ground(self, image_b64: str | None, tap_x: float, tap_y: float, timeout_s: float) -> dict:
        return {"unexpected": "shape"}


def test_grounding_service_uses_provider_response_on_success() -> None:
    service = GroundingService(provider_client=SuccessfulGroundingProvider())

    result = service.ground("aW1hZ2U=", tap_x=0.2, tap_y=0.8)

    assert result.object_label == "wire"
    assert result.scene_descriptors == ["descriptor-a", "descriptor-b"]
    assert result.confidence == 0.91
    assert result.safety_face_or_plate is False


def test_grounding_service_falls_back_when_provider_times_out() -> None:
    service = GroundingService(provider_client=TimeoutGroundingProvider())

    result = service.ground("d2lyZQ==", tap_x=0.4, tap_y=0.6)

    assert result.object_label == "object"
    assert result.scene_descriptors[0].startswith("tap near x=")


def test_grounding_service_falls_back_on_malformed_provider_response() -> None:
    service = GroundingService(provider_client=MalformedGroundingProvider())

    result = service.ground("bGVhZg==", tap_x=0.1, tap_y=0.9)

    assert result.object_label == "object"
    assert result.confidence == 0.4


def test_grounding_service_uses_internal_fallback_without_provider() -> None:
    service = GroundingService(provider_client=None)

    result = service.ground("ZmFjZSBwbGF0ZQ==", tap_x=0.1, tap_y=0.2)

    assert result.safety_face_or_plate is True
    assert len(result.scene_descriptors) == 3
