from __future__ import annotations

from functools import lru_cache

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vision_provider: str = "mock"
    vision_model: str = "gpt-4o-mini"
    vision_api_key: SecretStr | None = None
    vision_endpoint: AnyHttpUrl | None = None

    text_provider: str = "mock"
    text_model: str = "gpt-4o-mini"
    text_api_key: SecretStr | None = None
    text_endpoint: AnyHttpUrl | None = None

    ai_request_timeout_seconds: float = 20.0
    ai_max_retries: int = 2

    def validate_provider_secrets(self) -> list[str]:
        missing: list[str] = []
        if self.vision_provider.lower() != "mock" and self.vision_api_key is None:
            missing.append("VISION_API_KEY is required when VISION_PROVIDER is not 'mock'.")
        if self.text_provider.lower() != "mock" and self.text_api_key is None:
            missing.append("TEXT_API_KEY is required when TEXT_PROVIDER is not 'mock'.")
        return missing


@lru_cache
def get_settings() -> Settings:
    return Settings()
