"""Runtime settings for NaijaEstateAI (FastAPI + training).

Loads environment variables with sensible defaults. Central place for:
 - CORS origins
 - Log level
 - Prometheus toggle
 - Model metadata version
 - API auth token (simple) (optional)
"""
from __future__ import annotations
from pydantic import Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import List


class Settings(BaseSettings):
    app_name: str = "NaijaEstateAI API"
    log_level: str = Field("info", env="LOG_LEVEL")
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    api_token: str | None = Field(None, env="API_TOKEN")
    model_version: str = Field("1.0.0", env="MODEL_VERSION")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
