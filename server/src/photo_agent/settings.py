"""Runtime configuration, read from environment variables and the repo-root .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    anthropic_api_key: SecretStr | None = None
    """Key for the Claude API. Optional until the agent service lands in Phase 1."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
