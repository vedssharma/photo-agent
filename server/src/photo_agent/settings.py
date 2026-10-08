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
    """Key for the Claude API. Without it the app opens photos but the agent cannot edit."""

    anthropic_model: str = "claude-opus-5-5"
    """Claude model that drives the editing agent."""

    data_dir: Path = REPO_ROOT / ".data"
    """Where uploaded photos and their edit graphs are kept."""


@lru_cache
def get_settings() -> Settings:
    return Settings()
