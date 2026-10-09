"""Runtime configuration, read from environment variables and the repo-root .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    model_worker: Literal["process", "inline"] = "process"
    """How AI model jobs run: in a separate worker process (the default, so a slow or
    crashing model never stalls the web server), or on a thread inside it (for tests)."""

    model_backends: Literal["auto", "classical"] = "auto"
    """"auto" uses the open-source models when their packages are installed (`uv sync
    --extra models`), falling back to classical computer vision; "classical" never loads
    model weights."""

    model_device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    """Where models run; "auto" picks a GPU when there is one."""

    model_dir: Path | None = None
    """Where downloaded model weights are cached; defaults to `<data_dir>/models`."""

    @property
    def model_cache_dir(self) -> Path:
        return self.model_dir or self.data_dir / "models"


@lru_cache
def get_settings() -> Settings:
    return Settings()
