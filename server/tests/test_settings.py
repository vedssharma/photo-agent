from pathlib import Path

import pytest

from photo_agent.settings import Settings


def test_reads_key_from_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")

    settings = Settings(_env_file=env_file)

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-from-file"


def test_environment_overrides_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env")
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=sk-ant-from-file\n")

    settings = Settings(_env_file=env_file)

    assert settings.anthropic_api_key is not None
    assert settings.anthropic_api_key.get_secret_value() == "sk-ant-from-env"


def test_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert Settings(_env_file=None).anthropic_api_key is None


def test_blank_key_in_env_file_counts_as_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("ANTHROPIC_API_KEY=\n")

    assert Settings(_env_file=env_file).anthropic_api_key is None


def test_key_is_not_leaked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    assert "sk-ant-secret" not in repr(Settings(_env_file=None))
