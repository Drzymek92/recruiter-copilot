"""Shared fixtures. Tests never need a microphone, a GPU, or a network (F8 post-hoc path)."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_session_dir() -> Path:
    return PROJECT_ROOT / "examples" / "sample_session"


@pytest.fixture(autouse=True)
def no_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """SI1: a test must never accidentally reach a cloud provider."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
