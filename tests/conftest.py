"""Shared test fixtures for promptctl."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate each test: temp config dir, no license, no API key."""
    config_dir = tmp_path / ".promptctl"
    config_dir.mkdir()
    monkeypatch.setenv("PROMPTCTL_DIR", str(config_dir))
    monkeypatch.delenv("PROMPTCTL_LICENSE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def tmp_config_dir(tmp_path: Path) -> Path:
    """Pre-created config directory."""
    d = tmp_path / ".promptctl"
    d.mkdir(exist_ok=True)
    return d


@pytest.fixture
def pro_license_env(monkeypatch: pytest.MonkeyPatch) -> str:
    """Set a valid Pro license key in environment."""
    from promptctl.licensing import generate_key

    key = generate_key()
    monkeypatch.setenv("PROMPTCTL_LICENSE", key)
    return key


@pytest.fixture
def sample_template_path(tmp_path: Path) -> Path:
    """Create a sample prompt template YAML file."""
    import yaml

    template = {
        "name": "test-prompt",
        "system": "You are a helpful assistant.",
        "user": "Hello, {name}!",
        "variables": {"name": "World"},
        "model": "claude-sonnet-4-20250514",
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    path = tmp_path / "template.yaml"
    with open(path, "w") as f:
        yaml.dump(template, f)
    return path
