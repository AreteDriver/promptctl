"""Tests for promptctl.prompt.versioner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from promptctl.exceptions import PromptError
from promptctl.prompt.versioner import (
    list_versions,
    load_version,
    save_version,
)


@pytest.fixture
def template_file(tmp_path: Path) -> Path:
    """Create a sample template file."""
    f = tmp_path / "my-prompt.yaml"
    with open(f, "w") as fh:
        yaml.dump({"name": "my-prompt", "user": "Hello"}, fh)
    return f


class TestSaveVersion:
    def test_first_version(self, template_file: Path):
        version_num, path = save_version(str(template_file))
        assert version_num == 1
        assert path.exists()
        assert "v1.yaml" in path.name

    def test_increments_version(self, template_file: Path):
        save_version(str(template_file))
        version_num, path = save_version(str(template_file))
        assert version_num == 2
        assert "v2.yaml" in path.name

    def test_file_not_found(self):
        with pytest.raises(PromptError, match="not found"):
            save_version("/nonexistent.yaml")

    def test_free_tier_limit(self, template_file: Path):
        for _ in range(5):
            save_version(str(template_file))
        with pytest.raises(PromptError, match="Free tier"):
            save_version(str(template_file))

    def test_pro_tier_unlimited(self, template_file: Path, pro_license_env: str):
        for _ in range(10):
            version_num, _ = save_version(str(template_file))
        assert version_num == 10

    def test_content_preserved(self, template_file: Path):
        _, path = save_version(str(template_file))
        with open(path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "my-prompt"

    def test_uses_stem_for_name(self, tmp_path: Path):
        f = tmp_path / "analysis-prompt.yaml"
        with open(f, "w") as fh:
            yaml.dump({"name": "x", "user": "y"}, fh)
        _, path = save_version(str(f))
        assert "analysis-prompt" in str(path)


class TestListVersions:
    def test_empty(self):
        result = list_versions("nonexistent-prompt")
        assert result == []

    def test_lists_versions(self, template_file: Path):
        save_version(str(template_file))
        save_version(str(template_file))
        versions = list_versions("my-prompt")
        assert len(versions) == 2
        assert versions[0]["version"] == 1
        assert versions[1]["version"] == 2

    def test_sorted_order(self, template_file: Path):
        for _ in range(3):
            save_version(str(template_file))
        versions = list_versions("my-prompt")
        nums = [v["version"] for v in versions]
        assert nums == [1, 2, 3]


class TestLoadVersion:
    def test_load_existing(self, template_file: Path):
        save_version(str(template_file))
        path = load_version("my-prompt", 1)
        assert path.exists()

    def test_load_nonexistent(self, template_file: Path):
        with pytest.raises(PromptError, match="not found"):
            load_version("my-prompt", 99)
