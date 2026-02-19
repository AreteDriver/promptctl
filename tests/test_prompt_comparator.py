"""Tests for promptctl.prompt.comparator."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from promptctl.exceptions import LicenseError, PromptError
from promptctl.prompt.comparator import compare_models


def _mock_response(text="OK", input_tokens=10, output_tokens=5):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestCompareModels:
    def test_requires_pro(self, sample_template_path: Path):
        with pytest.raises(LicenseError, match="Pro license"):
            compare_models(str(sample_template_path))

    def test_default_models(
        self,
        sample_template_path: Path,
        pro_license_env: str,
        monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("response text")

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = compare_models(str(sample_template_path))

        assert len(result.entries) == 2
        assert result.prompt_name == "test-prompt"

    def test_custom_models(
        self,
        sample_template_path: Path,
        pro_license_env: str,
        monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = compare_models(
                str(sample_template_path),
                models="model-a, model-b, model-c",
            )

        assert len(result.entries) == 3
        assert result.entries[0].model == "model-a"
        assert result.entries[2].model == "model-c"

    def test_single_model_raises(
        self,
        sample_template_path: Path,
        pro_license_env: str,
    ):
        with pytest.raises(PromptError, match="at least 2"):
            compare_models(str(sample_template_path), models="only-one")

    def test_response_preview_truncated(
        self,
        sample_template_path: Path,
        pro_license_env: str,
        monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        long_text = "x" * 500
        mock_resp = _mock_response(long_text)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = compare_models(str(sample_template_path))

        for entry in result.entries:
            assert len(entry.response_preview) <= 200

    def test_entries_contain_cost(
        self,
        sample_template_path: Path,
        pro_license_env: str,
        monkeypatch,
    ):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response(input_tokens=100, output_tokens=50)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = compare_models(str(sample_template_path))

        for entry in result.entries:
            assert entry.input_tokens == 100
            assert entry.output_tokens == 50
            assert entry.latency_ms >= 0
