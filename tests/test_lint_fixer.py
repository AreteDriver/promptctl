"""Tests for promptctl.lint.fixer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from promptctl.exceptions import LicenseError
from promptctl.lint.fixer import _parse_fix_response, fix_violations


def _mock_response(text: str = "OK", input_tokens: int = 50, output_tokens: int = 100):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestFixViolations:
    def test_requires_pro(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("not_name: oops")
        with pytest.raises(LicenseError, match="Pro license"):
            fix_violations(str(f))

    def test_no_violations(self, tmp_path, pro_license_env):
        f = tmp_path / "clean.yaml"
        f.write_text("name: test\nversion: '1'\nprompt: hello")
        result = fix_violations(str(f))
        assert result.original == result.fixed
        assert "No violations" in result.explanation

    def test_with_violations(self, tmp_path, pro_license_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "bad.yaml"
        f.write_text("prompt: hello")  # Missing name and version

        fix_data = json.dumps(
            {
                "fixed": "name: my-prompt\nversion: '1'\nprompt: hello",
                "explanation": "Added missing name and version fields.",
                "violations_addressed": ["L002", "L003"],
            }
        )
        mock_resp = _mock_response(fix_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp
            result = fix_violations(str(f))

        assert "name: my-prompt" in result.fixed
        assert "L002" in result.violations_addressed
        assert result.input_tokens == 50

    def test_model_override(self, tmp_path, pro_license_env, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "bad.yaml"
        f.write_text("prompt: hello")

        fix_data = json.dumps(
            {
                "fixed": "name: t\nversion: '1'\nprompt: hello",
                "explanation": "Fixed.",
                "violations_addressed": ["L002"],
            }
        )
        mock_resp = _mock_response(fix_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp
            result = fix_violations(str(f), model="claude-haiku-4-5-20251001")

        assert result.model == "claude-haiku-4-5-20251001"


class TestParseFixResponse:
    def test_raw_json(self):
        data = _parse_fix_response('{"fixed": "ok"}')
        assert data["fixed"] == "ok"

    def test_json_code_block(self):
        data = _parse_fix_response('```json\n{"fixed": "ok"}\n```')
        assert data["fixed"] == "ok"

    def test_plain_code_block(self):
        data = _parse_fix_response('```\n{"fixed": "ok"}\n```')
        assert data["fixed"] == "ok"

    def test_invalid_json(self):
        data = _parse_fix_response("Not JSON")
        assert data == {}
