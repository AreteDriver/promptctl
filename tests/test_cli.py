"""Tests for promptctl.cli — full CLI integration tests."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from typer.testing import CliRunner

from promptctl.cli import app

runner = CliRunner()


class TestVersion:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "promptctl" in result.output
        assert "0.1.0" in result.output

    def test_short_version_flag(self):
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestStatus:
    def test_status_no_key(self):
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "not set" in result.output
        assert "free" in result.output

    def test_status_with_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-12345678")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "sk-ant-t" in result.output

    def test_status_with_short_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "short")
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "***" in result.output

    def test_status_with_pro_license(self, pro_license_env):
        result = runner.invoke(app, ["status"])
        assert "pro" in result.output


class TestConfigCommands:
    def test_config_init(self):
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 0
        assert "created" in result.output.lower()

    def test_config_init_already_exists(self):
        runner.invoke(app, ["config", "init"])
        result = runner.invoke(app, ["config", "init"])
        assert result.exit_code == 1
        assert "already exists" in result.output.lower()

    def test_config_set(self):
        result = runner.invoke(app, ["config", "set", "model", "claude-haiku-4-5-20251001"])
        assert result.exit_code == 0
        assert "Set model" in result.output

    def test_config_set_invalid_float(self):
        result = runner.invoke(app, ["config", "set", "temperature", "not-a-number"])
        assert result.exit_code == 1

    def test_config_show(self):
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "model" in result.output

    def test_config_show_masks_api_key(self, tmp_config_dir):
        config_path = tmp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"api_key": "test-dummy-key-for-masking"}, f)
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "test-dummy-key-for-masking" not in result.output
        assert "..." in result.output


def _mock_response(text="OK", input_tokens=10, output_tokens=5):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestPromptRunCommand:
    def test_run_basic(self, sample_template_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("Hello from Claude!")

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = runner.invoke(app, ["prompt", "run", str(sample_template_path)])

        assert result.exit_code == 0
        assert "Hello from Claude!" in result.output

    def test_run_json_output(self, sample_template_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("JSON response")

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = runner.invoke(app, ["prompt", "run", str(sample_template_path), "--json"])

        assert result.exit_code == 0
        # JSON output should be parseable
        output = result.output.strip()
        assert "JSON response" in output

    def test_run_file_not_found(self):
        result = runner.invoke(app, ["prompt", "run", "/nonexistent.yaml"])
        assert result.exit_code == 1
        assert "Error" in result.output


class TestPromptVersionCommand:
    def test_version_save(self, sample_template_path):
        result = runner.invoke(app, ["prompt", "version", str(sample_template_path)])
        assert result.exit_code == 0
        assert "v1" in result.output

    def test_version_file_not_found(self):
        result = runner.invoke(app, ["prompt", "version", "/nonexistent.yaml"])
        assert result.exit_code == 1


class TestPromptHistoryCommand:
    def test_history_empty(self):
        result = runner.invoke(app, ["prompt", "history", "nonexistent"])
        assert result.exit_code == 0
        assert "No versions" in result.output

    def test_history_with_versions(self, sample_template_path):
        runner.invoke(app, ["prompt", "version", str(sample_template_path)])
        # The template stem determines the name
        result = runner.invoke(app, ["prompt", "history", "template"])
        assert result.exit_code == 0
        assert "v1" in result.output


class TestPromptCompareCommand:
    def test_compare_requires_pro(self, sample_template_path):
        result = runner.invoke(app, ["prompt", "compare", str(sample_template_path)])
        assert result.exit_code == 1
        assert "Pro license" in result.output


class TestReviewDiffCommand:
    def test_review_empty_diff(self):
        with patch("promptctl.review.differ.subprocess.run") as mock_run:
            mock_run.return_value = type("CP", (), {"stdout": "", "returncode": 0})()
            result = runner.invoke(app, ["review", "diff"])
        assert result.exit_code == 0
        assert "No staged changes" in result.output

    def test_review_diff_with_content(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        review_json = json.dumps(
            {
                "summary": "Looks good",
                "findings": [],
            }
        )
        mock_resp = _mock_response(review_json)

        with (
            patch("promptctl.review.differ.subprocess.run") as mock_run,
            patch("promptctl.client.anthropic.Anthropic") as mock_cls,
        ):
            mock_run.return_value = type("CP", (), {"stdout": "diff content", "returncode": 0})()
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = runner.invoke(app, ["review", "diff"])

        assert result.exit_code == 0
        assert "No issues found" in result.output


class TestReviewFileCommand:
    def test_review_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code_file = tmp_path / "test.py"
        code_file.write_text("print('hello')")

        review_json = json.dumps(
            {
                "summary": "Simple script",
                "findings": [
                    {
                        "severity": "info",
                        "category": "style",
                        "message": "Consider adding docstring",
                    }
                ],
            }
        )
        mock_resp = _mock_response(review_json)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = runner.invoke(app, ["review", "file", str(code_file)])

        assert result.exit_code == 0
        assert "docstring" in result.output

    def test_review_file_not_found(self):
        result = runner.invoke(app, ["review", "file", "/nonexistent.py"])
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_review_file_json_output(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        code_file = tmp_path / "test.py"
        code_file.write_text("x = 1")

        review_json = json.dumps({"summary": "OK", "findings": []})
        mock_resp = _mock_response(review_json)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = runner.invoke(app, ["review", "file", str(code_file), "--json"])

        assert result.exit_code == 0
