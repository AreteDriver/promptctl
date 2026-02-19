"""Tests for promptctl.prompt.runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from promptctl.exceptions import PromptError
from promptctl.models import PromptResult, PromptTemplate
from promptctl.prompt.runner import interpolate, load_template, run_prompt


class TestLoadTemplate:
    def test_valid_template(self, sample_template_path: Path):
        t = load_template(str(sample_template_path))
        assert t.name == "test-prompt"
        assert t.system == "You are a helpful assistant."
        assert "{name}" in t.user
        assert t.variables["name"] == "World"

    def test_file_not_found(self):
        with pytest.raises(PromptError, match="not found"):
            load_template("/nonexistent/template.yaml")

    def test_invalid_yaml(self, tmp_path: Path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": : {bad yaml {{")
        with pytest.raises(PromptError, match="Invalid YAML"):
            load_template(str(bad))

    def test_not_a_mapping(self, tmp_path: Path):
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n")
        with pytest.raises(PromptError, match="mapping"):
            load_template(str(bad))

    def test_missing_name(self, tmp_path: Path):
        bad = tmp_path / "no_name.yaml"
        with open(bad, "w") as f:
            yaml.dump({"user": "hello"}, f)
        with pytest.raises(PromptError, match="name"):
            load_template(str(bad))

    def test_missing_user(self, tmp_path: Path):
        bad = tmp_path / "no_user.yaml"
        with open(bad, "w") as f:
            yaml.dump({"name": "test"}, f)
        with pytest.raises(PromptError, match="user"):
            load_template(str(bad))

    def test_minimal_template(self, tmp_path: Path):
        minimal = tmp_path / "minimal.yaml"
        with open(minimal, "w") as f:
            yaml.dump({"name": "basic", "user": "Hello"}, f)
        t = load_template(str(minimal))
        assert t.name == "basic"
        assert t.system == ""
        assert t.variables == {}


class TestInterpolate:
    def test_basic_interpolation(self):
        t = PromptTemplate(
            name="test",
            user="Hello, {name}! Welcome to {place}.",
            variables={"name": "Alice", "place": "Wonderland"},
        )
        assert interpolate(t) == "Hello, Alice! Welcome to Wonderland."

    def test_no_variables(self):
        t = PromptTemplate(name="test", user="No vars here.")
        assert interpolate(t) == "No vars here."

    def test_missing_variable_left_alone(self):
        t = PromptTemplate(
            name="test",
            user="Hello, {name}!",
            variables={},
        )
        assert interpolate(t) == "Hello, {name}!"


def _mock_response(text="OK", input_tokens=10, output_tokens=5):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestRunPrompt:
    def test_non_streaming(self, sample_template_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("Generated text")

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = run_prompt(str(sample_template_path))

        assert result.response == "Generated text"
        assert isinstance(result, PromptResult)

    def test_model_override(self, sample_template_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            run_prompt(str(sample_template_path), model="claude-haiku-4-5-20251001")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_temperature_override(self, sample_template_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            run_prompt(str(sample_template_path), temperature=0.0)

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0

    def test_interpolated_variables(self, sample_template_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            run_prompt(str(sample_template_path))

        call_kwargs = mock_client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "World" in user_content
        assert "{name}" not in user_content

    def test_file_not_found(self):
        with pytest.raises(PromptError, match="not found"):
            run_prompt("/nonexistent.yaml")
