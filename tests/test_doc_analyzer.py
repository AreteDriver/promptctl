"""Tests for promptctl.doc.analyzer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from promptctl.doc.analyzer import (
    _parse_json_response,
    analyze_document,
    ask_document,
    estimate_tokens,
    read_document,
    summarize_document,
)
from promptctl.exceptions import DocError


def _mock_response(text: str = "OK", input_tokens: int = 50, output_tokens: int = 100):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestReadDocument:
    def test_read_text_file(self, tmp_path: Path):
        f = tmp_path / "doc.txt"
        f.write_text("Hello world")
        content, wc = read_document(str(f))
        assert content == "Hello world"
        assert wc == 2

    def test_read_markdown_file(self, tmp_path: Path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nParagraph here.")
        content, wc = read_document(str(f))
        assert "# Title" in content
        assert wc == 4  # "# Title Paragraph here." split by whitespace

    def test_read_python_file(self, tmp_path: Path):
        f = tmp_path / "script.py"
        f.write_text("def foo():\n    pass\n")
        content, _ = read_document(str(f))
        assert "def foo" in content

    def test_read_json_file(self, tmp_path: Path):
        f = tmp_path / "data.json"
        f.write_text('{"key": "value"}')
        content, _ = read_document(str(f))
        assert '"key"' in content

    def test_read_yaml_file(self, tmp_path: Path):
        f = tmp_path / "config.yaml"
        f.write_text("name: test\n")
        content, _ = read_document(str(f))
        assert "name:" in content

    def test_read_csv_file(self, tmp_path: Path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c\n1,2,3\n")
        content, _ = read_document(str(f))
        assert "a,b,c" in content

    def test_unsupported_extension(self, tmp_path: Path):
        f = tmp_path / "doc.pdf"
        f.write_text("fake pdf")
        with pytest.raises(DocError, match="Unsupported file type"):
            read_document(str(f))

    def test_file_not_found(self):
        with pytest.raises(DocError, match="not found"):
            read_document("/nonexistent/file.txt")

    def test_not_a_file(self, tmp_path: Path):
        with pytest.raises(DocError, match="Not a file"):
            read_document(str(tmp_path))


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("a" * 400) == 100

    def test_empty(self):
        assert estimate_tokens("") == 0


class TestParseJsonResponse:
    def test_raw_json(self):
        data = _parse_json_response('{"summary": "OK"}')
        assert data["summary"] == "OK"

    def test_json_code_block(self):
        data = _parse_json_response('```json\n{"summary": "OK"}\n```')
        assert data["summary"] == "OK"

    def test_plain_code_block(self):
        data = _parse_json_response('```\n{"summary": "OK"}\n```')
        assert data["summary"] == "OK"

    def test_invalid_json(self):
        data = _parse_json_response("Not JSON at all")
        assert data == {}


class TestAnalyzeDocument:
    def test_basic(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("The quick brown fox jumps over the lazy dog.")

        response_data = {
            "summary": "A sentence about a fox",
            "key_points": ["Fox jumps"],
            "entities": ["fox", "dog"],
            "themes": ["animals"],
        }
        mock_resp = _mock_response(json.dumps(response_data))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = analyze_document(str(f))

        assert result.summary == "A sentence about a fox"
        assert result.key_points == ["Fox jumps"]
        assert result.entities == ["fox", "dog"]
        assert result.word_count == 9

    def test_model_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("Test content")
        mock_resp = _mock_response(json.dumps({"summary": "OK", "key_points": []}))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = analyze_document(str(f), model="claude-haiku-4-5-20251001")

        assert result.model == "claude-haiku-4-5-20251001"

    def test_file_not_found(self):
        with pytest.raises(DocError, match="not found"):
            analyze_document("/nonexistent.txt")


class TestAskDocument:
    def test_basic(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("The answer is 42.")

        response_data = {
            "answer": "42",
            "confidence": "high",
            "source_quotes": ["The answer is 42."],
        }
        mock_resp = _mock_response(json.dumps(response_data))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = ask_document(str(f), "What is the answer?")

        assert result.answer == "42"
        assert result.confidence == "high"

    def test_cache_control_in_system(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("Content here.")
        mock_resp = _mock_response(json.dumps({"answer": "OK"}))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            ask_document(str(f), "Question?")

        call_kwargs = mock_client.messages.create.call_args[1]
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}


class TestSummarizeDocument:
    def test_small_doc(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("A short document.")

        response_data = {
            "executive_summary": "A brief doc.",
            "sections": ["Only one section"],
        }
        mock_resp = _mock_response(json.dumps(response_data))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = summarize_document(str(f))

        assert result.executive_summary == "A brief doc."
        assert result.chunks_processed == 1

    def test_model_override(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        f = tmp_path / "doc.txt"
        f.write_text("Content.")
        mock_resp = _mock_response(json.dumps({"executive_summary": "OK", "sections": []}))

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = summarize_document(str(f), model="claude-haiku-4-5-20251001")

        assert result.model == "claude-haiku-4-5-20251001"
