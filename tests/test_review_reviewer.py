"""Tests for promptctl.review.reviewer."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from promptctl.review.reviewer import _parse_review_response, review_code


def _mock_review_response(data: dict):
    """Create a mock Anthropic response with JSON content."""
    text = json.dumps(data)
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=50, output_tokens=100)
    return SimpleNamespace(content=[content], usage=usage)


class TestParseReviewResponse:
    def test_valid_json(self):
        data = {
            "summary": "Looks good",
            "findings": [
                {
                    "severity": "warning",
                    "category": "style",
                    "file": "main.py",
                    "line": 10,
                    "message": "Long line",
                    "suggestion": "Break it up",
                }
            ],
        }
        report = _parse_review_response(json.dumps(data))
        assert report.summary == "Looks good"
        assert len(report.findings) == 1
        assert report.findings[0].severity == "warning"
        assert report.findings[0].category == "style"

    def test_json_in_code_block(self):
        text = '```json\n{"summary": "OK", "findings": []}\n```'
        report = _parse_review_response(text)
        assert report.summary == "OK"
        assert report.findings == []

    def test_json_in_plain_code_block(self):
        text = '```\n{"summary": "OK", "findings": []}\n```'
        report = _parse_review_response(text)
        assert report.summary == "OK"

    def test_invalid_json_fallback(self):
        report = _parse_review_response("This is not JSON at all")
        assert report.summary.startswith("This is not JSON")
        assert report.findings == []

    def test_malformed_finding_skipped(self):
        data = {
            "summary": "Mixed",
            "findings": [
                {"severity": "error", "category": "security", "message": "Good finding"},
                "not a dict",
            ],
        }
        report = _parse_review_response(json.dumps(data))
        assert len(report.findings) == 1

    def test_empty_findings(self):
        data = {"summary": "No issues found.", "findings": []}
        report = _parse_review_response(json.dumps(data))
        assert report.error_count == 0

    def test_cost_and_tokens_stored(self):
        data = {"summary": "OK", "findings": []}
        report = _parse_review_response(
            json.dumps(data),
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
        )
        assert report.model == "claude-sonnet-4-20250514"
        assert report.input_tokens == 100
        assert report.output_tokens == 50
        assert report.cost_usd > 0

    def test_finding_defaults(self):
        data = {
            "summary": "x",
            "findings": [{"message": "bare finding"}],
        }
        report = _parse_review_response(json.dumps(data))
        f = report.findings[0]
        assert f.severity == "info"
        assert f.category == "style"
        assert f.file == ""
        assert f.line == 0


class TestReviewCode:
    def test_basic_review(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        review_data = {
            "summary": "Code looks good",
            "findings": [
                {
                    "severity": "info",
                    "category": "style",
                    "message": "Consider adding docstring",
                }
            ],
        }
        mock_resp = _mock_review_response(review_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            report = review_code("def foo(): pass")

        assert report.summary == "Code looks good"
        assert len(report.findings) == 1

    def test_source_file_included(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        review_data = {"summary": "OK", "findings": []}
        mock_resp = _mock_review_response(review_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            review_code("x = 1", source_file="app.py")

        call_kwargs = mock_client.messages.create.call_args[1]
        user_content = call_kwargs["messages"][0]["content"]
        assert "File: app.py" in user_content

    def test_model_override(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        review_data = {"summary": "OK", "findings": []}
        mock_resp = _mock_review_response(review_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            review_code("x = 1", model="claude-haiku-4-5-20251001")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-haiku-4-5-20251001"

    def test_temperature_zero(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        review_data = {"summary": "OK", "findings": []}
        mock_resp = _mock_review_response(review_data)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            review_code("x = 1")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.0
