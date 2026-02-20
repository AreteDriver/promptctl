"""Tests for promptctl.doc.chunker."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from promptctl.doc.chunker import chunk_text, map_reduce_summarize


def _mock_response(text: str = "OK", input_tokens: int = 50, output_tokens: int = 100):
    content = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content], usage=usage)


class TestChunkText:
    def test_short_text_single_chunk(self):
        result = chunk_text("hello", chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == "hello"

    def test_long_text_multiple_chunks(self):
        text = "a" * 300
        result = chunk_text(text, chunk_size=100, overlap=10)
        assert len(result) >= 3

    def test_overlap_between_chunks(self):
        text = "abcdefghij" * 30  # 300 chars
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert len(result) >= 2
        # Last 20 chars of chunk 1 should be first 20 of chunk 2
        assert result[0][-20:] == result[1][:20]

    def test_empty_text(self):
        result = chunk_text("", chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0] == ""


class TestMapReduceSummarize:
    def test_two_chunks(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        # Map responses (one per chunk)
        map_resp1 = _mock_response(json.dumps({"section_summary": "Section 1 about AI"}), 100, 50)
        map_resp2 = _mock_response(json.dumps({"section_summary": "Section 2 about ML"}), 100, 50)
        # Reduce response
        reduce_resp = _mock_response(
            json.dumps(
                {
                    "executive_summary": "A paper about AI and ML",
                    "sections": ["AI overview", "ML details"],
                }
            ),
            80,
            60,
        )

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = [map_resp1, map_resp2, reduce_resp]

            # Small chunk size to force 2 chunks
            text = "x" * 200
            with patch("promptctl.doc.chunker.CHUNK_CHAR_SIZE", 120):
                with patch("promptctl.doc.chunker.OVERLAP_CHARS", 10):
                    result = map_reduce_summarize(text, "claude-sonnet-4-20250514", 50)

        assert result.executive_summary == "A paper about AI and ML"
        assert result.chunks_processed >= 2
        assert result.input_tokens == 280  # 100 + 100 + 80
        assert result.output_tokens == 160  # 50 + 50 + 60

    def test_cost_aggregation(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        map_resp = _mock_response(json.dumps({"section_summary": "summary"}), 100, 50)
        reduce_resp = _mock_response(
            json.dumps({"executive_summary": "final", "sections": []}), 80, 40
        )

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = [map_resp, reduce_resp]

            result = map_reduce_summarize("short", "claude-sonnet-4-20250514", 1)

        assert result.cost_usd > 0
        assert result.model == "claude-sonnet-4-20250514"

    def test_invalid_json_in_map_phase(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")

        map_resp = _mock_response("Not JSON at all", 50, 25)
        reduce_resp = _mock_response(
            json.dumps({"executive_summary": "synthesized", "sections": []}),
            50,
            25,
        )

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = [map_resp, reduce_resp]

            result = map_reduce_summarize("short", "claude-sonnet-4-20250514", 1)

        # Should still produce a result using raw text fallback
        assert result.executive_summary == "synthesized"
