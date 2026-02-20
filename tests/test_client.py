"""Tests for promptctl.client."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from promptctl.client import (
    get_client,
    send_message,
    send_message_streaming,
    send_message_with_tools,
)
from promptctl.exceptions import ClientError


def _mock_response(text: str = "Hello!", input_tokens: int = 10, output_tokens: int = 5):
    """Create a mock Anthropic Message response."""
    content_block = SimpleNamespace(type="text", text=text)
    usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)
    return SimpleNamespace(content=[content_block], usage=usage)


class TestGetClient:
    def test_no_api_key_raises(self):
        with pytest.raises(ClientError, match="No API key"):
            get_client()

    def test_with_api_key(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key")
        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            client = get_client()
            mock_cls.assert_called_once_with(api_key="sk-ant-test-key")
            assert client is mock_cls.return_value


class TestSendMessage:
    def test_basic_call(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("Test response", 15, 8)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = send_message(
                model="claude-sonnet-4-20250514",
                system="Be helpful.",
                messages=[{"role": "user", "content": "Hello"}],
            )

        assert result.response == "Test response"
        assert result.input_tokens == 15
        assert result.output_tokens == 8
        assert result.model == "claude-sonnet-4-20250514"
        assert result.latency_ms > 0

    def test_no_system_prompt(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = send_message(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hi"}],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert "system" not in call_kwargs
        assert result.response == "Hello!"

    def test_empty_messages_default(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = send_message(model="claude-sonnet-4-20250514")

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["messages"] == []
        assert result.response == "Hello!"

    def test_custom_temperature_and_max_tokens(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            send_message(
                model="claude-sonnet-4-20250514",
                temperature=0.5,
                max_tokens=2048,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 2048

    def test_api_error_raises_client_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        import anthropic

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = anthropic.APIError(
                message="rate limited",
                request=MagicMock(),
                body=None,
            )

            with pytest.raises(ClientError, match="Anthropic API error"):
                send_message(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_cost_calculation(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("x", input_tokens=1000, output_tokens=500)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = send_message(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
            )

        # Sonnet: 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        assert abs(result.cost_usd - 0.0105) < 1e-9

    def test_multiple_content_blocks(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        blocks = [
            SimpleNamespace(type="text", text="Hello "),
            SimpleNamespace(type="tool_use", id="t1"),  # Non-text block
            SimpleNamespace(type="text", text="world!"),
        ]
        usage = SimpleNamespace(input_tokens=10, output_tokens=5)
        mock_resp = SimpleNamespace(content=blocks, usage=usage)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result = send_message(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result.response == "Hello world!"

    def test_system_as_list_for_cache_control(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()
        system_blocks = [
            {"type": "text", "text": "Document content", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Answer questions."},
        ]

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            send_message(
                model="claude-sonnet-4-20250514",
                system=system_blocks,
                messages=[{"role": "user", "content": "What is this?"}],
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == system_blocks

    def test_extra_kwargs_passed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            send_message(
                model="claude-sonnet-4-20250514",
                top_p=0.9,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["top_p"] == 0.9


class TestSendMessageWithTools:
    def test_basic_with_tool_call(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        blocks = [
            SimpleNamespace(type="text", text="Let me validate that."),
            SimpleNamespace(
                type="tool_use",
                id="tu_123",
                name="validate_yaml",
                input={"yaml_content": "name: test"},
            ),
        ]
        usage = SimpleNamespace(input_tokens=20, output_tokens=15)
        mock_resp = SimpleNamespace(content=blocks, usage=usage)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result, tool_calls = send_message_with_tools(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Check this"}],
                tools=[{"name": "validate_yaml", "description": "Validate YAML"}],
            )

        assert result.response == "Let me validate that."
        assert len(tool_calls) == 1
        assert tool_calls[0]["name"] == "validate_yaml"
        assert tool_calls[0]["id"] == "tu_123"
        assert tool_calls[0]["input"] == {"yaml_content": "name: test"}

    def test_no_tool_calls(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response("Just text")

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            result, tool_calls = send_message_with_tools(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result.response == "Just text"
        assert tool_calls == []

    def test_tools_passed_to_api(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        tools = [{"name": "validate_yaml", "description": "Check"}]

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            send_message_with_tools(
                model="claude-sonnet-4-20250514",
                tools=tools,
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["tools"] == tools

    def test_api_error_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        import anthropic

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.side_effect = anthropic.APIError(
                message="error",
                request=MagicMock(),
                body=None,
            )

            with pytest.raises(ClientError, match="Anthropic API error"):
                send_message_with_tools(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_system_prompt_passed(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        mock_resp = _mock_response()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_client.messages.create.return_value = mock_resp

            send_message_with_tools(
                model="claude-sonnet-4-20250514",
                system="Be a YAML expert.",
            )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == "Be a YAML expert."


class TestSendMessageStreaming:
    def _make_stream_events(
        self, text: str = "Hi!", input_tokens: int = 10, output_tokens: int = 5
    ):
        """Create mock streaming events."""
        events = [
            SimpleNamespace(
                type="message_start",
                message=SimpleNamespace(usage=SimpleNamespace(input_tokens=input_tokens)),
            ),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text=text),
            ),
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=output_tokens),
            ),
        ]
        return events

    def test_basic_streaming(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        events = self._make_stream_events("Hello stream!", 20, 10)

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.__iter__ = MagicMock(return_value=iter(events))
            mock_client.messages.stream.return_value = mock_stream

            result = send_message_streaming(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result.response == "Hello stream!"
        assert result.input_tokens == 20
        assert result.output_tokens == 10

    def test_on_token_callback(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        events = self._make_stream_events("token!", 5, 3)
        tokens_received: list[str] = []

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.__iter__ = MagicMock(return_value=iter(events))
            mock_client.messages.stream.return_value = mock_stream

            send_message_streaming(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
                on_token=tokens_received.append,
            )

        assert "token!" in tokens_received

    def test_api_error_raises_client_error(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        import anthropic

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(
                side_effect=anthropic.APIError(
                    message="stream error", request=MagicMock(), body=None
                )
            )
            mock_client.messages.stream.return_value = mock_stream

            with pytest.raises(ClientError, match="streaming error"):
                send_message_streaming(
                    model="claude-sonnet-4-20250514",
                    messages=[{"role": "user", "content": "Hi"}],
                )

    def test_no_system_prompt(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        events = self._make_stream_events()

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.__iter__ = MagicMock(return_value=iter(events))
            mock_client.messages.stream.return_value = mock_stream

            send_message_streaming(model="claude-sonnet-4-20250514")

        call_kwargs = mock_client.messages.stream.call_args[1]
        assert "system" not in call_kwargs

    def test_event_without_expected_attrs(self, monkeypatch: pytest.MonkeyPatch):
        """Events with unexpected shapes should be silently skipped."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        events = [
            SimpleNamespace(type="unknown_event"),
            SimpleNamespace(
                type="content_block_delta",
                delta=SimpleNamespace(text="ok"),
            ),
            SimpleNamespace(
                type="message_delta",
                usage=SimpleNamespace(output_tokens=2),
            ),
        ]

        with patch("promptctl.client.anthropic.Anthropic") as mock_cls:
            mock_client = mock_cls.return_value
            mock_stream = MagicMock()
            mock_stream.__enter__ = MagicMock(return_value=mock_stream)
            mock_stream.__exit__ = MagicMock(return_value=False)
            mock_stream.__iter__ = MagicMock(return_value=iter(events))
            mock_client.messages.stream.return_value = mock_stream

            result = send_message_streaming(
                model="claude-sonnet-4-20250514",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result.response == "ok"
