"""Anthropic SDK wrapper for promptctl."""

from __future__ import annotations

import time
from typing import Any

import anthropic

from promptctl.config import get_api_key
from promptctl.exceptions import ClientError
from promptctl.models import PromptResult, calculate_cost

# Default model constants
MODEL_SONNET = "claude-sonnet-4-20250514"
MODEL_HAIKU = "claude-haiku-4-5-20251001"
MODEL_OPUS = "claude-opus-4-20250514"


def get_client() -> anthropic.Anthropic:
    """Create an Anthropic client from configured API key."""
    api_key = get_api_key()
    if not api_key:
        raise ClientError(
            "No API key configured. Set ANTHROPIC_API_KEY environment variable "
            "or run 'promptctl config set api_key <key>'."
        )
    return anthropic.Anthropic(api_key=api_key)


def send_message(
    model: str,
    system: str | list[dict[str, Any]] = "",
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    **kwargs: Any,
) -> PromptResult:
    """Send a message to Claude and return structured result."""
    client = get_client()
    if messages is None:
        messages = []

    api_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        api_kwargs["system"] = system
    api_kwargs.update(kwargs)

    start = time.monotonic()
    try:
        response = client.messages.create(**api_kwargs)
    except anthropic.APIError as e:
        raise ClientError(f"Anthropic API error: {e}") from e

    elapsed_ms = (time.monotonic() - start) * 1000

    text = ""
    for block in response.content:
        if block.type == "text":
            text += block.text

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)

    return PromptResult(
        model=model,
        response=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=elapsed_ms,
    )


def send_message_with_tools(
    model: str,
    system: str = "",
    messages: list[dict[str, Any]] | None = None,
    tools: list[dict[str, Any]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    **kwargs: Any,
) -> tuple[PromptResult, list[dict[str, Any]]]:
    """Send a message with tools and return (result, tool_calls).

    Returns the text response as PromptResult plus a list of tool_use blocks
    extracted from the response.
    """
    client = get_client()
    if messages is None:
        messages = []

    api_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        api_kwargs["system"] = system
    if tools:
        api_kwargs["tools"] = tools
    api_kwargs.update(kwargs)

    start = time.monotonic()
    try:
        response = client.messages.create(**api_kwargs)
    except anthropic.APIError as e:
        raise ClientError(f"Anthropic API error: {e}") from e

    elapsed_ms = (time.monotonic() - start) * 1000

    text = ""
    tool_calls: list[dict[str, Any]] = []
    for block in response.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "input": block.input})

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = calculate_cost(model, input_tokens, output_tokens)

    result = PromptResult(
        model=model,
        response=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=elapsed_ms,
    )
    return result, tool_calls


def send_message_streaming(
    model: str,
    system: str = "",
    messages: list[dict[str, str]] | None = None,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    on_token: Any = None,
    **kwargs: Any,
) -> PromptResult:
    """Send a message with streaming, calling on_token for each text delta."""
    client = get_client()
    if messages is None:
        messages = []

    api_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system:
        api_kwargs["system"] = system
    api_kwargs.update(kwargs)

    start = time.monotonic()
    full_text = ""
    input_tokens = 0
    output_tokens = 0

    try:
        with client.messages.stream(**api_kwargs) as stream:
            for event in stream:
                if hasattr(event, "type"):
                    if event.type == "content_block_delta" and hasattr(event, "delta"):
                        if hasattr(event.delta, "text"):
                            full_text += event.delta.text
                            if on_token:
                                on_token(event.delta.text)
                    elif event.type == "message_delta" and hasattr(event, "usage"):
                        output_tokens = event.usage.output_tokens
                    elif event.type == "message_start" and hasattr(event, "message"):
                        if hasattr(event.message, "usage"):
                            input_tokens = event.message.usage.input_tokens
    except anthropic.APIError as e:
        raise ClientError(f"Anthropic API streaming error: {e}") from e

    elapsed_ms = (time.monotonic() - start) * 1000
    cost = calculate_cost(model, input_tokens, output_tokens)

    return PromptResult(
        model=model,
        response=full_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        latency_ms=elapsed_ms,
    )
