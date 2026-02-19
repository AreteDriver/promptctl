"""Prompt template runner — load, interpolate, execute."""

from __future__ import annotations

from pathlib import Path

import yaml

from promptctl.client import send_message, send_message_streaming
from promptctl.config import get_config
from promptctl.exceptions import PromptError
from promptctl.models import PromptResult, PromptTemplate


def load_template(path: str) -> PromptTemplate:
    """Load a prompt template from a YAML file."""
    p = Path(path)
    if not p.exists():
        raise PromptError(f"Template file not found: {path}")
    try:
        with open(p) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise PromptError(f"Invalid YAML in template: {e}") from e

    if not isinstance(data, dict):
        raise PromptError(f"Template must be a YAML mapping, got {type(data).__name__}")

    if "name" not in data:
        raise PromptError("Template missing required 'name' field")
    if "user" not in data:
        raise PromptError("Template missing required 'user' field")

    return PromptTemplate(**data)


def interpolate(template: PromptTemplate) -> str:
    """Interpolate variables into the user prompt."""
    text = template.user
    for key, value in template.variables.items():
        text = text.replace(f"{{{key}}}", value)
    return text


def run_prompt(
    template_path: str,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    stream: bool = False,
) -> PromptResult:
    """Load a template, interpolate, and run against Claude."""
    template = load_template(template_path)
    config = get_config()

    resolved_model = model or template.model or str(config.get("model", ""))
    resolved_temp = temperature if temperature is not None else template.temperature
    resolved_max = max_tokens if max_tokens is not None else template.max_tokens

    user_text = interpolate(template)
    messages = [{"role": "user", "content": user_text}]

    if stream:
        return send_message_streaming(
            model=resolved_model,
            system=template.system,
            messages=messages,
            max_tokens=resolved_max,
            temperature=resolved_temp,
        )
    return send_message(
        model=resolved_model,
        system=template.system,
        messages=messages,
        max_tokens=resolved_max,
        temperature=resolved_temp,
    )
