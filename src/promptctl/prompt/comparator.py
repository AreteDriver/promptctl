"""Multi-model prompt comparison (Pro feature)."""

from __future__ import annotations

from promptctl.client import send_message
from promptctl.exceptions import PromptError
from promptctl.gates import require_pro
from promptctl.models import (
    ClaudeModel,
    ComparisonEntry,
    ComparisonResult,
)
from promptctl.prompt.runner import interpolate, load_template


@require_pro("multi-model comparison")
def compare_models(
    template_path: str,
    models: str | None = None,
) -> ComparisonResult:
    """Run the same prompt against multiple models and compare results."""
    template = load_template(template_path)

    if models:
        model_list = [m.strip() for m in models.split(",")]
    else:
        model_list = [ClaudeModel.SONNET, ClaudeModel.HAIKU]

    if len(model_list) < 2:
        raise PromptError("Comparison requires at least 2 models")

    user_text = interpolate(template)
    messages = [{"role": "user", "content": user_text}]

    entries: list[ComparisonEntry] = []
    for model in model_list:
        result = send_message(
            model=model,
            system=template.system,
            messages=messages,
            max_tokens=template.max_tokens,
            temperature=template.temperature,
        )
        preview = result.response[:200] if result.response else ""
        entries.append(
            ComparisonEntry(
                model=model,
                response_preview=preview,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=result.cost_usd,
                latency_ms=result.latency_ms,
            )
        )

    return ComparisonResult(prompt_name=template.name, entries=entries)
