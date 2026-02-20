"""AI-powered lint fix suggestions using Claude tool use (Pro)."""

from __future__ import annotations

import json
import logging
from typing import Any

from promptctl.client import send_message
from promptctl.config import get_config
from promptctl.gates import require_pro
from promptctl.lint.checker import check_file
from promptctl.models import LintFix

logger = logging.getLogger(__name__)

FIX_SYSTEM_PROMPT = """\
You are a YAML prompt template expert. Given a YAML file and its lint violations, \
produce a corrected version of the file. Return ONLY valid JSON:
{
  "fixed": "The corrected YAML content",
  "explanation": "Brief explanation of changes made",
  "violations_addressed": ["L001", "L002"]
}
"""


@require_pro("lint fix")
def fix_violations(path: str, model: str | None = None) -> LintFix:
    """Suggest fixes for lint violations using Claude (Pro feature).

    Reads the file, runs local lint checks, then sends both the file
    content and violations to Claude for AI-suggested fixes.
    """
    from pathlib import Path

    p = Path(path)
    original = p.read_text(encoding="utf-8")
    report = check_file(path)

    if not report.violations:
        return LintFix(
            original=original,
            fixed=original,
            explanation="No violations to fix.",
        )

    config = get_config()
    resolved_model = model or str(config.get("model", ""))

    violations_text = "\n".join(f"- [{v.rule_id}] {v.message}" for v in report.violations)
    user_content = (
        f"YAML file content:\n```yaml\n{original}\n```\n\nLint violations found:\n{violations_text}"
    )

    result = send_message(
        model=resolved_model,
        system=FIX_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=4096,
        temperature=0.0,
    )

    data = _parse_fix_response(result.response)

    return LintFix(
        original=original,
        fixed=data.get("fixed", original),
        explanation=data.get("explanation", result.response[:500]),
        violations_addressed=data.get("violations_addressed", []),
        model=resolved_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )


def _parse_fix_response(text: str) -> dict[str, Any]:
    """Extract JSON from the fix response."""
    stripped = text.strip()
    if "```json" in stripped:
        start = stripped.index("```json") + 7
        end = stripped.index("```", start)
        stripped = stripped[start:end].strip()
    elif "```" in stripped:
        start = stripped.index("```") + 3
        end = stripped.index("```", start)
        stripped = stripped[start:end].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {}
