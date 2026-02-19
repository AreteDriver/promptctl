"""Structured code review via Claude API."""

from __future__ import annotations

import json
import logging

from promptctl.client import send_message
from promptctl.config import get_config
from promptctl.models import (
    ReviewFinding,
    ReviewReport,
    calculate_cost,
)

logger = logging.getLogger(__name__)

REVIEW_SYSTEM_PROMPT = """\
You are an expert code reviewer. Analyze the provided code or diff and return \
a JSON object with your findings.

Review across these 6 dimensions:
1. **Correctness** — Does it work? Edge cases? Race conditions?
2. **Security** — Input validation, injection risks, credential handling
3. **Performance** — Time/space complexity, unnecessary operations
4. **Maintainability** — Readability, naming, modularity, DRY
5. **Testing** — Coverage gaps, missing edge cases
6. **Style** — Formatting, conventions, consistency

Return ONLY valid JSON in this exact format:
{
  "summary": "Brief overall assessment",
  "findings": [
    {
      "severity": "error|warning|info",
      "category": "correctness|security|performance|maintainability|testing|style",
      "file": "filename or empty string",
      "line": 0,
      "message": "What the issue is",
      "suggestion": "How to fix it"
    }
  ]
}

If the code looks good, return {"summary": "No issues found.", "findings": []}.
"""


def review_code(
    code: str,
    model: str | None = None,
    source_file: str = "",
) -> ReviewReport:
    """Review code using Claude and return structured report."""
    config = get_config()
    resolved_model = model or str(config.get("model", ""))

    user_content = code
    if source_file:
        user_content = f"File: {source_file}\n\n{code}"

    result = send_message(
        model=resolved_model,
        system=REVIEW_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        max_tokens=4096,
        temperature=0.0,
    )

    return _parse_review_response(
        result.response,
        model=resolved_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


def _parse_review_response(
    response: str,
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> ReviewReport:
    """Parse Claude's JSON response into a ReviewReport."""
    # Try to extract JSON from the response
    text = response.strip()

    # Handle markdown code blocks
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse review response as JSON")
        return ReviewReport(
            summary=response[:500],
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=calculate_cost(model, input_tokens, output_tokens),
        )

    findings = []
    for item in data.get("findings", []):
        if not isinstance(item, dict):
            logger.debug("Skipping non-dict finding: %s", item)
            continue
        try:
            findings.append(
                ReviewFinding(
                    severity=item.get("severity", "info"),
                    category=item.get("category", "style"),
                    file=item.get("file", ""),
                    line=item.get("line", 0),
                    message=item.get("message", ""),
                    suggestion=item.get("suggestion", ""),
                )
            )
        except (ValueError, KeyError):
            logger.debug("Skipping malformed finding: %s", item)

    return ReviewReport(
        findings=findings,
        summary=data.get("summary", ""),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=calculate_cost(model, input_tokens, output_tokens),
    )
