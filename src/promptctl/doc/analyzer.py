"""Document analysis via Claude API — analyze, ask, summarize."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from promptctl.client import send_message
from promptctl.config import get_config
from promptctl.exceptions import DocError
from promptctl.models import DocAnalysis, DocAnswer, DocSummary

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv"}

# Token estimation: ~4 chars per token
CHARS_PER_TOKEN = 4
MAX_CONTEXT_TOKENS = 180_000

ANALYZE_SYSTEM_PROMPT = """\
You are a document analysis expert. Analyze the provided document and return \
a JSON object with your findings.

Return ONLY valid JSON in this exact format:
{
  "summary": "Brief overview of the document",
  "key_points": ["Point 1", "Point 2"],
  "entities": ["Entity 1", "Entity 2"],
  "themes": ["Theme 1", "Theme 2"]
}
"""

ASK_SYSTEM_PROMPT_SUFFIX = """\
Answer questions about the document above. Return ONLY valid JSON:
{
  "answer": "Your answer here",
  "confidence": "high|medium|low",
  "source_quotes": ["Relevant quote from the document"]
}
"""

SUMMARIZE_SYSTEM_PROMPT = """\
You are an expert summarizer. Create an executive summary of the provided \
document. Return ONLY valid JSON:
{
  "executive_summary": "Concise executive summary",
  "sections": ["Section 1 summary", "Section 2 summary"]
}
"""


def read_document(path: str) -> tuple[str, int]:
    """Read a document file and return (content, word_count).

    Raises DocError for unsupported types or read failures.
    """
    p = Path(path)
    if not p.exists():
        raise DocError(f"File not found: {path}")
    if not p.is_file():
        raise DocError(f"Not a file: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise DocError(
            f"Unsupported file type: {p.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise DocError(f"Cannot read binary file: {path}") from None
    word_count = len(content.split())
    return content, word_count


def estimate_tokens(text: str) -> int:
    """Estimate token count from text length."""
    return len(text) // CHARS_PER_TOKEN


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from a response, handling code blocks."""
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


def analyze_document(path: str, model: str | None = None) -> DocAnalysis:
    """Analyze a document: extract key points, entities, themes."""
    content, word_count = read_document(path)
    config = get_config()
    resolved_model = model or str(config.get("model", ""))

    result = send_message(
        model=resolved_model,
        system=ANALYZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=0.0,
    )

    data = _parse_json_response(result.response)
    return DocAnalysis(
        summary=data.get("summary", result.response[:500]),
        key_points=data.get("key_points", []),
        entities=data.get("entities", []),
        themes=data.get("themes", []),
        word_count=word_count,
        token_estimate=estimate_tokens(content),
        model=resolved_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )


def ask_document(path: str, question: str, model: str | None = None) -> DocAnswer:
    """Ask a question about a document. Uses cache_control on system prompt."""
    content, _ = read_document(path)
    config = get_config()
    resolved_model = model or str(config.get("model", ""))

    # Use cache_control for the document content (prompt caching)
    system: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"Document content:\n\n{content}",
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": ASK_SYSTEM_PROMPT_SUFFIX,
        },
    ]

    result = send_message(
        model=resolved_model,
        system=system,
        messages=[{"role": "user", "content": question}],
        max_tokens=4096,
        temperature=0.0,
    )

    data = _parse_json_response(result.response)
    return DocAnswer(
        answer=data.get("answer", result.response[:500]),
        confidence=data.get("confidence", ""),
        source_quotes=data.get("source_quotes", []),
        model=resolved_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
    )


def summarize_document(path: str, model: str | None = None) -> DocSummary:
    """Generate executive summary. Uses map-reduce for large docs."""
    content, word_count = read_document(path)
    config = get_config()
    resolved_model = model or str(config.get("model", ""))

    token_estimate = estimate_tokens(content)
    if token_estimate > MAX_CONTEXT_TOKENS:
        from promptctl.doc.chunker import map_reduce_summarize

        return map_reduce_summarize(content, resolved_model, word_count)

    result = send_message(
        model=resolved_model,
        system=SUMMARIZE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
        temperature=0.0,
    )

    data = _parse_json_response(result.response)
    return DocSummary(
        executive_summary=data.get("executive_summary", result.response[:500]),
        sections=data.get("sections", []),
        word_count=word_count,
        model=resolved_model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cost_usd=result.cost_usd,
        chunks_processed=1,
    )
