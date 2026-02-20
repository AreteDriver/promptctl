"""Map-reduce chunking for documents exceeding context limits."""

from __future__ import annotations

import json
import logging

from promptctl.client import send_message
from promptctl.models import DocSummary, calculate_cost

logger = logging.getLogger(__name__)

# Chunk size in characters (~25k tokens at 4 chars/token)
CHUNK_CHAR_SIZE = 400_000
OVERLAP_CHARS = 2000

CHUNK_SUMMARIZE_PROMPT = """\
Summarize this section of a larger document. Focus on key information, \
main arguments, and important details. Return ONLY valid JSON:
{
  "section_summary": "Summary of this section"
}
"""

REDUCE_PROMPT = """\
You are given summaries of different sections of a document. \
Synthesize them into a cohesive executive summary. Return ONLY valid JSON:
{
  "executive_summary": "Cohesive executive summary",
  "sections": ["Section 1 summary", "Section 2 summary"]
}
"""


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks by character count."""
    if chunk_size is None:
        chunk_size = CHUNK_CHAR_SIZE
    if overlap is None:
        overlap = OVERLAP_CHARS
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start + overlap >= len(text):
            break
    return chunks


def map_reduce_summarize(text: str, model: str, word_count: int) -> DocSummary:
    """Summarize a large document via map-reduce.

    1. Split into chunks
    2. Map: summarize each chunk independently
    3. Reduce: synthesize chunk summaries into executive summary
    """
    chunks = chunk_text(text)
    total_input = 0
    total_output = 0

    # Map phase: summarize each chunk
    chunk_summaries = []
    for chunk in chunks:
        result = send_message(
            model=model,
            system=CHUNK_SUMMARIZE_PROMPT,
            messages=[{"role": "user", "content": chunk}],
            max_tokens=2048,
            temperature=0.0,
        )
        total_input += result.input_tokens
        total_output += result.output_tokens

        try:
            data = json.loads(result.response.strip())
            chunk_summaries.append(data.get("section_summary", result.response[:500]))
        except json.JSONDecodeError:
            chunk_summaries.append(result.response[:500])

    # Reduce phase: synthesize summaries
    combined = "\n\n---\n\n".join(f"Section {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries))
    reduce_result = send_message(
        model=model,
        system=REDUCE_PROMPT,
        messages=[{"role": "user", "content": combined}],
        max_tokens=4096,
        temperature=0.0,
    )
    total_input += reduce_result.input_tokens
    total_output += reduce_result.output_tokens

    try:
        data = json.loads(reduce_result.response.strip())
    except json.JSONDecodeError:
        data = {}

    total_cost = calculate_cost(model, total_input, total_output)

    return DocSummary(
        executive_summary=data.get("executive_summary", reduce_result.response[:500]),
        sections=data.get("sections", chunk_summaries),
        word_count=word_count,
        model=model,
        input_tokens=total_input,
        output_tokens=total_output,
        cost_usd=total_cost,
        chunks_processed=len(chunks),
    )
