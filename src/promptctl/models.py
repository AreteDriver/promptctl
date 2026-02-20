"""Pydantic v2 models and enums for promptctl."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from enum import Enum

    class StrEnum(str, Enum):
        """Backport for Python 3.10."""


from pydantic import BaseModel, Field


class ClaudeModel(StrEnum):
    """Supported Claude model identifiers."""

    OPUS = "claude-opus-4-20250514"
    SONNET = "claude-sonnet-4-20250514"
    HAIKU = "claude-haiku-4-5-20251001"


# Cost per million tokens (USD)
MODEL_COSTS: dict[str, tuple[float, float]] = {
    ClaudeModel.OPUS: (15.0, 75.0),
    ClaudeModel.SONNET: (3.0, 15.0),
    ClaudeModel.HAIKU: (0.80, 4.0),
}

DEFAULT_MODEL = ClaudeModel.SONNET
DEFAULT_TEMPERATURE = 1.0
DEFAULT_MAX_TOKENS = 4096


class Severity(StrEnum):
    """Review finding severity."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ReviewCategory(StrEnum):
    """Code review dimension."""

    CORRECTNESS = "correctness"
    SECURITY = "security"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    TESTING = "testing"
    STYLE = "style"


class PromptTemplate(BaseModel):
    """A prompt template loaded from YAML."""

    name: str
    system: str = ""
    user: str
    variables: dict[str, str] = Field(default_factory=dict)
    model: str = DEFAULT_MODEL
    temperature: float = DEFAULT_TEMPERATURE
    max_tokens: int = DEFAULT_MAX_TOKENS


class PromptResult(BaseModel):
    """Result from executing a prompt."""

    model: str
    response: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ReviewFinding(BaseModel):
    """A single finding from code review."""

    severity: Severity
    category: ReviewCategory
    file: str = ""
    line: int = 0
    message: str
    suggestion: str = ""


class ReviewReport(BaseModel):
    """Complete code review report."""

    findings: list[ReviewFinding] = Field(default_factory=list)
    summary: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def error_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == Severity.INFO)


class ComparisonEntry(BaseModel):
    """Result for one model in a comparison."""

    model: str
    response_preview: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ComparisonResult(BaseModel):
    """Result of comparing responses across models."""

    prompt_name: str = ""
    entries: list[ComparisonEntry] = Field(default_factory=list)


# --- Document Intelligence Models ---


class DocAnalysis(BaseModel):
    """Result from document analysis."""

    summary: str = ""
    key_points: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    word_count: int = 0
    token_estimate: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class DocAnswer(BaseModel):
    """Answer to a question about a document."""

    answer: str = ""
    confidence: str = ""
    source_quotes: list[str] = Field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class DocSummary(BaseModel):
    """Executive summary of a document."""

    executive_summary: str = ""
    sections: list[str] = Field(default_factory=list)
    word_count: int = 0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    chunks_processed: int = 0


# --- Lint Models ---


class LintSeverity(StrEnum):
    """Lint rule severity levels."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class LintCategory(StrEnum):
    """Lint rule categories."""

    STRUCTURE = "structure"
    SYNTAX = "syntax"
    SECURITY = "security"
    STYLE = "style"
    COMPLEXITY = "complexity"


class LintRule(BaseModel):
    """A single lint rule definition."""

    id: str
    name: str
    description: str
    severity: LintSeverity = LintSeverity.WARNING
    category: LintCategory = LintCategory.STRUCTURE


class LintViolation(BaseModel):
    """A single lint violation found in a file."""

    rule_id: str
    file: str = ""
    line: int = 0
    column: int = 0
    message: str
    suggestion: str = ""
    severity: LintSeverity = LintSeverity.WARNING


class LintReport(BaseModel):
    """Complete lint report for a file."""

    violations: list[LintViolation] = Field(default_factory=list)
    summary: str = ""
    file_path: str = ""

    @property
    def error_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == LintSeverity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == LintSeverity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for v in self.violations if v.severity == LintSeverity.INFO)


class LintFix(BaseModel):
    """AI-suggested fix for lint violations."""

    original: str = ""
    fixed: str = ""
    explanation: str = ""
    violations_addressed: list[str] = Field(default_factory=list)
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate USD cost for a request."""
    costs = MODEL_COSTS.get(model)
    if costs is None:
        return 0.0
    input_rate, output_rate = costs
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
