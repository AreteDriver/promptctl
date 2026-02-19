"""Tests for promptctl.models."""

from __future__ import annotations

from promptctl.models import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    MODEL_COSTS,
    ClaudeModel,
    ComparisonEntry,
    ComparisonResult,
    PromptResult,
    PromptTemplate,
    ReviewCategory,
    ReviewFinding,
    ReviewReport,
    Severity,
    calculate_cost,
)


class TestClaudeModel:
    def test_sonnet_value(self):
        assert "sonnet" in ClaudeModel.SONNET

    def test_opus_value(self):
        assert "opus" in ClaudeModel.OPUS

    def test_haiku_value(self):
        assert "haiku" in ClaudeModel.HAIKU

    def test_default_model_is_sonnet(self):
        assert DEFAULT_MODEL == ClaudeModel.SONNET

    def test_all_models_in_costs(self):
        for model in ClaudeModel:
            assert model in MODEL_COSTS


class TestSeverity:
    def test_values(self):
        assert Severity.ERROR == "error"
        assert Severity.WARNING == "warning"
        assert Severity.INFO == "info"


class TestReviewCategory:
    def test_all_six_dimensions(self):
        expected = {"correctness", "security", "performance", "maintainability", "testing", "style"}
        actual = {c.value for c in ReviewCategory}
        assert actual == expected


class TestPromptTemplate:
    def test_minimal(self):
        t = PromptTemplate(name="test", user="Hello")
        assert t.name == "test"
        assert t.user == "Hello"
        assert t.system == ""
        assert t.variables == {}
        assert t.model == DEFAULT_MODEL
        assert t.temperature == DEFAULT_TEMPERATURE
        assert t.max_tokens == DEFAULT_MAX_TOKENS

    def test_full(self):
        t = PromptTemplate(
            name="full",
            system="You are helpful.",
            user="What is {topic}?",
            variables={"topic": "Python"},
            model=ClaudeModel.HAIKU,
            temperature=0.5,
            max_tokens=2048,
        )
        assert t.system == "You are helpful."
        assert t.variables["topic"] == "Python"
        assert t.model == ClaudeModel.HAIKU


class TestPromptResult:
    def test_defaults(self):
        r = PromptResult(model="test", response="hi")
        assert r.input_tokens == 0
        assert r.output_tokens == 0
        assert r.cost_usd == 0.0
        assert r.latency_ms == 0.0

    def test_json_roundtrip(self):
        r = PromptResult(
            model="claude-sonnet-4-20250514",
            response="Hello!",
            input_tokens=10,
            output_tokens=5,
            cost_usd=0.001,
            latency_ms=150.5,
        )
        data = r.model_dump()
        r2 = PromptResult(**data)
        assert r2 == r


class TestReviewFinding:
    def test_minimal(self):
        f = ReviewFinding(
            severity=Severity.WARNING,
            category=ReviewCategory.SECURITY,
            message="SQL injection risk",
        )
        assert f.file == ""
        assert f.line == 0
        assert f.suggestion == ""

    def test_full(self):
        f = ReviewFinding(
            severity=Severity.ERROR,
            category=ReviewCategory.CORRECTNESS,
            file="main.py",
            line=42,
            message="Off-by-one error",
            suggestion="Use range(n) instead of range(n+1)",
        )
        assert f.file == "main.py"
        assert f.line == 42


class TestReviewReport:
    def test_empty_report(self):
        r = ReviewReport()
        assert r.error_count == 0
        assert r.warning_count == 0
        assert r.info_count == 0
        assert r.findings == []

    def test_counts(self):
        r = ReviewReport(
            findings=[
                ReviewFinding(
                    severity=Severity.ERROR,
                    category=ReviewCategory.SECURITY,
                    message="a",
                ),
                ReviewFinding(
                    severity=Severity.ERROR,
                    category=ReviewCategory.CORRECTNESS,
                    message="b",
                ),
                ReviewFinding(
                    severity=Severity.WARNING,
                    category=ReviewCategory.STYLE,
                    message="c",
                ),
                ReviewFinding(
                    severity=Severity.INFO,
                    category=ReviewCategory.TESTING,
                    message="d",
                ),
            ]
        )
        assert r.error_count == 2
        assert r.warning_count == 1
        assert r.info_count == 1


class TestComparisonResult:
    def test_empty(self):
        c = ComparisonResult()
        assert c.entries == []

    def test_with_entries(self):
        c = ComparisonResult(
            prompt_name="test",
            entries=[
                ComparisonEntry(model="sonnet", response_preview="Hi", cost_usd=0.01),
                ComparisonEntry(model="haiku", response_preview="Hey", cost_usd=0.001),
            ],
        )
        assert len(c.entries) == 2


class TestCalculateCost:
    def test_sonnet_cost(self):
        cost = calculate_cost(ClaudeModel.SONNET, input_tokens=1000, output_tokens=500)
        # 1000 * 3.0/1M + 500 * 15.0/1M = 0.003 + 0.0075 = 0.0105
        assert abs(cost - 0.0105) < 1e-9

    def test_haiku_cost(self):
        cost = calculate_cost(ClaudeModel.HAIKU, input_tokens=1000, output_tokens=500)
        # 1000 * 0.80/1M + 500 * 4.0/1M = 0.0008 + 0.002 = 0.0028
        assert abs(cost - 0.0028) < 1e-9

    def test_unknown_model_returns_zero(self):
        assert calculate_cost("unknown-model", 1000, 500) == 0.0

    def test_zero_tokens(self):
        assert calculate_cost(ClaudeModel.SONNET, 0, 0) == 0.0
