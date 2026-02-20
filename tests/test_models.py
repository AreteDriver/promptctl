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
    DocAnalysis,
    DocAnswer,
    DocSummary,
    LintCategory,
    LintFix,
    LintReport,
    LintRule,
    LintSeverity,
    LintViolation,
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


class TestDocAnalysis:
    def test_defaults(self):
        d = DocAnalysis()
        assert d.summary == ""
        assert d.key_points == []
        assert d.entities == []
        assert d.themes == []
        assert d.word_count == 0
        assert d.token_estimate == 0

    def test_full(self):
        d = DocAnalysis(
            summary="A report about AI",
            key_points=["Point 1", "Point 2"],
            entities=["OpenAI", "Anthropic"],
            themes=["safety", "alignment"],
            word_count=500,
            token_estimate=125,
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        assert len(d.key_points) == 2
        assert d.model == "claude-sonnet-4-20250514"

    def test_json_roundtrip(self):
        d = DocAnalysis(summary="test", key_points=["a"])
        data = d.model_dump()
        d2 = DocAnalysis(**data)
        assert d2 == d


class TestDocAnswer:
    def test_defaults(self):
        a = DocAnswer()
        assert a.answer == ""
        assert a.confidence == ""
        assert a.source_quotes == []

    def test_full(self):
        a = DocAnswer(
            answer="42",
            confidence="high",
            source_quotes=["The answer is 42."],
            model="claude-sonnet-4-20250514",
        )
        assert a.confidence == "high"
        assert len(a.source_quotes) == 1


class TestDocSummary:
    def test_defaults(self):
        s = DocSummary()
        assert s.executive_summary == ""
        assert s.sections == []
        assert s.chunks_processed == 0

    def test_chunks_processed(self):
        s = DocSummary(executive_summary="Summary", chunks_processed=3)
        assert s.chunks_processed == 3


class TestLintSeverity:
    def test_values(self):
        assert LintSeverity.ERROR == "error"
        assert LintSeverity.WARNING == "warning"
        assert LintSeverity.INFO == "info"


class TestLintCategory:
    def test_values(self):
        expected = {"structure", "syntax", "security", "style", "complexity"}
        actual = {c.value for c in LintCategory}
        assert actual == expected


class TestLintRule:
    def test_construction(self):
        r = LintRule(id="L001", name="test-rule", description="A test rule")
        assert r.id == "L001"
        assert r.severity == LintSeverity.WARNING
        assert r.category == LintCategory.STRUCTURE


class TestLintViolation:
    def test_defaults(self):
        v = LintViolation(rule_id="L001", message="Bad thing")
        assert v.file == ""
        assert v.line == 0
        assert v.column == 0
        assert v.suggestion == ""
        assert v.severity == LintSeverity.WARNING

    def test_full(self):
        v = LintViolation(
            rule_id="L006",
            file="config.yaml",
            line=10,
            column=5,
            message="Hardcoded secret",
            suggestion="Use env var",
            severity=LintSeverity.ERROR,
        )
        assert v.file == "config.yaml"
        assert v.severity == LintSeverity.ERROR


class TestLintReport:
    def test_empty(self):
        r = LintReport()
        assert r.violations == []
        assert r.error_count == 0
        assert r.warning_count == 0
        assert r.info_count == 0

    def test_counts(self):
        r = LintReport(
            violations=[
                LintViolation(rule_id="L001", message="a", severity=LintSeverity.ERROR),
                LintViolation(rule_id="L002", message="b", severity=LintSeverity.ERROR),
                LintViolation(rule_id="L007", message="c", severity=LintSeverity.WARNING),
                LintViolation(rule_id="L008", message="d", severity=LintSeverity.INFO),
            ]
        )
        assert r.error_count == 2
        assert r.warning_count == 1
        assert r.info_count == 1


class TestLintFix:
    def test_defaults(self):
        f = LintFix()
        assert f.original == ""
        assert f.fixed == ""
        assert f.violations_addressed == []

    def test_full(self):
        f = LintFix(
            original="password: secret123",
            fixed="password: ${SECRET}",
            explanation="Use environment variable",
            violations_addressed=["L006"],
            model="claude-sonnet-4-20250514",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.001,
        )
        assert len(f.violations_addressed) == 1
        assert f.model == "claude-sonnet-4-20250514"


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
