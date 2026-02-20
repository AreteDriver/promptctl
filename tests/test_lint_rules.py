"""Tests for promptctl.lint.rules."""

from __future__ import annotations

from promptctl.lint.rules import RULES, get_all_rules, get_rule
from promptctl.models import LintCategory, LintSeverity


class TestGetAllRules:
    def test_returns_all_eight(self):
        rules = get_all_rules()
        assert len(rules) == 8

    def test_ids_sequential(self):
        rules = get_all_rules()
        ids = [r.id for r in rules]
        assert ids == [f"L{i:03d}" for i in range(1, 9)]

    def test_returns_copy(self):
        rules = get_all_rules()
        rules.pop()
        assert len(get_all_rules()) == 8


class TestGetRule:
    def test_existing_rule(self):
        rule = get_rule("L001")
        assert rule is not None
        assert rule.name == "invalid-yaml-syntax"
        assert rule.severity == LintSeverity.ERROR
        assert rule.category == LintCategory.SYNTAX

    def test_nonexistent_rule(self):
        assert get_rule("L999") is None

    def test_all_rules_retrievable(self):
        for rule in RULES:
            found = get_rule(rule.id)
            assert found is not None
            assert found.id == rule.id


class TestRuleDefinitions:
    def test_security_rules_are_errors(self):
        for rule in RULES:
            if rule.category == LintCategory.SECURITY:
                assert rule.severity == LintSeverity.ERROR

    def test_all_have_descriptions(self):
        for rule in RULES:
            assert rule.description
            assert rule.name
