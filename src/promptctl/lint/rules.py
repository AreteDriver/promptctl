"""Built-in lint rules for prompt template YAML files."""

from __future__ import annotations

from promptctl.models import LintCategory, LintRule, LintSeverity

RULES: list[LintRule] = [
    LintRule(
        id="L001",
        name="invalid-yaml-syntax",
        description="File is not valid YAML.",
        severity=LintSeverity.ERROR,
        category=LintCategory.SYNTAX,
    ),
    LintRule(
        id="L002",
        name="missing-name-field",
        description="Prompt template should have a 'name' field.",
        severity=LintSeverity.ERROR,
        category=LintCategory.STRUCTURE,
    ),
    LintRule(
        id="L003",
        name="missing-version-field",
        description="Prompt template should have a 'version' field.",
        severity=LintSeverity.WARNING,
        category=LintCategory.STRUCTURE,
    ),
    LintRule(
        id="L004",
        name="duplicate-keys",
        description="YAML contains duplicate mapping keys.",
        severity=LintSeverity.ERROR,
        category=LintCategory.SYNTAX,
    ),
    LintRule(
        id="L005",
        name="unused-variables",
        description="Template variables defined but not used in prompt text.",
        severity=LintSeverity.WARNING,
        category=LintCategory.STYLE,
    ),
    LintRule(
        id="L006",
        name="hardcoded-secrets",
        description="Potential secrets or API keys found in template.",
        severity=LintSeverity.ERROR,
        category=LintCategory.SECURITY,
    ),
    LintRule(
        id="L007",
        name="naming-convention",
        description="Template name should be lowercase with hyphens or underscores.",
        severity=LintSeverity.INFO,
        category=LintCategory.STYLE,
    ),
    LintRule(
        id="L008",
        name="deep-nesting",
        description="YAML nesting exceeds recommended depth.",
        severity=LintSeverity.WARNING,
        category=LintCategory.COMPLEXITY,
    ),
]

_RULES_BY_ID: dict[str, LintRule] = {r.id: r for r in RULES}


def get_all_rules() -> list[LintRule]:
    """Return all built-in lint rules."""
    return list(RULES)


def get_rule(rule_id: str) -> LintRule | None:
    """Get a rule by its ID, or None if not found."""
    return _RULES_BY_ID.get(rule_id)
