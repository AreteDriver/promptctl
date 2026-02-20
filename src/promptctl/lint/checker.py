"""Local YAML lint checker — no API calls required."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from promptctl.exceptions import LintError
from promptctl.lint.rules import get_rule
from promptctl.models import LintReport, LintSeverity, LintViolation

# Patterns that suggest hardcoded secrets
SECRET_PATTERNS = [
    re.compile(r"sk-ant-[a-zA-Z0-9]{10,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{20,}", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*[:=]\s*['\"][^'\"]{10,}['\"]", re.IGNORECASE),
    re.compile(r"password\s*[:=]\s*['\"][^'\"]{5,}['\"]", re.IGNORECASE),
    re.compile(r"secret\s*[:=]\s*['\"][^'\"]{5,}['\"]", re.IGNORECASE),
]

MAX_NESTING_DEPTH = 6


class _DuplicateKeyLoader(yaml.SafeLoader):
    """SafeLoader subclass that detects duplicate keys."""

    duplicates: list[str]

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        self.duplicates = getattr(self, "duplicates", [])
        keys: set[str] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in keys:
                self.duplicates.append(str(key))
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _measure_depth(obj: object, current: int = 0) -> int:
    """Measure maximum nesting depth of a data structure."""
    if isinstance(obj, dict):
        if not obj:
            return current + 1
        return max(_measure_depth(v, current + 1) for v in obj.values())
    if isinstance(obj, list):
        if not obj:
            return current + 1
        return max(_measure_depth(item, current + 1) for item in obj)
    return current


def check_file(path: str) -> LintReport:
    """Run all lint checks on a YAML file. Pure local — no API calls.

    Raises LintError if the file cannot be read.
    """
    p = Path(path)
    if not p.exists():
        raise LintError(f"File not found: {path}")
    if not p.is_file():
        raise LintError(f"Not a file: {path}")

    try:
        content = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise LintError(f"Cannot read binary file: {path}") from None

    violations: list[LintViolation] = []

    # L001: invalid YAML syntax
    try:
        data = _load_yaml_with_duplicates(content, violations, path)
    except yaml.YAMLError:
        rule = get_rule("L001")
        violations.append(
            LintViolation(
                rule_id="L001",
                file=path,
                message="Invalid YAML syntax.",
                severity=rule.severity if rule else LintSeverity.ERROR,
            )
        )
        return LintReport(
            violations=violations,
            summary="1 error (invalid YAML)",
            file_path=path,
        )

    if not isinstance(data, dict):
        return LintReport(
            violations=violations,
            summary=_build_summary(violations),
            file_path=path,
        )

    # L002: missing name field
    _check_missing_field(data, "name", "L002", path, violations)

    # L003: missing version field
    _check_missing_field(data, "version", "L003", path, violations)

    # L005: unused variables
    _check_unused_variables(data, content, path, violations)

    # L006: hardcoded secrets
    _check_hardcoded_secrets(content, path, violations)

    # L007: naming convention
    _check_naming_convention(data, path, violations)

    # L008: deep nesting
    _check_deep_nesting(data, path, violations)

    return LintReport(
        violations=violations,
        summary=_build_summary(violations),
        file_path=path,
    )


def _load_yaml_with_duplicates(content: str, violations: list[LintViolation], path: str) -> object:
    """Parse YAML and detect duplicate keys (L004)."""
    loader = _DuplicateKeyLoader(content)
    try:
        data = loader.get_single_data()
    finally:
        loader.dispose()

    dupes = getattr(loader, "duplicates", [])
    if dupes:
        rule = get_rule("L004")
        for key in dupes:
            violations.append(
                LintViolation(
                    rule_id="L004",
                    file=path,
                    message=f"Duplicate key: '{key}'.",
                    severity=rule.severity if rule else LintSeverity.ERROR,
                )
            )
    return data


def _check_missing_field(
    data: dict,
    field: str,
    rule_id: str,
    path: str,
    violations: list[LintViolation],
) -> None:
    """Check if a required/recommended field is missing."""
    if field not in data:
        rule = get_rule(rule_id)
        if rule:
            violations.append(
                LintViolation(
                    rule_id=rule_id,
                    file=path,
                    message=f"Missing '{field}' field.",
                    severity=rule.severity,
                )
            )


def _check_unused_variables(
    data: dict,
    content: str,
    path: str,
    violations: list[LintViolation],
) -> None:
    """Check for template variables defined but not used (L005)."""
    variables = data.get("variables", {})
    if not isinstance(variables, dict):
        return
    # Look for {{var}} references in prompt/system/template fields
    text_fields = ""
    for key in ("prompt", "system", "template", "messages"):
        val = data.get(key)
        if isinstance(val, str):
            text_fields += val
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, dict):
                    text_fields += str(item.get("content", ""))
                elif isinstance(item, str):
                    text_fields += item

    rule = get_rule("L005")
    for var_name in variables:
        # Check for {{var_name}} or {var_name} patterns
        if f"{{{{{var_name}}}}}" not in text_fields and f"{{{var_name}}}" not in text_fields:
            violations.append(
                LintViolation(
                    rule_id="L005",
                    file=path,
                    message=f"Variable '{var_name}' defined but not used in prompt.",
                    severity=rule.severity if rule else LintSeverity.WARNING,
                )
            )


def _check_hardcoded_secrets(
    content: str,
    path: str,
    violations: list[LintViolation],
) -> None:
    """Check for hardcoded secrets/API keys (L006)."""
    rule = get_rule("L006")
    for i, line in enumerate(content.splitlines(), 1):
        for pattern in SECRET_PATTERNS:
            if pattern.search(line):
                violations.append(
                    LintViolation(
                        rule_id="L006",
                        file=path,
                        line=i,
                        message="Potential hardcoded secret detected.",
                        suggestion="Use environment variables instead.",
                        severity=rule.severity if rule else LintSeverity.ERROR,
                    )
                )
                break  # One violation per line


def _check_naming_convention(
    data: dict,
    path: str,
    violations: list[LintViolation],
) -> None:
    """Check that template name follows convention (L007)."""
    name = data.get("name")
    if not isinstance(name, str):
        return
    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        rule = get_rule("L007")
        violations.append(
            LintViolation(
                rule_id="L007",
                file=path,
                message=f"Name '{name}' should be lowercase with hyphens/underscores.",
                suggestion="Use lowercase letters, digits, hyphens, or underscores.",
                severity=rule.severity if rule else LintSeverity.INFO,
            )
        )


def _check_deep_nesting(
    data: dict,
    path: str,
    violations: list[LintViolation],
) -> None:
    """Check for excessive nesting depth (L008)."""
    depth = _measure_depth(data)
    if depth > MAX_NESTING_DEPTH:
        rule = get_rule("L008")
        violations.append(
            LintViolation(
                rule_id="L008",
                file=path,
                message=f"Nesting depth {depth} exceeds maximum of {MAX_NESTING_DEPTH}.",
                suggestion="Consider flattening the structure.",
                severity=rule.severity if rule else LintSeverity.WARNING,
            )
        )


def _build_summary(violations: list[LintViolation]) -> str:
    """Build a human-readable summary string."""
    if not violations:
        return "No issues found."
    errors = sum(1 for v in violations if v.severity == LintSeverity.ERROR)
    warnings = sum(1 for v in violations if v.severity == LintSeverity.WARNING)
    infos = sum(1 for v in violations if v.severity == LintSeverity.INFO)
    parts = []
    if errors:
        parts.append(f"{errors} error{'s' if errors > 1 else ''}")
    if warnings:
        parts.append(f"{warnings} warning{'s' if warnings > 1 else ''}")
    if infos:
        parts.append(f"{infos} info")
    return ", ".join(parts) + "."
