"""Tests for promptctl.lint.checker."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptctl.exceptions import LintError
from promptctl.lint.checker import _measure_depth, check_file


def _write_yaml(tmp_path: Path, name: str, content: str) -> str:
    f = tmp_path / name
    f.write_text(content)
    return str(f)


class TestCheckFileErrors:
    def test_file_not_found(self):
        with pytest.raises(LintError, match="not found"):
            check_file("/nonexistent.yaml")

    def test_not_a_file(self, tmp_path):
        with pytest.raises(LintError, match="Not a file"):
            check_file(str(tmp_path))


class TestL001InvalidYaml:
    def test_invalid_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "bad.yaml", "{{invalid: yaml: [")
        report = check_file(path)
        assert any(v.rule_id == "L001" for v in report.violations)
        assert report.error_count >= 1

    def test_valid_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "good.yaml", "name: test\nversion: '1'")
        report = check_file(path)
        assert not any(v.rule_id == "L001" for v in report.violations)


class TestL002MissingName:
    def test_missing_name(self, tmp_path):
        path = _write_yaml(tmp_path, "no_name.yaml", "version: '1'\nprompt: hello")
        report = check_file(path)
        assert any(v.rule_id == "L002" for v in report.violations)

    def test_has_name(self, tmp_path):
        path = _write_yaml(tmp_path, "has_name.yaml", "name: test\nversion: '1'")
        report = check_file(path)
        assert not any(v.rule_id == "L002" for v in report.violations)


class TestL003MissingVersion:
    def test_missing_version(self, tmp_path):
        path = _write_yaml(tmp_path, "no_ver.yaml", "name: test\nprompt: hello")
        report = check_file(path)
        assert any(v.rule_id == "L003" for v in report.violations)

    def test_has_version(self, tmp_path):
        path = _write_yaml(tmp_path, "has_ver.yaml", "name: test\nversion: '1'")
        report = check_file(path)
        assert not any(v.rule_id == "L003" for v in report.violations)


class TestL004DuplicateKeys:
    def test_duplicate_keys(self, tmp_path):
        content = "name: first\nname: second\nversion: '1'"
        path = _write_yaml(tmp_path, "dupes.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L004" for v in report.violations)
        assert any("Duplicate key" in v.message for v in report.violations)

    def test_no_duplicates(self, tmp_path):
        path = _write_yaml(tmp_path, "clean.yaml", "name: test\nversion: '1'")
        report = check_file(path)
        assert not any(v.rule_id == "L004" for v in report.violations)


class TestL005UnusedVariables:
    def test_unused_variable(self, tmp_path):
        content = (
            "name: test\nversion: '1'\nvariables:\n  unused_var: hello\nprompt: No variables here"
        )
        path = _write_yaml(tmp_path, "unused.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L005" for v in report.violations)
        assert any("unused_var" in v.message for v in report.violations)

    def test_used_variable_double_brace(self, tmp_path):
        content = "name: test\nversion: '1'\nvariables:\n  user: world\nprompt: Hello {{user}}"
        path = _write_yaml(tmp_path, "used.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L005" for v in report.violations)

    def test_used_variable_single_brace(self, tmp_path):
        content = "name: test\nversion: '1'\nvariables:\n  user: world\nprompt: Hello {user}"
        path = _write_yaml(tmp_path, "used_single.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L005" for v in report.violations)

    def test_no_variables_section(self, tmp_path):
        content = "name: test\nversion: '1'\nprompt: hello"
        path = _write_yaml(tmp_path, "no_vars.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L005" for v in report.violations)


class TestL006HardcodedSecrets:
    def test_api_key_pattern(self, tmp_path):
        content = "name: test\nversion: '1'\napi_key: 'sk-ant-abcdefghijklmnop'"
        path = _write_yaml(tmp_path, "secret.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L006" for v in report.violations)

    def test_password_pattern(self, tmp_path):
        content = "name: test\nversion: '1'\npassword: 'my_secret_password'"
        path = _write_yaml(tmp_path, "pw.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L006" for v in report.violations)

    def test_no_secrets(self, tmp_path):
        content = "name: test\nversion: '1'\nprompt: Hello world"
        path = _write_yaml(tmp_path, "clean.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L006" for v in report.violations)

    def test_line_number_reported(self, tmp_path):
        content = "name: test\nversion: '1'\nkey: 'sk-ant-abcdefghijklmnop'"
        path = _write_yaml(tmp_path, "line.yaml", content)
        report = check_file(path)
        secrets = [v for v in report.violations if v.rule_id == "L006"]
        assert secrets[0].line == 3


class TestL007NamingConvention:
    def test_uppercase_name(self, tmp_path):
        content = "name: MyPrompt\nversion: '1'"
        path = _write_yaml(tmp_path, "upper.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L007" for v in report.violations)

    def test_valid_name_with_hyphens(self, tmp_path):
        content = "name: my-prompt\nversion: '1'"
        path = _write_yaml(tmp_path, "good.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L007" for v in report.violations)

    def test_valid_name_with_underscores(self, tmp_path):
        content = "name: my_prompt\nversion: '1'"
        path = _write_yaml(tmp_path, "good2.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L007" for v in report.violations)

    def test_name_not_string(self, tmp_path):
        content = "name: 123\nversion: '1'"
        path = _write_yaml(tmp_path, "int.yaml", content)
        report = check_file(path)
        # Non-string name — no L007 violation
        assert not any(v.rule_id == "L007" for v in report.violations)


class TestL008DeepNesting:
    def test_deeply_nested(self, tmp_path):
        content = (
            "name: test\nversion: '1'\n"
            "a:\n  b:\n    c:\n      d:\n        e:\n"
            "          f:\n            g: deep"
        )
        path = _write_yaml(tmp_path, "deep.yaml", content)
        report = check_file(path)
        assert any(v.rule_id == "L008" for v in report.violations)

    def test_shallow_nesting(self, tmp_path):
        content = "name: test\nversion: '1'\na:\n  b: c"
        path = _write_yaml(tmp_path, "shallow.yaml", content)
        report = check_file(path)
        assert not any(v.rule_id == "L008" for v in report.violations)


class TestMeasureDepth:
    def test_flat_dict(self):
        assert _measure_depth({"a": 1, "b": 2}) == 1

    def test_nested_dict(self):
        assert _measure_depth({"a": {"b": {"c": 1}}}) == 3

    def test_list(self):
        assert _measure_depth({"a": [1, 2, 3]}) == 2

    def test_scalar(self):
        assert _measure_depth("hello") == 0

    def test_empty_dict(self):
        assert _measure_depth({}) == 1

    def test_empty_list(self):
        assert _measure_depth([]) == 1


class TestSummary:
    def test_clean_file(self, tmp_path):
        content = "name: test\nversion: '1'\nprompt: hello"
        path = _write_yaml(tmp_path, "clean.yaml", content)
        report = check_file(path)
        assert report.summary == "No issues found."

    def test_error_summary(self, tmp_path):
        path = _write_yaml(tmp_path, "bad.yaml", "{{invalid")
        report = check_file(path)
        assert "error" in report.summary

    def test_non_dict_yaml(self, tmp_path):
        path = _write_yaml(tmp_path, "list.yaml", "- item1\n- item2")
        report = check_file(path)
        # Non-dict YAML should not crash
        assert report.file_path == path
