"""Tests for promptctl.exceptions."""

from __future__ import annotations

from promptctl.exceptions import (
    ClientError,
    ConfigError,
    DocError,
    LicenseError,
    LintError,
    PromptctlError,
    PromptError,
    ReviewError,
)


class TestExceptionHierarchy:
    def test_base_is_exception(self):
        assert issubclass(PromptctlError, Exception)

    def test_config_error_inherits(self):
        assert issubclass(ConfigError, PromptctlError)

    def test_client_error_inherits(self):
        assert issubclass(ClientError, PromptctlError)

    def test_prompt_error_inherits(self):
        assert issubclass(PromptError, PromptctlError)

    def test_review_error_inherits(self):
        assert issubclass(ReviewError, PromptctlError)

    def test_doc_error_inherits(self):
        assert issubclass(DocError, PromptctlError)

    def test_lint_error_inherits(self):
        assert issubclass(LintError, PromptctlError)

    def test_license_error_inherits(self):
        assert issubclass(LicenseError, PromptctlError)

    def test_all_catchable_by_base(self):
        for cls in (
            ConfigError,
            ClientError,
            PromptError,
            ReviewError,
            DocError,
            LintError,
            LicenseError,
        ):
            try:
                raise cls("test")
            except PromptctlError:
                pass

    def test_message_preserved(self):
        e = ConfigError("bad config")
        assert str(e) == "bad config"

    def test_all_constructible_with_message(self):
        for cls in (
            ConfigError,
            ClientError,
            PromptError,
            ReviewError,
            DocError,
            LintError,
            LicenseError,
        ):
            e = cls(f"{cls.__name__} test")
            assert cls.__name__ in str(e)
