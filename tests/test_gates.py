"""Tests for promptctl.gates."""

from __future__ import annotations

import pytest

from promptctl.exceptions import LicenseError
from promptctl.gates import require_pro


class TestRequirePro:
    def test_free_tier_raises(self):
        @require_pro("test_feature")
        def gated_func():
            return "success"

        with pytest.raises(LicenseError, match="test_feature"):
            gated_func()

    def test_pro_tier_passes(self, pro_license_env: str):
        @require_pro("test_feature")
        def gated_func():
            return "success"

        assert gated_func() == "success"

    def test_error_message_includes_env_var(self):
        @require_pro("export")
        def gated_func():
            return "success"

        with pytest.raises(LicenseError, match="PROMPTCTL_LICENSE"):
            gated_func()

    def test_preserves_function_name(self):
        @require_pro("feature")
        def my_function():
            """My docstring."""
            return 42

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."

    def test_passes_args_and_kwargs(self, pro_license_env: str):
        @require_pro("feature")
        def add(a: int, b: int, extra: int = 0) -> int:
            return a + b + extra

        assert add(1, 2, extra=3) == 6

    def test_returns_none_when_function_returns_none(self, pro_license_env: str):
        @require_pro("feature")
        def void_func():
            pass

        assert void_func() is None
