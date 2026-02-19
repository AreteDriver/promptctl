"""Tests for promptctl.licensing."""

from __future__ import annotations

import pytest

from promptctl.exceptions import LicenseError
from promptctl.licensing import (
    LicenseInfo,
    Tier,
    _compute_checksum,
    generate_key,
    get_license,
    validate_key,
)


class TestTier:
    def test_free_value(self):
        assert Tier.FREE == "free"

    def test_pro_value(self):
        assert Tier.PRO == "pro"

    def test_is_string(self):
        assert isinstance(Tier.FREE, str)


class TestLicenseInfo:
    def test_free_tier(self):
        info = LicenseInfo(tier=Tier.FREE)
        assert not info.is_pro
        assert info.key == ""

    def test_pro_tier(self):
        info = LicenseInfo(tier=Tier.PRO, key="PCTL-1234-5678-ABCD")
        assert info.is_pro
        assert info.key == "PCTL-1234-5678-ABCD"


class TestComputeChecksum:
    def test_deterministic(self):
        a = _compute_checksum("TEST-KEY0")
        b = _compute_checksum("TEST-KEY0")
        assert a == b

    def test_four_chars_uppercase(self):
        result = _compute_checksum("XXXX-YYYY")
        assert len(result) == 4
        assert result == result.upper()

    def test_different_bodies_differ(self):
        a = _compute_checksum("AAAA-BBBB")
        b = _compute_checksum("CCCC-DDDD")
        assert a != b


class TestValidateKey:
    def test_valid_key(self):
        key = generate_key()
        info = validate_key(key)
        assert info.is_pro
        assert info.key == key

    def test_empty_key_raises(self):
        with pytest.raises(LicenseError, match="Empty"):
            validate_key("")

    def test_wrong_segment_count(self):
        with pytest.raises(LicenseError, match="segments"):
            validate_key("PCTL-1234-5678")

    def test_too_many_segments(self):
        with pytest.raises(LicenseError, match="segments"):
            validate_key("PCTL-1234-5678-ABCD-EXTRA")

    def test_wrong_prefix(self):
        with pytest.raises(LicenseError, match="prefix"):
            validate_key("ASPD-1234-5678-ABCD")

    def test_bad_checksum(self):
        with pytest.raises(LicenseError, match="checksum"):
            validate_key("PCTL-1234-5678-ZZZZ")

    def test_whitespace_stripped(self):
        key = generate_key()
        info = validate_key(f"  {key}  ")
        assert info.is_pro

    def test_case_insensitive_checksum(self):
        key = generate_key("ABCD-EFGH")
        parts = key.split("-")
        mixed = f"PCTL-{parts[1]}-{parts[2]}-{parts[3].lower()}"
        info = validate_key(mixed)
        assert info.is_pro


class TestGenerateKey:
    def test_default_body(self):
        key = generate_key()
        assert key.startswith("PCTL-TEST-KEY0-")
        assert len(key.split("-")) == 4

    def test_custom_body(self):
        key = generate_key("CUST-BODY")
        assert key.startswith("PCTL-CUST-BODY-")

    def test_roundtrip(self):
        key = generate_key("AABB-CCDD")
        info = validate_key(key)
        assert info.is_pro


class TestGetLicense:
    def test_no_env_returns_free(self):
        info = get_license()
        assert not info.is_pro
        assert info.tier == Tier.FREE

    def test_valid_env_returns_pro(self, monkeypatch: pytest.MonkeyPatch):
        key = generate_key()
        monkeypatch.setenv("PROMPTCTL_LICENSE", key)
        info = get_license()
        assert info.is_pro

    def test_invalid_env_returns_free(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("PROMPTCTL_LICENSE", "PCTL-XXXX-YYYY-ZZZZ")
        info = get_license()
        assert not info.is_pro
