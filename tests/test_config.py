"""Tests for promptctl.config."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from promptctl.config import get_api_key, get_config, init_config, set_value
from promptctl.exceptions import ConfigError


class TestInitConfig:
    def test_creates_config_file(self, tmp_config_dir: Path):
        path = init_config()
        assert path.exists()
        assert path.name == "config.yaml"

    def test_sets_permissions(self, tmp_config_dir: Path):
        path = init_config()
        mode = path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_contains_defaults(self, tmp_config_dir: Path):
        path = init_config()
        with open(path) as f:
            data = yaml.safe_load(f)
        assert "model" in data
        assert "temperature" in data
        assert "max_tokens" in data

    def test_already_exists_raises(self, tmp_config_dir: Path):
        init_config()
        with pytest.raises(ConfigError, match="already exists"):
            init_config()


class TestGetConfig:
    def test_defaults_when_no_file(self):
        config = get_config()
        assert "model" in config
        assert isinstance(config["temperature"], float)

    def test_reads_existing_file(self, tmp_config_dir: Path):
        config_path = tmp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"model": "claude-haiku-4-5-20251001", "custom_key": "value"}, f)
        config = get_config()
        assert config["model"] == "claude-haiku-4-5-20251001"
        assert config["custom_key"] == "value"

    def test_invalid_yaml_raises(self, tmp_config_dir: Path):
        config_path = tmp_config_dir / "config.yaml"
        config_path.write_text(": : bad yaml {{")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            get_config()

    def test_empty_file_returns_defaults(self, tmp_config_dir: Path):
        config_path = tmp_config_dir / "config.yaml"
        config_path.write_text("")
        config = get_config()
        assert "model" in config


class TestSetValue:
    def test_set_string(self, tmp_config_dir: Path):
        set_value("model", "claude-haiku-4-5-20251001")
        config = get_config()
        assert config["model"] == "claude-haiku-4-5-20251001"

    def test_set_temperature_coerces_float(self, tmp_config_dir: Path):
        set_value("temperature", "0.5")
        config = get_config()
        assert config["temperature"] == 0.5

    def test_set_max_tokens_coerces_int(self, tmp_config_dir: Path):
        set_value("max_tokens", "2048")
        config = get_config()
        assert config["max_tokens"] == 2048

    def test_invalid_float_raises(self, tmp_config_dir: Path):
        with pytest.raises(ConfigError, match="Invalid float"):
            set_value("temperature", "not-a-number")

    def test_invalid_int_raises(self, tmp_config_dir: Path):
        with pytest.raises(ConfigError, match="Invalid integer"):
            set_value("max_tokens", "not-a-number")

    def test_creates_config_if_missing(self):
        set_value("model", "test-model")
        config = get_config()
        assert config["model"] == "test-model"


class TestGetApiKey:
    def test_from_env(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-123")
        assert get_api_key() == "sk-ant-test-123"

    def test_from_config(self, tmp_config_dir: Path):
        config_path = tmp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"api_key": "sk-from-config"}, f)
        assert get_api_key() == "sk-from-config"

    def test_env_takes_precedence(self, monkeypatch: pytest.MonkeyPatch, tmp_config_dir: Path):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        config_path = tmp_config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"api_key": "sk-from-config"}, f)
        assert get_api_key() == "sk-from-env"

    def test_no_key_returns_empty(self):
        assert get_api_key() == ""
