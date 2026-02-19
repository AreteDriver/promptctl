"""Configuration management for promptctl."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import yaml

from promptctl.exceptions import ConfigError
from promptctl.models import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, DEFAULT_TEMPERATURE

_ENV_DIR = "PROMPTCTL_DIR"
_ENV_API_KEY = "ANTHROPIC_API_KEY"
_DEFAULT_DIR = Path.home() / ".promptctl"

_DEFAULTS: dict[str, str | float | int] = {
    "model": str(DEFAULT_MODEL),
    "temperature": DEFAULT_TEMPERATURE,
    "max_tokens": DEFAULT_MAX_TOKENS,
}


def _config_dir() -> Path:
    """Get config directory, respecting PROMPTCTL_DIR env override."""
    env = os.environ.get(_ENV_DIR, "")
    if env:
        return Path(env)
    return _DEFAULT_DIR


def _config_path() -> Path:
    return _config_dir() / "config.yaml"


def init_config() -> Path:
    """Create default config file. Returns path."""
    path = _config_path()
    if path.exists():
        raise ConfigError(f"Config already exists at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(dict(_DEFAULTS), f, default_flow_style=False)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600
    return path


def get_config() -> dict[str, str | float | int]:
    """Load config, falling back to defaults."""
    path = _config_path()
    if not path.exists():
        return dict(_DEFAULTS)
    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {path}: {e}") from e
    merged = dict(_DEFAULTS)
    merged.update(data)
    return merged


def set_value(key: str, value: str) -> None:
    """Set a config value. Creates config if needed."""
    path = _config_path()
    if not path.exists():
        init_config()
    config = get_config()
    # Coerce numeric types
    if key in ("temperature",):
        try:
            config[key] = float(value)
        except ValueError as e:
            raise ConfigError(f"Invalid float for '{key}': {value}") from e
    elif key in ("max_tokens",):
        try:
            config[key] = int(value)
        except ValueError as e:
            raise ConfigError(f"Invalid integer for '{key}': {value}") from e
    else:
        config[key] = value
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_api_key() -> str:
    """Get Anthropic API key from env or config."""
    key = os.environ.get(_ENV_API_KEY, "")
    if key:
        return key
    config = get_config()
    return str(config.get("api_key", ""))
