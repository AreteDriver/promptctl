"""Prompt version management — save immutable snapshots."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from promptctl.exceptions import PromptError
from promptctl.licensing import MAX_FREE_VERSIONS, get_license

_ENV_DIR = "PROMPTCTL_DIR"
_DEFAULT_DIR = Path.home() / ".promptctl"


def _prompts_dir() -> Path:
    env = os.environ.get(_ENV_DIR, "")
    base = Path(env) if env else _DEFAULT_DIR
    return base / "prompts"


def save_version(template_path: str) -> tuple[int, Path]:
    """Save a prompt template as a versioned snapshot.

    Returns (version_number, saved_path).
    """
    src = Path(template_path)
    if not src.exists():
        raise PromptError(f"Template file not found: {template_path}")

    name = src.stem
    prompt_dir = _prompts_dir() / name
    prompt_dir.mkdir(parents=True, exist_ok=True)

    # Determine next version number
    existing = _get_version_numbers(prompt_dir)
    next_version = max(existing, default=0) + 1

    # Check free tier limit
    license_info = get_license()
    if not license_info.is_pro and next_version > MAX_FREE_VERSIONS:
        raise PromptError(
            f"Free tier limited to {MAX_FREE_VERSIONS} versions per prompt. "
            "Upgrade to Pro for unlimited versions."
        )

    dest = prompt_dir / f"v{next_version}.yaml"
    shutil.copy2(src, dest)
    return next_version, dest


def list_versions(name: str) -> list[dict[str, int | str]]:
    """List all versions for a named prompt."""
    prompt_dir = _prompts_dir() / name
    if not prompt_dir.exists():
        return []

    versions = []
    for num in sorted(_get_version_numbers(prompt_dir)):
        path = prompt_dir / f"v{num}.yaml"
        versions.append({"version": num, "path": str(path)})
    return versions


def load_version(name: str, version: int) -> Path:
    """Get the path to a specific version."""
    prompt_dir = _prompts_dir() / name
    path = prompt_dir / f"v{version}.yaml"
    if not path.exists():
        raise PromptError(f"Version v{version} not found for prompt '{name}'")
    return path


def _get_version_numbers(prompt_dir: Path) -> list[int]:
    """Extract version numbers from v{N}.yaml files."""
    numbers = []
    for f in prompt_dir.iterdir():
        match = re.match(r"v(\d+)\.yaml$", f.name)
        if match:
            numbers.append(int(match.group(1)))
    return numbers
