"""Git diff extraction for code review."""

from __future__ import annotations

import subprocess
from pathlib import Path

from promptctl.exceptions import ReviewError


def get_staged_diff() -> str:
    """Get the staged git diff (git diff --cached)."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except FileNotFoundError:
        raise ReviewError("git not found. Ensure git is installed.") from None
    except subprocess.CalledProcessError as e:
        raise ReviewError(f"git diff failed: {e.stderr.strip()}") from e
    except subprocess.TimeoutExpired:
        raise ReviewError("git diff timed out after 30 seconds") from None
    return result.stdout


def get_file_content(path: str) -> str:
    """Read a file's content for review."""
    p = Path(path)
    if not p.exists():
        raise ReviewError(f"File not found: {path}")
    if not p.is_file():
        raise ReviewError(f"Not a file: {path}")
    try:
        return p.read_text()
    except (OSError, UnicodeDecodeError) as e:
        raise ReviewError(f"Could not read {path}: {e}") from e
