"""Tests for promptctl.review.differ."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from promptctl.exceptions import ReviewError
from promptctl.review.differ import get_file_content, get_staged_diff


class TestGetStagedDiff:
    def test_returns_diff_output(self):
        mock_result = type(
            "CompletedProcess",
            (),
            {"stdout": "diff --git a/foo.py\n+hello", "returncode": 0},
        )()
        with patch("promptctl.review.differ.subprocess.run", return_value=mock_result):
            diff = get_staged_diff()
        assert "foo.py" in diff

    def test_empty_diff(self):
        mock_result = type("CompletedProcess", (), {"stdout": "", "returncode": 0})()
        with patch("promptctl.review.differ.subprocess.run", return_value=mock_result):
            diff = get_staged_diff()
        assert diff == ""

    def test_git_not_found(self):
        with patch(
            "promptctl.review.differ.subprocess.run",
            side_effect=FileNotFoundError,
        ):
            with pytest.raises(ReviewError, match="git not found"):
                get_staged_diff()

    def test_git_error(self):
        import subprocess

        with patch(
            "promptctl.review.differ.subprocess.run",
            side_effect=subprocess.CalledProcessError(128, "git", stderr="fatal: not a git repo"),
        ):
            with pytest.raises(ReviewError, match="git diff failed"):
                get_staged_diff()

    def test_timeout(self):
        import subprocess

        with patch(
            "promptctl.review.differ.subprocess.run",
            side_effect=subprocess.TimeoutExpired("git", 30),
        ):
            with pytest.raises(ReviewError, match="timed out"):
                get_staged_diff()


class TestGetFileContent:
    def test_reads_file(self, tmp_path: Path):
        f = tmp_path / "code.py"
        f.write_text("print('hello')")
        content = get_file_content(str(f))
        assert content == "print('hello')"

    def test_file_not_found(self):
        with pytest.raises(ReviewError, match="not found"):
            get_file_content("/nonexistent/file.py")

    def test_not_a_file(self, tmp_path: Path):
        with pytest.raises(ReviewError, match="Not a file"):
            get_file_content(str(tmp_path))

    def test_binary_file(self, tmp_path: Path):
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02\xff")
        # UnicodeDecodeError should be caught
        with pytest.raises(ReviewError, match="Could not read"):
            get_file_content(str(f))
