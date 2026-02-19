# promptctl — Claude Code Guide

## Quick Start
```bash
cd /home/arete/projects/promptctl
source .venv/bin/activate
pytest tests/ -q                          # 170 tests
ruff check src/ tests/ && ruff format --check src/ tests/  # lint
```

## Project Structure
```
src/promptctl/
├── __init__.py          # __version__ = "0.1.0"
├── __main__.py          # Entry point
├── cli.py               # Typer app + sub-Typers (prompt, review, config)
├── client.py            # Anthropic SDK wrapper (send_message, streaming)
├── config.py            # ~/.promptctl/config.yaml management
├── exceptions.py        # PromptctlError hierarchy
├── gates.py             # @require_pro decorator
├── licensing.py         # PCTL-XXXX-XXXX-XXXX HMAC keys
├── models.py            # Pydantic v2 models (PromptTemplate, ReviewReport, etc.)
├── prompt/
│   ├── runner.py        # Load YAML templates, interpolate, run
│   ├── versioner.py     # Immutable prompt version snapshots
│   └── comparator.py    # Multi-model A/B comparison (Pro)
└── review/
    ├── differ.py        # Git diff extraction
    └── reviewer.py      # Structured 6-dimension code review
```

## Architecture
- **Typer CLI** with sub-Typers: `prompt_app`, `review_app`, `config_app`
- **Anthropic SDK** wrapper in `client.py` — never call SDK directly from CLI
- **Pydantic v2** models for all data structures
- **Free/Pro licensing** via HMAC keys (`PCTL` prefix, `promptctl-v1` salt)
- **Config** at `~/.promptctl/config.yaml` (overridable via `PROMPTCTL_DIR` env)

## Testing
- Mock `anthropic.Anthropic` with `SimpleNamespace` — never call real API
- `conftest.py` has autouse `_isolate` fixture (redirects config dir, clears env)
- `pro_license_env` fixture for Pro-gated tests
- `sample_template_path` fixture for prompt template tests

## Commands
```
promptctl --version          # Version info
promptctl status             # API key, license, config status
promptctl config init        # Create default config
promptctl config set K V     # Set config value
promptctl config show        # Show current config
promptctl prompt run FILE    # Run prompt template
promptctl prompt version FILE # Save versioned snapshot
promptctl prompt history NAME # List prompt versions
promptctl prompt compare FILE # Multi-model comparison (Pro)
promptctl review diff        # Review staged git diff
promptctl review file PATH   # Review a specific file
```

## Conventions
- `line-length = 100` (ruff)
- `from __future__ import annotations` in all files
- `raise typer.Exit(1) from None` in except blocks (B904)
- Coverage gate: 90% (`fail_under = 90`)
- License salt in `.gitleaks.toml` allowlist
