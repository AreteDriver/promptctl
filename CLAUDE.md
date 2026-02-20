# promptctl — Claude Code Guide

## Quick Start
```bash
cd /home/arete/projects/promptctl
source .venv/bin/activate
pytest tests/ -q                          # 286 tests
ruff check src/ tests/ && ruff format --check src/ tests/  # lint
```

## Project Structure
```
src/promptctl/
├── __init__.py          # __version__ = "0.2.0"
├── __main__.py          # Entry point
├── cli.py               # Typer app + sub-Typers (prompt, review, doc, lint, config)
├── client.py            # Anthropic SDK wrapper (send_message, send_message_with_tools, streaming)
├── config.py            # ~/.promptctl/config.yaml management
├── exceptions.py        # PromptctlError hierarchy (6 subtypes)
├── gates.py             # @require_pro decorator
├── licensing.py         # PCTL-XXXX-XXXX-XXXX HMAC keys
├── models.py            # Pydantic v2 models + LintSeverity/LintCategory enums
├── prompt/
│   ├── runner.py        # Load YAML templates, interpolate, run
│   ├── versioner.py     # Immutable prompt version snapshots
│   └── comparator.py    # Multi-model A/B comparison (Pro)
├── review/
│   ├── differ.py        # Git diff extraction
│   └── reviewer.py      # Structured 6-dimension code review
├── doc/
│   ├── analyzer.py      # analyze_document, ask_document, summarize_document
│   └── chunker.py       # Map-reduce chunking for large documents
└── lint/
    ├── rules.py         # 8 built-in lint rules (L001-L008)
    ├── checker.py       # Pure local YAML checker (no API)
    └── fixer.py         # AI-powered fix suggestions (Pro)
```

## Architecture
- **Typer CLI** with sub-Typers: `prompt_app`, `review_app`, `doc_app`, `lint_app`, `config_app`
- **Anthropic SDK** wrapper in `client.py` — never call SDK directly from CLI
- **Pydantic v2** models for all data structures
- **Free/Pro licensing** via HMAC keys (`PCTL` prefix, `promptctl-v1` salt)
- **Config** at `~/.promptctl/config.yaml` (overridable via `PROMPTCTL_DIR` env)
- **Doc module**: Long context + prompt caching (cache_control) + map-reduce for large docs
- **Lint module**: Pure local checks (checker.py) + AI fix suggestions (fixer.py, Pro-gated)

## Testing
- Mock `anthropic.Anthropic` with `SimpleNamespace` — never call real API
- `conftest.py` has autouse `_isolate` fixture (redirects config dir, clears env)
- `pro_license_env` fixture for Pro-gated tests
- `sample_template_path` fixture for prompt template tests

## Commands
```
promptctl --version              # Version info
promptctl status                 # API key, license, config status
promptctl config init            # Create default config
promptctl config set K V         # Set config value
promptctl config show            # Show current config
promptctl prompt run FILE        # Run prompt template
promptctl prompt version FILE    # Save versioned snapshot
promptctl prompt history NAME    # List prompt versions
promptctl prompt compare FILE    # Multi-model comparison (Pro)
promptctl review diff            # Review staged git diff
promptctl review file PATH       # Review a specific file
promptctl doc analyze FILE       # Key points, entities, themes
promptctl doc ask FILE QUESTION  # Q&A about a document
promptctl doc summarize FILE     # Executive summary (map-reduce for large)
promptctl lint check FILE        # Local YAML lint (no API)
promptctl lint fix FILE          # AI fix suggestions (Pro)
promptctl lint rules             # List all 8 built-in rules
```

## Conventions
- `line-length = 100` (ruff)
- `from __future__ import annotations` in all files
- `raise typer.Exit(1) from None` in except blocks (B904)
- Coverage gate: 90% (`fail_under = 90`)
- License salt in `.gitleaks.toml` allowlist
- StrEnum backport for Python 3.10 compatibility
