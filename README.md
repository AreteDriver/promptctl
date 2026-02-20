# promptctl

Claude API toolkit CLI — prompt engineering, code review, document intelligence, and template linting.

[![CI](https://github.com/AreteDriver/promptctl/actions/workflows/ci.yml/badge.svg)](https://github.com/AreteDriver/promptctl/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AreteDriver/promptctl/actions/workflows/codeql.yml/badge.svg)](https://github.com/AreteDriver/promptctl/actions/workflows/codeql.yml)

## Install

```bash
pip install -e ".[dev]"
```

Requires Python 3.10+ and an [Anthropic API key](https://console.anthropic.com/).

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### Prompt Engineering

```bash
# Run a prompt template
promptctl prompt run greeting.yaml

# Save a versioned snapshot
promptctl prompt version greeting.yaml

# View version history
promptctl prompt history greeting
```

### Code Review

Review code with structured findings across 6 dimensions:

```bash
# Review staged git changes
promptctl review diff

# Review a specific file
promptctl review file src/app.py --json
```

### Document Intelligence

Analyze, query, and summarize documents (long context + prompt caching):

```bash
# Extract key points, entities, themes
promptctl doc analyze report.md

# Ask questions about a document
promptctl doc ask report.md "What are the main findings?"

# Executive summary (map-reduce for large docs)
promptctl doc summarize report.md --json
```

### Template Linting

8 built-in rules for YAML prompt templates (no API required):

```bash
# Check a template for issues
promptctl lint check template.yaml

# List all lint rules
promptctl lint rules

# AI-powered fix suggestions (Pro)
promptctl lint fix template.yaml
```

### Configuration

```bash
promptctl config init              # Create default config
promptctl config set model claude-haiku-4-5-20251001
promptctl config show              # View current config
promptctl status                   # API key + license status
```

## Pro Features

Unlock additional capabilities with a Pro license key:

```bash
export PROMPTCTL_LICENSE=PCTL-XXXX-XXXX-XXXX
```

| Feature | Free | Pro |
|---------|------|-----|
| Prompt run (basic + streaming) | Yes | Yes |
| Prompt versioning | 5 max | Unlimited |
| Code review (diff, file) | Yes | Yes |
| Document intelligence (analyze, ask, summarize) | Yes | Yes |
| Template linting (local checks) | Yes | Yes |
| Multi-model comparison | - | Yes |
| AI-powered lint fixes | - | Yes |
| JSON/markdown export | - | Yes |

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v --cov=promptctl --cov-report=term-missing
ruff check src/ tests/ && ruff format --check src/ tests/
```

## License

MIT
