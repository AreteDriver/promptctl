# promptctl

Claude API toolkit CLI — prompt engineering, code review, and more.

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

Run a prompt template against Claude:

```bash
# Create a template
cat > greeting.yaml <<EOF
name: greeting
system: You are a friendly assistant.
user: "Say hello to {name} in {language}."
variables:
  name: World
  language: French
EOF

# Run it
promptctl prompt run greeting.yaml

# Save a versioned snapshot
promptctl prompt version greeting.yaml

# View version history
promptctl prompt history greeting
```

### Code Review

Review code with structured findings across 6 dimensions (correctness, security, performance, maintainability, testing, style):

```bash
# Review staged git changes
promptctl review diff

# Review a specific file
promptctl review file src/app.py

# JSON output
promptctl review file src/app.py --json
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
| Multi-model comparison | - | Yes |
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
