"""Typer CLI for promptctl."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

import promptctl
from promptctl.exceptions import (
    ClientError,
    ConfigError,
    DocError,
    LicenseError,
    LintError,
    PromptError,
    ReviewError,
)
from promptctl.licensing import get_license
from promptctl.models import LintReport, ReviewReport

app = typer.Typer(
    name="promptctl",
    help="Claude API toolkit for prompt engineering and code review.",
)
prompt_app = typer.Typer(name="prompt", help="Prompt engineering commands.")
review_app = typer.Typer(name="review", help="Code review commands.")
doc_app = typer.Typer(name="doc", help="Document intelligence commands.")
lint_app = typer.Typer(name="lint", help="Prompt template linting.")
config_app = typer.Typer(name="config", help="Configuration management.")

app.add_typer(prompt_app)
app.add_typer(review_app)
app.add_typer(doc_app)
app.add_typer(lint_app)
app.add_typer(config_app)

console = Console()


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"promptctl {promptctl.__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """Claude API toolkit for prompt engineering and code review."""


@app.command()
def status() -> None:
    """Show version, license tier, and API key status."""
    from promptctl.config import get_api_key

    info = get_license()
    api_key = get_api_key()

    console.print(f"[bold]promptctl[/bold] {promptctl.__version__}")
    console.print(f"License: [cyan]{info.tier.value}[/cyan]")
    if api_key:
        if len(api_key) > 12:
            masked = api_key[:8] + "..." + api_key[-4:]
        else:
            masked = "***"
        console.print(f"API Key: [green]{masked}[/green]")
    else:
        console.print("API Key: [red]not set[/red]")


# --- Config commands ---


@config_app.command("init")
def config_init() -> None:
    """Create default config file."""
    from promptctl.config import init_config

    try:
        path = init_config()
        console.print(f"[green]Config created at {path}[/green]")
    except ConfigError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key to set.")],
    value: Annotated[str, typer.Argument(help="Value to set.")],
) -> None:
    """Set a config value."""
    from promptctl.config import set_value

    try:
        set_value(key, value)
        console.print(f"[green]Set {key} = {value}[/green]")
    except ConfigError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@config_app.command("show")
def config_show() -> None:
    """Show current configuration."""
    from promptctl.config import get_config

    config = get_config()
    for k, v in sorted(config.items()):
        if k == "api_key" and v:
            v_str = str(v)
            display = v_str[:8] + "..." if len(v_str) > 8 else "***"
            console.print(f"  {k}: {display}")
        else:
            console.print(f"  {k}: {v}")


# --- Prompt commands ---


@prompt_app.command("run")
def prompt_run(
    template: Annotated[str, typer.Argument(help="Path to prompt template YAML.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    temperature: Annotated[
        float | None,
        typer.Option("--temperature", "-t", help="Override temperature."),
    ] = None,
    stream: Annotated[bool, typer.Option("--stream", "-s", help="Stream response tokens.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Run a prompt template against Claude."""
    from promptctl.prompt.runner import run_prompt

    try:
        result = run_prompt(template, model=model, temperature=temperature, stream=stream)
        if json_output:
            console.print(result.model_dump_json(indent=2))
        else:
            tokens = f"{result.input_tokens}→{result.output_tokens}"
            console.print(
                f"\n[bold]{result.model}[/bold]  ({tokens} tokens, ${result.cost_usd:.4f})"
            )
            console.print(result.response)
    except (PromptError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@prompt_app.command("compare")
def prompt_compare(
    template: Annotated[str, typer.Argument(help="Path to prompt template YAML.")],
    models: Annotated[
        str | None,
        typer.Option("--models", help="Comma-separated model list."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Compare responses across multiple Claude models (Pro)."""
    from promptctl.prompt.comparator import compare_models

    try:
        result = compare_models(template, models=models)
        if json_output:
            console.print(result.model_dump_json(indent=2))
        else:
            from rich.table import Table

            table = Table(title="Model Comparison")
            table.add_column("Model")
            table.add_column("Tokens (in→out)")
            table.add_column("Cost")
            table.add_column("Latency")
            table.add_column("Response Preview")
            for entry in result.entries:
                table.add_row(
                    entry.model,
                    f"{entry.input_tokens}→{entry.output_tokens}",
                    f"${entry.cost_usd:.4f}",
                    f"{entry.latency_ms:.0f}ms",
                    entry.response_preview[:80],
                )
            console.print(table)
    except (PromptError, ClientError, LicenseError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@prompt_app.command("version")
def prompt_version(
    template: Annotated[str, typer.Argument(help="Path to prompt template YAML.")],
) -> None:
    """Save a prompt template as a versioned snapshot."""
    from promptctl.prompt.versioner import save_version

    try:
        version_num, path = save_version(template)
        console.print(f"[green]Saved as v{version_num}[/green] → {path}")
    except PromptError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@prompt_app.command("history")
def prompt_history(
    name: Annotated[str, typer.Argument(help="Prompt name to list versions for.")],
) -> None:
    """List versioned snapshots for a prompt."""
    from promptctl.prompt.versioner import list_versions

    try:
        versions = list_versions(name)
        if not versions:
            console.print(f"No versions found for '{name}'")
            return
        for v in versions:
            console.print(f"  v{v['version']}  {v['path']}")
    except PromptError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


# --- Review commands ---


@review_app.command("diff")
def review_diff(
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Review staged git diff."""
    from promptctl.review.differ import get_staged_diff
    from promptctl.review.reviewer import review_code

    try:
        diff_text = get_staged_diff()
        if not diff_text.strip():
            console.print("[yellow]No staged changes to review.[/yellow]")
            raise typer.Exit()
        report = review_code(diff_text, model=model)
        _print_review(report, json_output)
    except (ReviewError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@review_app.command("file")
def review_file(
    path: Annotated[str, typer.Argument(help="Path to file to review.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Review a specific file."""
    from promptctl.review.differ import get_file_content
    from promptctl.review.reviewer import review_code

    try:
        content = get_file_content(path)
        report = review_code(content, model=model, source_file=path)
        _print_review(report, json_output)
    except (ReviewError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


def _print_review(report: ReviewReport, json_output: bool) -> None:
    """Format and print a review report."""
    if json_output:
        console.print(report.model_dump_json(indent=2))
        return

    from rich.table import Table

    if report.summary:
        console.print(f"\n[bold]Summary:[/bold] {report.summary}")

    if report.findings:
        table = Table(title="Review Findings")
        table.add_column("Severity", style="bold")
        table.add_column("Category")
        table.add_column("Location")
        table.add_column("Message")
        for f in report.findings:
            sev_style = {
                "error": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(f.severity, "")
            loc = f"{f.file}:{f.line}" if f.file and f.line else f.file or ""
            table.add_row(
                f"[{sev_style}]{f.severity}[/{sev_style}]",
                f.category,
                loc,
                f.message,
            )
        console.print(table)
    else:
        console.print("[green]No issues found.[/green]")

    console.print(
        f"\nModel: {report.model}  "
        f"Tokens: {report.input_tokens}→{report.output_tokens}  "
        f"Cost: ${report.cost_usd:.4f}"
    )


# --- Doc commands ---


@doc_app.command("analyze")
def doc_analyze(
    file: Annotated[str, typer.Argument(help="Path to document file.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Analyze a document — extract key points, entities, themes."""
    from promptctl.doc.analyzer import analyze_document

    try:
        result = analyze_document(file, model=model)
        if json_output:
            console.print(result.model_dump_json(indent=2))
        else:
            console.print(f"\n[bold]Summary:[/bold] {result.summary}")
            if result.key_points:
                console.print("\n[bold]Key Points:[/bold]")
                for kp in result.key_points:
                    console.print(f"  - {kp}")
            if result.entities:
                console.print(f"\n[bold]Entities:[/bold] {', '.join(result.entities)}")
            if result.themes:
                console.print(f"\n[bold]Themes:[/bold] {', '.join(result.themes)}")
            console.print(f"\n{result.word_count} words, ~{result.token_estimate} tokens")
    except (DocError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@doc_app.command("ask")
def doc_ask(
    file: Annotated[str, typer.Argument(help="Path to document file.")],
    question: Annotated[str, typer.Argument(help="Question about the document.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Ask a question about a document."""
    from promptctl.doc.analyzer import ask_document

    try:
        result = ask_document(file, question, model=model)
        if json_output:
            console.print(result.model_dump_json(indent=2))
        else:
            console.print(f"\n[bold]Answer:[/bold] {result.answer}")
            if result.confidence:
                console.print(f"Confidence: {result.confidence}")
            if result.source_quotes:
                console.print("\n[bold]Sources:[/bold]")
                for q in result.source_quotes:
                    console.print(f'  "{q}"')
    except (DocError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@doc_app.command("summarize")
def doc_summarize(
    file: Annotated[str, typer.Argument(help="Path to document file.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Generate an executive summary of a document."""
    from promptctl.doc.analyzer import summarize_document

    try:
        result = summarize_document(file, model=model)
        if json_output:
            console.print(result.model_dump_json(indent=2))
        else:
            console.print(f"\n[bold]Summary:[/bold] {result.executive_summary}")
            if result.sections:
                console.print("\n[bold]Sections:[/bold]")
                for s in result.sections:
                    console.print(f"  - {s}")
            if result.chunks_processed > 1:
                console.print(f"\n(Map-reduce: {result.chunks_processed} chunks)")
    except (DocError, ClientError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


# --- Lint commands ---


@lint_app.command("check")
def lint_check(
    file: Annotated[str, typer.Argument(help="Path to YAML file to lint.")],
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Check a YAML prompt template for issues (no API call)."""
    from promptctl.lint.checker import check_file

    try:
        report = check_file(file)
        if json_output:
            console.print(report.model_dump_json(indent=2))
        else:
            _print_lint_report(report)
    except LintError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@lint_app.command("fix")
def lint_fix(
    file: Annotated[str, typer.Argument(help="Path to YAML file to fix.")],
    model: Annotated[str | None, typer.Option("--model", "-m", help="Override model.")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """Suggest AI-powered fixes for lint violations (Pro)."""
    from promptctl.lint.fixer import fix_violations

    try:
        fix = fix_violations(file, model=model)
        if json_output:
            console.print(fix.model_dump_json(indent=2))
        else:
            if not fix.violations_addressed:
                console.print("[green]No violations to fix.[/green]")
                return
            console.print(f"\n[bold]Explanation:[/bold] {fix.explanation}")
            console.print("\n[bold]Fixed YAML:[/bold]")
            console.print(fix.fixed)
            console.print(f"\nAddressed: {', '.join(fix.violations_addressed)}")
    except (LintError, ClientError, LicenseError) as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None


@lint_app.command("rules")
def lint_rules(
    json_output: Annotated[bool, typer.Option("--json", help="Output as JSON.")] = False,
) -> None:
    """List all built-in lint rules."""
    from promptctl.lint.rules import get_all_rules

    rules = get_all_rules()
    if json_output:
        import json

        console.print(json.dumps([r.model_dump() for r in rules], indent=2))
    else:
        from rich.table import Table

        table = Table(title="Lint Rules")
        table.add_column("ID", style="bold")
        table.add_column("Name")
        table.add_column("Severity")
        table.add_column("Category")
        table.add_column("Description")
        for r in rules:
            sev_style = {
                "error": "red",
                "warning": "yellow",
                "info": "blue",
            }.get(r.severity.value, "")
            table.add_row(
                r.id,
                r.name,
                f"[{sev_style}]{r.severity.value}[/{sev_style}]",
                r.category.value,
                r.description,
            )
        console.print(table)


def _print_lint_report(report: LintReport) -> None:
    """Format and print a lint report."""
    if not report.violations:
        console.print("[green]No issues found.[/green]")
        return

    from rich.table import Table

    table = Table(title="Lint Results")
    table.add_column("Rule", style="bold")
    table.add_column("Severity")
    table.add_column("Line")
    table.add_column("Message")
    for v in report.violations:
        sev_style = {
            "error": "red",
            "warning": "yellow",
            "info": "blue",
        }.get(v.severity.value, "")
        table.add_row(
            v.rule_id,
            f"[{sev_style}]{v.severity.value}[/{sev_style}]",
            str(v.line) if v.line else "",
            v.message,
        )
    console.print(table)
    console.print(f"\n{report.summary}")
