from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from deepdive.config import DeepDiveConfig
from deepdive.models import ResearchReport
from deepdive.pipeline import ResearchPipeline
from deepdive.report.markdown import report_to_markdown

console = Console()


async def _run_research(question: str, config: DeepDiveConfig, output_path: Path | None) -> None:
    pipeline = ResearchPipeline(config=config)
    report: ResearchReport | None = None

    with Live(console=console, refresh_per_second=4) as live:
        async for event in pipeline.run(question):
            if event.type == "start":
                live.update(Status(f"[bold]Researching:[/] {question}"))

            elif event.type == "queries_generated":
                queries = event.data["queries"]
                live.update(
                    Panel(
                        "\n".join(f"  • {q}" for q in queries),
                        title=f"[cyan]Generated {len(queries)} search queries[/]",
                    )
                )

            elif event.type == "search_result":
                live.update(
                    Status(
                        f"[green]Found:[/] {event.data['title'][:60]}  "
                        f"[dim]({event.data['url'][:50]})[/]"
                    )
                )

            elif event.type == "page_scraped":
                chars = event.data["chars"]
                live.update(
                    Status(
                        f"[yellow]Scraped:[/] {event.data['url'][:60]}  [dim]({chars:,} chars)[/]"
                    )
                )

            elif event.type == "claims_extracted":
                live.update(
                    Status(
                        f"[magenta]Extracted {event.data['count']} claims from[/] "
                        f"{event.data['url'][:50]}"
                    )
                )

            elif event.type == "report_section":
                live.update(Status(f"[blue]Writing section:[/] {event.data['heading']}"))

            elif event.type == "done":
                report = ResearchReport.model_validate(event.data["report"])
                live.update(Status("[bold green]Done![/]"))

            elif event.type == "error":
                live.update(
                    Panel(
                        f"[red]{event.data.get('message', 'Unknown error')}[/]",
                        title="[red]Error[/]",
                    )
                )
                return

    if report is not None:
        md = report_to_markdown(report)
        console.print()
        console.print(Markdown(md))
        console.print()
        console.print(
            f"[dim]Sources: {len(report.sources)} | Cost: ${report.total_cost_usd:.4f}[/]"
        )
        if output_path is not None:
            output_path.write_text(md)
            console.print(f"\n[bold green]Report saved to[/] {output_path}")


@click.group()
def main():
    """DeepDive — AI deep research agent."""


@main.command()
@click.argument("question")
@click.option("--model", "-m", default=None, help="LLM model name (e.g. llama3.2)")
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["duckduckgo", "searxng"]),
    default=None,
    help="Search backend",
)
@click.option("--queries", "-q", default=None, type=int, help="Number of search queries")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Save the final report as a Markdown file.",
)
def research(question, model, backend, queries, output):
    """Research a question. Searches the web, analyzes sources, writes a cited report.

    Example: deepdive research "What caused the 2008 financial crisis?" -o report.md
    """
    config = DeepDiveConfig()
    if model:
        config.llm_model = model
    if backend:
        config.search_backend = backend
    if queries:
        config.queries_per_question = queries

    try:
        asyncio.run(_run_research(question, config, output))
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/]")
        sys.exit(1)


@main.command()
def serve():
    """Start the DeepDive API server."""
    from deepdive.api.main import run

    run()
