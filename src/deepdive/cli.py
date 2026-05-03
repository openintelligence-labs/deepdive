from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from actants import setup_logging
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status

from deepdive.config import DeepDiveConfig
from deepdive.models import ResearchReport
from deepdive.pipeline import ResearchPipeline
from deepdive.report.exporters import available_formats, export
from deepdive.report.exporters.bibtex import to_bibtex
from deepdive.report.markdown import report_to_markdown

console = Console()


async def _run_research(
    question: str,
    config: DeepDiveConfig,
    output_path: Path | None,
    trace_path: Path | None = None,
    source_filter=None,
    export_format: str = "markdown",
    plan_only: bool = False,
    force: bool = False,
) -> None:
    # Refuse to clobber an existing output file unless the user opted in with
    # --force. Cheap protection against losing a long-running report run.
    if output_path is not None and output_path.exists() and not force:
        console.print(
            f"[red]Refusing to overwrite existing file:[/] {output_path}\n"
            "[dim]Pass --force to overwrite, or pick a different -o path.[/]"
        )
        sys.exit(2)
    pipeline = ResearchPipeline(config=config, trace_path=trace_path, source_filter=source_filter)
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
                if plan_only:
                    # Stop iteration here — generators get GC'd cleanly.
                    return

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
        # Always render markdown for the on-screen preview (Rich understands it).
        md = report_to_markdown(report)
        console.print()
        console.print(Markdown(md))
        console.print()
        console.print(
            f"[dim]Sources: {len(report.sources)} | Cost: ${report.total_cost_usd:.4f}[/]"
        )
        if output_path is not None:
            try:
                payload = export(report, export_format)
            except KeyError as e:
                console.print(f"[red]{e}[/]")
                sys.exit(2)
            output_path.write_text(payload, encoding="utf-8")
            console.print(f"\n[bold green]Report saved to[/] {output_path}")
            # LaTeX needs a sibling .bib file to actually compile.
            if export_format == "latex":
                bib_path = output_path.with_suffix(".bib")
                # The exporter writes \bibliography{references} — match that stem.
                if bib_path.stem != "references":
                    bib_path = output_path.parent / "references.bib"
                bib_path.write_text(to_bibtex(report), encoding="utf-8")
                console.print(f"[bold green]Bibliography saved to[/] {bib_path}")


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
@click.option("--debug/--no-debug", default=False, help="Verbose logging.")
@click.option(
    "--log-format",
    type=click.Choice(["pretty", "json"]),
    default="pretty",
    help="Log rendering format.",
)
@click.option(
    "--ground/--no-ground",
    default=True,
    help="Require span-grounded citations (default: on). Drops claims whose "
    "supporting excerpt isn't found verbatim in the source.",
)
@click.option(
    "--include-ungrounded",
    is_flag=True,
    default=False,
    help="Even with --ground, include ungrounded claims in the report. For debugging.",
)
@click.option(
    "--trace",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Record an audit trace (.jsonl) of every LLM call, search, and scrape. "
    "Replay later with `deepdive replay <trace>` for byte-identical reproducibility.",
)
@click.option(
    "--allow-domains",
    default=None,
    help="Comma-separated allowlist (e.g. 'arxiv.org,nih.gov'). "
    "Restricts search to these hosts (matches subdomains).",
)
@click.option(
    "--block-domains",
    default=None,
    help="Comma-separated blocklist (e.g. 'medium.com,*.substack.com').",
)
@click.option(
    "--export",
    "export_format",
    type=click.Choice(available_formats()),
    default="markdown",
    help="Output format for the saved report. LaTeX also writes a sibling references.bib.",
)
@click.option(
    "--offline",
    is_flag=True,
    default=False,
    help="Strict offline mode: every outbound destination must be loopback. "
    "Cloud LLM endpoints raise; non-loopback URLs are dropped from the scraper.",
)
@click.option(
    "--corpus",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a sqlite-vec corpus (built with `deepdive index`). "
    "When set, search runs against the local corpus instead of the web.",
)
@click.option(
    "--plan-only",
    is_flag=True,
    default=False,
    help="Print the generated search-query plan and exit without running.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite the -o output file if it already exists.",
)
def research(
    question,
    model,
    backend,
    queries,
    output,
    debug,
    log_format,
    ground,
    include_ungrounded,
    trace,
    allow_domains,
    block_domains,
    export_format,
    offline,
    corpus,
    plan_only,
    force,
):
    """Research a question. Searches the web, analyzes sources, writes a cited report.

    Example: deepdive research "What caused the 2008 financial crisis?" -o report.md
    """
    setup_logging(level="debug" if debug else "info", format=log_format)
    config = DeepDiveConfig()
    config.ground_citations = ground
    config.include_ungrounded = include_ungrounded
    config.offline = offline
    if corpus is not None:
        config.corpus_path = str(corpus)
        config.search_backend = "corpus"
    if model:
        config.llm_model = model
    if backend:
        config.search_backend = backend
    if queries:
        config.queries_per_question = queries

    # If user gave -o report.md but no --trace, default the trace to a sibling
    # file so the audit trail is always available.
    trace_path = trace
    if trace_path is None and output is not None:
        trace_path = Path(str(output) + ".trace.jsonl")

    # Build the source filter from --allow-domains / --block-domains.
    source_filter = _build_source_filter(allow_domains, block_domains)

    try:
        asyncio.run(
            _run_research(
                question,
                config,
                output,
                trace_path,
                source_filter,
                export_format,
                plan_only,
                force,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Research interrupted.[/]")
        sys.exit(1)


def _build_source_filter(allow_domains, block_domains):
    """Compose a SourceFilter from CLI flags. Returns None if nothing was set."""
    from deepdive.search.filters import SourceFilter

    if not allow_domains and not block_domains:
        return None

    allow = tuple(d.strip() for d in (allow_domains or "").split(",") if d.strip())
    block = tuple(d.strip() for d in (block_domains or "").split(",") if d.strip())
    return SourceFilter(allow=allow, block=block)


@main.command()
def serve():
    """Start the DeepDive API server."""
    from deepdive.api.main import run

    run()


async def _run_index(root: Path, output: Path, embed_model: str) -> None:
    from deepdive.corpus.indexer import CorpusIndex

    async with CorpusIndex(output, embed_model=embed_model) as index:
        if root.is_file():
            added = await index.index_file(root)
            console.print(f"  [green]{root}[/] [dim]→[/] {added} chunks")
        else:
            async for path, added in index.index_directory(root):
                marker = "✓" if added > 0 else "↺"
                rel = path.relative_to(root.resolve())
                console.print(f"  [green]{marker}[/] {rel} [dim]→[/] {added} chunks")
        stats = index.stats()
    console.print(
        f"\n[bold]Index:[/] {output}  [dim]({stats['documents']} documents, "
        f"{stats['chunks']} chunks)[/]"
    )


@main.command(name="index")
@click.argument("root", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to write the sqlite-vec index (e.g. ~/.local/share/deepdive/index.db).",
)
@click.option(
    "--embed-model",
    default="nomic-embed-text",
    help="Ollama embedding model. nomic-embed-text is the default.",
)
def index_corpus(root: Path, output: Path, embed_model: str):
    """Index a directory or file into a sqlite-vec corpus.

    Supported formats: PDF, Markdown, HTML, plain text. Re-indexing is
    incremental — files unchanged since their last index are skipped.

    Example: deepdive index ~/Documents/papers -o ~/.local/share/deepdive/index.db
    """
    asyncio.run(_run_index(root, output, embed_model))


async def _run_replay(trace_path: Path, output: Path | None, force: bool = False) -> None:
    if output is not None and output.exists() and not force:
        console.print(
            f"[red]Refusing to overwrite existing file:[/] {output}\n"
            "[dim]Pass --force to overwrite.[/]"
        )
        sys.exit(2)
    # Read the original question + config-relevant info from the trace's first event.
    import json as _json

    from actants import LLM

    from deepdive.trace import replay_provider
    from deepdive.trace.replayer import ReplayingScraper, ReplayingSearch

    question = ""
    saved_config: dict = {}
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _json.loads(line)
            if obj.get("type") == "event":
                etype = obj.get("event_type")
                if etype == "start" and not question:
                    question = obj["data"].get("question", "")
                elif etype == "config" and not saved_config:
                    saved_config = obj["data"]
            if question and saved_config:
                break
    if not question:
        console.print("[red]Could not find a 'start' event in the trace.[/]")
        sys.exit(2)

    provider, index = replay_provider(trace_path)
    config = DeepDiveConfig()
    # Restore prompt-affecting fields from the recording so prompts match.
    for k, v in saved_config.items():
        if hasattr(config, k):
            setattr(config, k, v)
    pipeline = ResearchPipeline(
        config=config,
        llm=LLM(provider=provider, model="replay"),
        search_client=ReplayingSearch(index),  # type: ignore[arg-type]
        scraper=ReplayingScraper(index),  # type: ignore[arg-type]
        # Match the recording's grounding mode — recorded LLM calls must hit
        # the same prompt path (grounded vs legacy) as the original run.
        ground=saved_config.get("ground_citations", True),
    )
    report: ResearchReport | None = None
    async for event in pipeline.run(question):
        if event.type == "done":
            report = ResearchReport.model_validate(event.data["report"])
        elif event.type == "error":
            console.print(f"[red]error during replay:[/] {event.data}")

    if report is None:
        console.print("[red]Replay produced no report.[/]")
        sys.exit(2)

    md = report_to_markdown(report)
    if output is not None:
        output.write_text(md, encoding="utf-8")
        console.print(f"[green]Replayed report written to {output}[/]")
    else:
        console.print(Markdown(md))


@main.command()
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write replayed Markdown report here.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite the -o output file if it already exists.",
)
def replay(trace_path: Path, output: Path | None, force: bool):
    """Replay a recorded trace offline. Produces a byte-identical report.

    Example: deepdive replay report.md.trace.jsonl -o replayed.md
    """
    asyncio.run(_run_replay(trace_path, output, force))


@main.command(name="inspect")
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def inspect_trace(trace_path: Path):
    """Pretty-print a trace file: counts of LLM calls, searches, scrapes, events."""
    import json as _json

    counts: dict[str, int] = {}
    sample_event_types: list[str] = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _json.loads(line)
            t = obj.get("type", "?")
            counts[t] = counts.get(t, 0) + 1
            if t == "event":
                sample_event_types.append(obj.get("event_type", "?"))

    console.print(f"[bold]Trace:[/] {trace_path}")
    for k in sorted(counts):
        console.print(f"  {k}: {counts[k]}")
    if sample_event_types:
        console.print(
            f"  event types ({len(sample_event_types)}): "
            f"{', '.join(dict.fromkeys(sample_event_types))}"
        )


@main.group(name="trace")
def trace_group():
    """Trace-file utilities."""


@trace_group.command(name="verify")
@click.argument("trace_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def trace_verify(trace_path: Path):
    """Re-validate every recorded claim's excerpt against the recorded source HTML.

    Reports drift — useful before publishing reports months after the original run
    or after upgrading the model. Pure offline; no network calls.
    """
    import json as _json

    from deepdive.analysis.grounding import find_excerpt_offsets

    pages: dict[str, str] = {}
    claim_excerpts: list[tuple[str, str]] = []  # (url, excerpt)

    with trace_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            obj = _json.loads(line)
            t = obj.get("type")
            if t == "scrape" and obj.get("page"):
                pages[obj["key"]] = obj["page"].get("text", "")
            elif t == "event" and obj.get("event_type") == "done":
                report = obj["data"].get("report", {})
                for section in report.get("sections", []):
                    for claim in section.get("claims", []):
                        for cit in claim.get("citations", []):
                            url = cit.get("url")
                            excerpt = cit.get("excerpt")
                            if url and excerpt:
                                claim_excerpts.append((url, excerpt))

    total = len(claim_excerpts)
    matched = 0
    drift: list[tuple[str, str]] = []
    for url, excerpt in claim_excerpts:
        body = pages.get(url, "")
        if body and find_excerpt_offsets(body, excerpt) is not None:
            matched += 1
        else:
            drift.append((url, excerpt))

    console.print(f"[bold]Verified:[/] {matched}/{total} claim excerpts found in source")
    if drift:
        console.print(f"[yellow]Drift on {len(drift)} claims:[/]")
        for url, excerpt in drift[:5]:
            short = excerpt[:80] + ("…" if len(excerpt) > 80 else "")
            console.print(f"  [dim]{url}[/]\n    > {short}")
        if len(drift) > 5:
            console.print(f"  [dim]…and {len(drift) - 5} more[/]")
        sys.exit(1)


@main.command(name="serve-mcp")
@click.option(
    "--http",
    "use_http",
    is_flag=True,
    default=False,
    help="Serve over Streamable HTTP instead of stdio. Used by remote MCP clients.",
)
@click.option("--host", default="127.0.0.1", help="HTTP host (only with --http).")
@click.option("--port", default=8765, type=int, help="HTTP port (only with --http).")
def serve_mcp_cmd(use_http: bool, host: str, port: int):
    """Run DeepDive as an MCP server.

    Default transport is stdio (Claude Desktop spawns this directly). Pass --http
    for IDE plugins or remote agents.

    Example (Claude Desktop): add to claude_desktop_config.json::

        {"mcpServers": {"deepdive": {"command": "deepdive", "args": ["serve-mcp"]}}}
    """
    from deepdive.mcp import serve_mcp

    transport = "streamable-http" if use_http else "stdio"
    serve_mcp(transport=transport, host=host, port=port)
