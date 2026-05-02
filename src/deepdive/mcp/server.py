"""Build a ``ToolRegistry`` exposing the DeepDive research pipeline as an MCP tool."""

from __future__ import annotations

from typing import Literal

from agentic_kit.tools.registry import ToolRegistry

from deepdive.config import DeepDiveConfig
from deepdive.models import ResearchReport
from deepdive.pipeline import ResearchPipeline
from deepdive.report.exporters import EXPORTERS
from deepdive.search.filters import SourceFilter


async def _research_tool(
    question: str,
    allow_domains: list[str] | None = None,
    block_domains: list[str] | None = None,
    max_pages: int = 5,
    export_format: str = "markdown",
) -> str:
    """Run a DeepDive research pipeline and return the rendered report.

    This is the single MCP tool the server exposes. Returns the report as a
    string in the requested format (default markdown). Errors during the run
    surface as a one-line string so the calling agent can decide what to do.
    """
    config = DeepDiveConfig()
    config.max_pages_per_research = max_pages
    config.queries_per_question = min(3, max_pages)

    source_filter = None
    if allow_domains or block_domains:
        source_filter = SourceFilter(
            allow=tuple(allow_domains or ()),
            block=tuple(block_domains or ()),
        )

    pipeline = ResearchPipeline(config=config, source_filter=source_filter)
    report: ResearchReport | None = None
    error_msg: str | None = None
    async for event in pipeline.run(question):
        if event.type == "done":
            report = ResearchReport.model_validate(event.data["report"])
        elif event.type == "error":
            error_msg = f"{event.data.get('stage', '?')}: {event.data.get('message', '?')}"
            break

    if report is None:
        return f"error: research failed ({error_msg or 'no report produced'})"

    fn = EXPORTERS.get(export_format)
    if fn is None:
        return f"error: unknown export format {export_format!r}"
    return fn(report)


def build_research_registry() -> ToolRegistry:
    """Build a ToolRegistry with the single ``research`` tool registered."""
    registry = ToolRegistry()
    registry.register_function(
        "research",
        "Run a deep-research pipeline on a question. Returns a cited report.",
        _research_tool,
        input_schema={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The research question to investigate.",
                },
                "allow_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional whitelist of hostnames "
                        "(e.g. ['arxiv.org', 'nih.gov']). "
                        "Subdomains match. Wildcards like '*.gov' supported."
                    ),
                },
                "block_domains": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional blocklist of hostnames "
                        "(e.g. ['medium.com', '*.substack.com'])."
                    ),
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Cap on pages scraped (default 5).",
                },
                "export_format": {
                    "type": "string",
                    "description": (
                        "Output format: markdown (default), latex, bibtex, "
                        "json, obsidian, notion."
                    ),
                },
            },
            "required": ["question"],
        },
    )
    return registry


def serve_mcp(
    *,
    transport: Literal["stdio", "streamable-http"] = "stdio",
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Run DeepDive as an MCP server. Blocks until killed.

    Default transport is ``stdio`` so Claude Desktop can spawn this process
    directly. Use ``streamable-http`` for IDE plugins or remote callers.
    """
    from agentic_kit.mcp import serve

    serve(
        build_research_registry(),
        transport=transport,
        host=host,
        port=port,
        name="deepdive",
        instructions=(
            "DeepDive is a local-first research agent. Call `research(question)` to "
            "get a cited Markdown report. Optional `allow_domains` / `block_domains` "
            "restrict the search. Optional `export_format` switches the output "
            "(markdown, latex, bibtex, json, obsidian, notion)."
        ),
    )
