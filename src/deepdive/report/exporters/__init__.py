"""Multi-format report exporters.

Each exporter is a pure function: ``ResearchReport`` → ``str``. Pick the format
that matches your downstream tool — LaTeX for academic publishing, BibTeX for
reference managers, JSON for pipelines, Obsidian for personal knowledge bases,
Notion-flavored markdown for the Notion paste path.

The OpenAI dev forum has had "export Deep Research references to BibTeX" as
an open issue since early 2025. We make every format the default.
"""

from __future__ import annotations

from collections.abc import Callable

from deepdive.models import ResearchReport
from deepdive.report.exporters.bibtex import to_bibtex
from deepdive.report.exporters.json_export import to_json
from deepdive.report.exporters.latex import to_latex
from deepdive.report.exporters.notion import to_notion_markdown
from deepdive.report.exporters.obsidian import to_obsidian
from deepdive.report.markdown import report_to_markdown

# Registry of named exporters. Keys are user-facing names accepted by the CLI's
# --export flag; values are pure functions.
EXPORTERS: dict[str, Callable[[ResearchReport], str]] = {
    "markdown": report_to_markdown,
    "latex": to_latex,
    "bibtex": to_bibtex,
    "json": to_json,
    "obsidian": to_obsidian,
    "notion": to_notion_markdown,
}


def export(report: ResearchReport, format: str) -> str:  # noqa: A002 — matches docs idiom
    """Render ``report`` to the named format. Raises KeyError for unknown formats."""
    if format not in EXPORTERS:
        raise KeyError(f"Unknown export format {format!r}. Known: {sorted(EXPORTERS)}")
    return EXPORTERS[format](report)


def available_formats() -> list[str]:
    return sorted(EXPORTERS)


__all__ = [
    "EXPORTERS",
    "available_formats",
    "export",
    "to_bibtex",
    "to_json",
    "to_latex",
    "to_notion_markdown",
    "to_obsidian",
]
