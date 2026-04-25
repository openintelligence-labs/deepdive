from __future__ import annotations

from datetime import UTC, datetime

from deepdive.models import (
    Citation,
    Claim,
    ReportSection,
    ResearchReport,
)
from deepdive.report.markdown import report_to_markdown


def _fake_report() -> ResearchReport:
    citation = Citation(url="https://example.com/a", title="Example A")
    claim = Claim(text="Example has servers", citations=[citation])
    section = ReportSection(
        heading="Key findings",
        body="Example.com operates servers [1].",
        claims=[claim],
    )
    return ResearchReport(
        question="What is example.com?",
        summary="Example.com is a demonstration domain.",
        sections=[section],
        sources=[citation],
        generated_at=datetime(2026, 4, 18, tzinfo=UTC),
        total_cost_usd=0.01234,
    )


def test_markdown_contains_question_and_summary():
    md = report_to_markdown(_fake_report())
    assert "# What is example.com?" in md
    assert "Example.com is a demonstration domain." in md


def test_markdown_includes_sections_with_headings():
    md = report_to_markdown(_fake_report())
    assert "## Key findings" in md
    assert "Example.com operates servers [1]." in md


def test_markdown_lists_sources_with_links():
    md = report_to_markdown(_fake_report())
    assert "## Sources" in md
    assert "[Example A](https://example.com/a)" in md


def test_markdown_includes_generation_metadata():
    md = report_to_markdown(_fake_report())
    assert "2026-04-18" in md
    assert "cost $0.0123" in md


def test_markdown_skips_sources_section_when_empty():
    report = _fake_report()
    report.sources = []
    md = report_to_markdown(report)
    assert "## Sources" not in md
