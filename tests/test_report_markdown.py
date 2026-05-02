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


def _grounded_report() -> ResearchReport:
    """Two claims: one grounded, one not."""
    cit_grounded = Citation(
        url="https://x.com/a", title="A",
        excerpt="example servers",
        offset_start=0, offset_end=15, grounded=True,
    )
    cit_ungrounded = Citation(
        url="https://x.com/b", title="B",
        excerpt="invented quote",
        grounded=False,
    )
    return ResearchReport(
        question="Q?",
        summary="s",
        sections=[
            ReportSection(
                heading="Findings",
                body="body",
                claims=[
                    Claim(text="grounded claim", citations=[cit_grounded]),
                    Claim(text="ungrounded claim", citations=[cit_ungrounded]),
                ],
            )
        ],
        sources=[cit_grounded, cit_ungrounded],
        generated_at=datetime(2026, 4, 29, tzinfo=UTC),
    )


def test_markdown_evidence_section_lists_grounding_status():
    md = report_to_markdown(_grounded_report())
    assert "## Evidence" in md
    assert "1/2 claims are span-grounded" in md
    # Grounded claim shows ✓
    assert "[✓]" in md
    # Ungrounded claim shows ✗
    assert "[✗]" in md
    # Excerpts surface as blockquotes
    assert "> example servers" in md
    assert "> invented quote" in md


def test_markdown_footer_shows_grounding_count():
    md = report_to_markdown(_grounded_report())
    assert "1/2 grounded" in md


def test_markdown_evidence_can_be_disabled():
    md = report_to_markdown(_grounded_report(), include_evidence=False)
    assert "## Evidence" not in md


def test_markdown_truncates_long_excerpts():
    long_excerpt = "x" * 500
    cit = Citation(
        url="https://x.com/a", title="A",
        excerpt=long_excerpt, offset_start=0, offset_end=500, grounded=True,
    )
    report = ResearchReport(
        question="Q",
        summary="s",
        sections=[
            ReportSection(
                heading="h", body="b",
                claims=[Claim(text="claim", citations=[cit])],
            )
        ],
    )
    md = report_to_markdown(report)
    # Truncated to ~240 chars + ellipsis, not the full 500
    assert "…" in md
    # Original full excerpt should NOT appear in full
    assert long_excerpt not in md
