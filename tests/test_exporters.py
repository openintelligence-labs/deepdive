"""Multi-format exporter tests."""

from __future__ import annotations

import json as _json
from datetime import UTC, datetime

import pytest

from deepdive.models import Citation, Claim, ReportSection, ResearchReport
from deepdive.report.exporters import EXPORTERS, available_formats, export
from deepdive.report.exporters.latex import cite_key, latex_escape, to_latex


def _fake_report() -> ResearchReport:
    cit = Citation(
        url="https://arxiv.org/abs/2305.10403",
        title="Example arxiv paper",
        excerpt="An example excerpt.",
        offset_start=0,
        offset_end=18,
        grounded=True,
    )
    cit2 = Citation(
        url="https://www.nature.com/articles/x",
        title="Nature article",
        excerpt="Another excerpt.",
        offset_start=0,
        offset_end=16,
        grounded=True,
    )
    section = ReportSection(
        heading="Background",
        body="Background body with $math$ and 50% _underscores_ & ampersands.",
        claims=[Claim(text="C1", citations=[cit])],
    )
    section2 = ReportSection(
        heading="Findings",
        body="Findings body.",
        claims=[Claim(text="C2", citations=[cit2])],
    )
    return ResearchReport(
        question="Test question?",
        summary="Short summary.",
        sections=[section, section2],
        sources=[cit, cit2],
        generated_at=datetime(2026, 4, 30, tzinfo=UTC),
        total_cost_usd=0.0,
    )


# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────


def test_six_formats_are_registered():
    assert set(available_formats()) == {"markdown", "latex", "bibtex", "json", "obsidian", "notion"}


def test_export_unknown_format_raises():
    with pytest.raises(KeyError, match="Unknown export format"):
        export(_fake_report(), "fortran")


def test_every_exporter_returns_str():
    r = _fake_report()
    for name in available_formats():
        out = EXPORTERS[name](r)
        assert isinstance(out, str)
        assert len(out) > 0


# ──────────────────────────────────────────────────────────────────────
# LaTeX
# ──────────────────────────────────────────────────────────────────────


def test_latex_escapes_special_characters():
    assert latex_escape("50%") == r"50\%"
    assert latex_escape("a_b") == r"a\_b"
    assert latex_escape("$math$") == r"\$math\$"
    assert latex_escape("a&b") == r"a\&b"


def test_latex_output_is_complete_document():
    out = to_latex(_fake_report())
    assert r"\documentclass" in out
    assert r"\begin{document}" in out
    assert r"\end{document}" in out
    assert r"\title{Test question?}" in out
    assert r"\maketitle" in out
    assert r"\bibliography{references}" in out


def test_latex_escapes_body_content():
    out = to_latex(_fake_report())
    # The original body has %, _, $, & — all should be escaped
    assert r"\%" in out
    assert r"\_" in out
    assert r"\$" in out
    assert r"\&" in out
    # And NOT raw
    assert "50%" not in out
    assert "_underscores_" not in out


def test_latex_includes_cite_calls_for_sections_with_claims():
    out = to_latex(_fake_report())
    # The fake report has 2 cited sources; both keys should appear
    keys = {cite_key(s) for s in _fake_report().sources}
    for k in keys:
        assert r"\cite{" + k + "}" in out


# ──────────────────────────────────────────────────────────────────────
# BibTeX
# ──────────────────────────────────────────────────────────────────────


def test_bibtex_emits_one_entry_per_unique_source():
    out = EXPORTERS["bibtex"](_fake_report())
    assert out.count("@misc{") == 2  # two unique sources


def test_bibtex_keys_match_latex_keys():
    """The .bib must use the same keys as the .tex \\cite{} calls."""
    report = _fake_report()
    bib = EXPORTERS["bibtex"](report)
    tex = to_latex(report)
    for src in report.sources:
        key = cite_key(src)
        assert f"@misc{{{key}," in bib
        assert r"\cite{" + key + "}" in tex


def test_bibtex_includes_url_and_year():
    out = EXPORTERS["bibtex"](_fake_report())
    assert r"\url{https://arxiv.org/abs/2305.10403}" in out
    assert "year = {2026}" in out


def test_bibtex_dedupes_identical_sources():
    cit = Citation(url="https://arxiv.org/abs/x", title="X")
    report = ResearchReport(
        question="Q",
        summary="s",
        sections=[ReportSection(heading="h", body="b", claims=[Claim(text="c", citations=[cit])])],
        sources=[cit, cit],  # duplicated on purpose
    )
    out = EXPORTERS["bibtex"](report)
    assert out.count("@misc{") == 1


def test_cite_key_is_stable_for_same_url():
    c1 = Citation(url="https://arxiv.org/abs/x", title="X")
    c2 = Citation(url="https://arxiv.org/abs/x", title="Different title")
    assert cite_key(c1) == cite_key(c2)


def test_cite_key_differs_for_different_urls():
    c1 = Citation(url="https://arxiv.org/abs/x", title="X")
    c2 = Citation(url="https://arxiv.org/abs/y", title="X")
    assert cite_key(c1) != cite_key(c2)


# ──────────────────────────────────────────────────────────────────────
# JSON
# ──────────────────────────────────────────────────────────────────────


def test_json_roundtrips_through_pydantic():
    out = EXPORTERS["json"](_fake_report())
    parsed = _json.loads(out)
    restored = ResearchReport.model_validate(parsed)
    assert restored.question == "Test question?"
    assert len(restored.sections) == 2
    # Span-grounding preserved
    assert restored.sections[0].claims[0].citations[0].grounded is True
    assert restored.sections[0].claims[0].citations[0].excerpt == "An example excerpt."


# ──────────────────────────────────────────────────────────────────────
# Obsidian
# ──────────────────────────────────────────────────────────────────────


def test_obsidian_includes_yaml_frontmatter():
    out = EXPORTERS["obsidian"](_fake_report())
    assert out.startswith("---\n")
    assert "title: Test question?" in out
    assert "tags: [research, deepdive]" in out


def test_obsidian_uses_wikilinks_for_sources():
    out = EXPORTERS["obsidian"](_fake_report())
    # Each source becomes a [[wikilink]] anchor
    assert "[[Example arxiv paper]]" in out
    assert "[[Nature article]]" in out


def test_obsidian_reports_grounding_count():
    out = EXPORTERS["obsidian"](_fake_report())
    assert "grounded: 2/2" in out


# ──────────────────────────────────────────────────────────────────────
# Notion
# ──────────────────────────────────────────────────────────────────────


def test_notion_uses_callout_for_summary():
    out = EXPORTERS["notion"](_fake_report())
    # Notion treats `> emoji ...` as a callout block
    assert "> 📝" in out


def test_notion_includes_grounding_footer():
    out = EXPORTERS["notion"](_fake_report())
    assert "🔍" in out
    assert "2/2 claims" in out


def test_notion_lists_sources_as_links():
    out = EXPORTERS["notion"](_fake_report())
    assert "[Example arxiv paper](https://arxiv.org/abs/2305.10403)" in out
