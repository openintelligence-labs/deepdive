"""BibTeX exporter — every cited source as a @misc entry.

Citation keys match those produced by ``latex.cite_key`` so a .tex + .bib pair
generated from the same report cross-reference cleanly.
"""

from __future__ import annotations

from deepdive.models import ResearchReport
from deepdive.report.exporters.latex import cite_key

_BIBTEX_ESCAPE = {
    "{": r"\{",
    "}": r"\}",
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
}


def _bibtex_escape(s: str) -> str:
    out = []
    for ch in s:
        out.append(_BIBTEX_ESCAPE.get(ch, ch))
    return "".join(out)


def to_bibtex(report: ResearchReport) -> str:
    """Render the report's sources as BibTeX @misc entries."""
    lines: list[str] = []
    seen: set[str] = set()
    year = report.generated_at.strftime("%Y")
    accessed = report.generated_at.strftime("%Y-%m-%d")

    for src in report.sources:
        key = cite_key(src)
        if key in seen:
            continue
        seen.add(key)
        title = _bibtex_escape((src.title or "Untitled").strip())
        url = str(src.url)
        lines.append(f"@misc{{{key},")
        lines.append(f"  title = {{{title}}},")
        lines.append(f"  howpublished = {{\\url{{{url}}}}},")
        lines.append(f"  year = {{{year}}},")
        lines.append(f"  note = {{Accessed: {accessed}}}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)
