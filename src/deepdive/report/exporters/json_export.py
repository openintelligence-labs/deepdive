"""JSON exporter — full ResearchReport model dump for downstream pipelines."""

from __future__ import annotations

import json

from deepdive.models import ResearchReport


def to_json(report: ResearchReport) -> str:
    """Render the full ResearchReport as pretty-printed JSON.

    Round-trips through ``ResearchReport.model_validate_json`` — the JSON
    contains every field including span-grounding offsets, citation excerpts,
    and per-section claims.
    """
    return json.dumps(report.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"
