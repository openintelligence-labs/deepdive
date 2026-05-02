"""Trace recording + replay for reproducible research runs.

A trace file (.trace.jsonl) captures every non-deterministic input the pipeline
saw — LLM responses, search results, fetched pages — so the same research run
can be re-played offline to produce a byte-identical report.

This is the reproducibility wedge from docs/DIFFERENTIATION.md. Compliance,
audit, peer-review, and "did the model lie?" workflows all need this.
"""

from __future__ import annotations

from deepdive.trace.recorder import (
    RecordingScraper,
    RecordingSearch,
    TraceRecorder,
    record_provider,
)
from deepdive.trace.replayer import ReplayingScraper, ReplayingSearch, replay_provider

__all__ = [
    "RecordingScraper",
    "RecordingSearch",
    "ReplayingScraper",
    "ReplayingSearch",
    "TraceRecorder",
    "record_provider",
    "replay_provider",
]
