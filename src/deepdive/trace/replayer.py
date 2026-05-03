"""Replay a recorded trace — produce a byte-identical report offline.

The replayer reads a .trace.jsonl, indexes recorded LLM calls by message-key
and search/scrape calls by their natural keys, and exposes wrapper providers
the pipeline can use exactly like the originals — except they NEVER hit the
network.
"""

from __future__ import annotations

import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolSpec,
    UsageDelta,
)

from deepdive.models import ScrapedPage, SearchResult
from deepdive.trace.recorder import key_messages


class ReplayMiss(LookupError):
    """Raised when the replayer cannot find a recorded answer for a live call."""


class _TraceIndex:
    """Indexed view of a trace file — keyed by message-key/url/query."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        # llm_call: a queue per key — handles multiple identical calls in order
        self.llm_calls: dict[str, list[CompletionResult]] = {}
        # search/scrape: same — dedup not safe (same query may run twice intentionally)
        self.searches: dict[str, list[list[SearchResult]]] = {}
        self.scrapes: dict[str, list[ScrapedPage | None]] = {}
        self.events: list[dict[str, Any]] = []
        # Lock guards the three queues against concurrent pop()s. The pipeline
        # runs sequentially today, but anyone parallelizing extraction across
        # pages would otherwise race on identical-key queues.
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                t = obj.get("type")
                if t == "llm_call":
                    key = obj["key"]
                    self.llm_calls.setdefault(key, []).append(
                        CompletionResult.model_validate(obj["result"])
                    )
                elif t == "search":
                    key = obj["key"]
                    results = [SearchResult.model_validate(r) for r in obj["results"]]
                    self.searches.setdefault(key, []).append(results)
                elif t == "scrape":
                    key = obj["key"]
                    page = (
                        ScrapedPage.model_validate(obj["page"]) if obj["page"] is not None else None
                    )
                    self.scrapes.setdefault(key, []).append(page)
                elif t == "event":
                    self.events.append(obj)
                # trace_start / trace_end are informational

    def pop_llm(self, key: str) -> CompletionResult:
        with self._lock:
            queue = self.llm_calls.get(key)
            if not queue:
                raise ReplayMiss(f"no recorded llm_call for key {key[:120]}…")
            return queue.pop(0)

    def pop_search(self, query: str, max_results: int) -> list[SearchResult]:
        key = json.dumps({"query": query, "max_results": max_results})
        with self._lock:
            queue = self.searches.get(key)
            if not queue:
                raise ReplayMiss(f"no recorded search for {key}")
            return queue.pop(0)

    def pop_scrape(self, url: str) -> ScrapedPage | None:
        with self._lock:
            queue = self.scrapes.get(url)
            if not queue:
                raise ReplayMiss(f"no recorded scrape for {url}")
            return queue.pop(0)


class _ReplayProvider(BaseLLMProvider):
    """LLM provider that never hits the network — answers from a TraceIndex."""

    def __init__(self, index: _TraceIndex) -> None:
        self._index = index
        self.name = "replay"
        # Both flags True so callers think the provider is fully capable.
        self.supports_tool_calls = True
        self.supports_streaming_tools = True

    async def complete(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        return self._index.pop_llm(key_messages(messages))

    async def stream(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        result = await self.complete(messages, model, temperature, max_tokens)
        if result.content:
            yield result.content

    async def stream_events(
        self,
        messages: list[ChatMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        *,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        from actants.llm.base import FinishDelta

        result = await self.complete(messages, model, temperature, max_tokens, tools=tools)
        if result.content:
            yield TextDelta(text=result.content)
        for tc in result.tool_calls:
            yield ToolCallDelta(tool_call=tc)
        yield UsageDelta(usage=result.usage, cost_usd=0.0)
        yield FinishDelta(reason="stop")

    async def health(self) -> bool:
        return True


def replay_provider(trace_path: str | Path) -> tuple[BaseLLMProvider, _TraceIndex]:
    """Build an offline LLM provider that answers from a trace file.

    Returns ``(provider, index)`` so the caller can also hand the index to
    ``ReplayingSearch`` and ``ReplayingScraper`` and share replay state.
    """
    index = _TraceIndex(trace_path)
    return _ReplayProvider(index), index


class ReplayingSearch:
    """Search backend that returns recorded results from a TraceIndex."""

    def __init__(self, index: _TraceIndex) -> None:
        self._index = index

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return self._index.pop_search(query, max_results)


class ReplayingScraper:
    """Scraper that returns recorded pages from a TraceIndex."""

    def __init__(self, index: _TraceIndex) -> None:
        self._index = index

    async def fetch(self, url: str) -> ScrapedPage | None:
        return self._index.pop_scrape(url)
