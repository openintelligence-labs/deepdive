from __future__ import annotations

import asyncio

import structlog
from ddgs import DDGS

from deepdive.models import SearchResult

log = structlog.get_logger(__name__)


def _ddg_text_blocking(query: str, max_results: int) -> list[dict]:
    """Synchronous DDGS call. Run on a worker thread from async code."""
    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


class DuckDuckGoClient:
    """Fallback search when SearxNG isn't available."""

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        # DDGS is synchronous; offloading to a thread keeps the event loop free.
        # Without this every concurrent search serializes behind the in-flight one.
        try:
            raw = await asyncio.to_thread(_ddg_text_blocking, query, max_results)
        except Exception as exc:
            log.warning("ddg_search_failed", query=query, error=str(exc))
            return []
        out: list[SearchResult] = []
        for item in raw:
            url = item.get("href") or item.get("link")
            title = item.get("title", "")
            snippet = item.get("body", "")
            if not url:
                continue
            try:
                out.append(SearchResult(url=url, title=title, snippet=snippet))
            except ValueError:
                continue
        return out
