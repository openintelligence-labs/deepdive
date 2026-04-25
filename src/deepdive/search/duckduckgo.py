from __future__ import annotations

import structlog
from ddgs import DDGS

from deepdive.models import SearchResult

log = structlog.get_logger(__name__)


class DuckDuckGoClient:
    """Fallback search when SearxNG isn't available."""

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
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
