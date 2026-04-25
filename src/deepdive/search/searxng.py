from __future__ import annotations

import httpx
import structlog

from deepdive.models import SearchResult

log = structlog.get_logger(__name__)


class SearxNGClient:
    """Thin async client for a self-hosted SearxNG instance."""

    def __init__(
        self,
        base_url: str = "http://localhost:8888",
        timeout: float = 10.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._external = client is not None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        params = {"q": query, "format": "json", "safesearch": 1}
        r = await self._client.get(f"{self.base_url}/search", params=params)
        r.raise_for_status()
        data = r.json()
        out: list[SearchResult] = []
        for item in data.get("results", [])[:max_results]:
            url = item.get("url")
            title = item.get("title") or ""
            snippet = item.get("content") or ""
            score = float(item.get("score") or 0.0)
            if not url:
                continue
            try:
                out.append(SearchResult(url=url, title=title, snippet=snippet, score=score))
            except ValueError:
                log.debug("skipping_bad_result", url=url)
                continue
        return out
