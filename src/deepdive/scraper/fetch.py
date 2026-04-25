from __future__ import annotations

import asyncio

import httpx
import structlog
from selectolax.parser import HTMLParser

from deepdive import __version__
from deepdive.models import ScrapedPage

log = structlog.get_logger(__name__)

_USER_AGENT = f"DeepDive/{__version__} (+https://github.com/openintelligence-labs/deepdive)"
_MAX_BYTES = 2_000_000


class Scraper:
    def __init__(
        self,
        timeout: float = 15.0,
        max_concurrency: int = 5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self._sem = asyncio.Semaphore(max_concurrency)
        self._external = client is not None
        self._client = client or httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        )

    async def aclose(self) -> None:
        if not self._external:
            await self._client.aclose()

    async def fetch(self, url: str) -> ScrapedPage | None:
        async with self._sem:
            try:
                r = await self._client.get(url)
                r.raise_for_status()
            except Exception as exc:
                log.debug("scrape_failed", url=url, error=str(exc))
                return None
            content = r.content[:_MAX_BYTES]
            try:
                title, text = self._extract(content)
            except Exception as exc:
                log.debug("parse_failed", url=url, error=str(exc))
                return None
            return ScrapedPage(url=url, title=title, text=text)

    async def fetch_many(self, urls: list[str]) -> list[ScrapedPage]:
        results = await asyncio.gather(*(self.fetch(u) for u in urls))
        return [r for r in results if r is not None]

    @staticmethod
    def _extract(html: bytes) -> tuple[str, str]:
        tree = HTMLParser(html)
        title_node = tree.css_first("title")
        title = title_node.text(strip=True) if title_node else ""
        # Strip script/style/nav noise
        for selector in ("script", "style", "nav", "footer", "header", "aside"):
            for node in tree.css(selector):
                node.decompose()
        body = tree.body
        text = body.text(separator=" ", strip=True) if body else ""
        # Collapse whitespace
        text = " ".join(text.split())
        return title, text
