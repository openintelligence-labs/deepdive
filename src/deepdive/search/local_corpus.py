"""Search backend that reads from a local corpus index.

Implements the same ``async search(query, max_results)`` interface as
``DuckDuckGoClient`` and ``SearxNGClient`` so it drops into the pipeline
unchanged. Each chunk becomes a SearchResult with a ``deepdive-corpus://``
URL and the chunk text as the snippet — the scraper layer recognizes this
scheme and returns the chunk verbatim instead of fetching the network.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from urllib.parse import quote

from deepdive.corpus.indexer import CorpusIndex
from deepdive.models import ScrapedPage, SearchResult


def _corpus_url(source_path: str, offset: int) -> str:
    """Stable, parseable URL for an in-corpus chunk."""
    return f"http://localhost/corpus/{quote(source_path, safe='')}#offset={offset}"


class LocalCorpusClient:
    """Cosine-similarity search over a sqlite-vec corpus."""

    def __init__(self, index: CorpusIndex) -> None:
        self.index = index

    async def search(self, query: str, *, max_results: int = 5) -> list[SearchResult]:
        hits = await self.index.search(query, k=max_results)
        out: list[SearchResult] = []
        for h in hits:
            url = _corpus_url(h.source_path, h.offset_start)
            title = Path(h.source_path).name
            snippet = h.text[:300]
            try:
                out.append(
                    SearchResult(url=url, title=title, snippet=snippet, source="corpus")
                )
            except Exception:
                continue
        return out


class LocalCorpusScraper:
    """Scraper that resolves ``deepdive-corpus://`` URLs to ScrapedPage objects.

    For non-corpus URLs, delegates to the inner scraper (or returns None when
    operating in strict offline mode and no inner scraper was provided).
    """

    def __init__(
        self,
        index: CorpusIndex,
        inner=None,
        *,
        offline: bool = False,
    ) -> None:
        self.index = index
        self.inner = inner
        self.offline = offline

    async def fetch(self, url: str) -> ScrapedPage | None:
        if "/corpus/" in url and "localhost" in url:
            # Re-parse to find the chunk by its source_path + offset.
            return await self._fetch_corpus(url)
        if self.offline:
            return None
        if self.inner is not None:
            return await self.inner.fetch(url)
        return None

    async def _fetch_corpus(self, url: str) -> ScrapedPage | None:
        from urllib.parse import unquote, urlparse

        parsed = urlparse(url)
        # Path format: /corpus/<encoded source_path>
        if not parsed.path.startswith("/corpus/"):
            return None
        source_path = unquote(parsed.path[len("/corpus/"):])
        # Offset is in the fragment as `offset=N`
        offset = 0
        if parsed.fragment.startswith("offset="):
            with contextlib.suppress(ValueError):
                offset = int(parsed.fragment.split("=", 1)[1])
        # Look the chunk up directly so we get exactly the indexed text.
        cur = self.index.conn.execute(
            "SELECT c.text FROM chunks c "
            "JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE d.source_path = ? AND c.offset_start = ?",
            (source_path, offset),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return ScrapedPage(
            url=url,
            title=Path(source_path).name,
            text=row[0],
        )
