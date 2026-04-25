from __future__ import annotations

import httpx
import pytest

from deepdive.scraper.fetch import Scraper


@pytest.mark.asyncio
async def test_fetch_extracts_title_and_text():
    html = b"""
    <html>
      <head><title>Hello World</title></head>
      <body>
        <script>var x = 1;</script>
        <nav>junk nav</nav>
        <main><p>Main content here.</p><p>Second paragraph.</p></main>
        <footer>footer junk</footer>
      </body>
    </html>
    """

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=html, headers={"content-type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        s = Scraper(client=c)
        page = await s.fetch("https://example.com/a")
        assert page is not None
        assert page.title == "Hello World"
        assert "Main content here" in page.text
        assert "junk nav" not in page.text
        assert "var x" not in page.text


@pytest.mark.asyncio
async def test_fetch_handles_http_error():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        s = Scraper(client=c)
        page = await s.fetch("https://example.com/a")
        assert page is None


@pytest.mark.asyncio
async def test_fetch_many_filters_failures():
    urls_seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        urls_seen.append(str(request.url))
        if "bad" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=b"<html><body>ok</body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        s = Scraper(client=c)
        pages = await s.fetch_many(
            [
                "https://example.com/good",
                "https://example.com/bad",
                "https://example.com/also-good",
            ]
        )
        assert len(pages) == 2
