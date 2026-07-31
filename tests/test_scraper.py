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


@pytest.mark.asyncio
async def test_fetch_rejects_loopback_url_by_default():
    s = Scraper()
    page = await s.fetch("http://127.0.0.1:11434/v1/chat")
    assert page is None
    await s.aclose()


@pytest.mark.asyncio
async def test_fetch_blocks_redirect_to_loopback():
    redirect_target = "http://127.0.0.1:8080/admin"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": redirect_target})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        s = Scraper(client=c, allow_private_hosts=False)
        # example.com is public and clears the guard; the 302 hop is the check.
        page = await s.fetch("https://example.com/jump")
        assert page is None


@pytest.mark.asyncio
async def test_fetch_many_survives_one_exception():
    """One URL raising mid-batch must not poison the gather."""

    async def handler(request: httpx.Request) -> httpx.Response:
        if "boom" in str(request.url):
            raise httpx.ConnectError("simulated boom")
        return httpx.Response(200, content=b"<html><body>ok</body></html>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        s = Scraper(client=c)
        pages = await s.fetch_many(
            ["https://example.com/a", "https://example.com/boom", "https://example.com/b"]
        )
        # 'a' and 'b' succeed; 'boom' raises and is filtered by return_exceptions.
        assert len(pages) == 2
