from __future__ import annotations

import httpx
import pytest

from deepdive.search.searxng import SearxNGClient


@pytest.mark.asyncio
async def test_search_parses_results():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["q"] == "quantum computing"
        assert request.url.params["format"] == "json"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Quantum A",
                        "content": "snippet a",
                        "score": 0.9,
                    },
                    {
                        "url": "https://example.com/b",
                        "title": "Quantum B",
                        "content": "snippet b",
                        "score": 0.7,
                    },
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        client = SearxNGClient(client=c)
        results = await client.search("quantum computing")
        assert len(results) == 2
        assert str(results[0].url) == "https://example.com/a"
        assert results[0].title == "Quantum A"
        assert results[0].score == 0.9


@pytest.mark.asyncio
async def test_search_skips_bad_urls():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": None, "title": "bad", "content": ""},
                    {"url": "not-a-url", "title": "bad2", "content": ""},
                    {"url": "https://good.com/x", "title": "good", "content": "ok"},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        client = SearxNGClient(client=c)
        results = await client.search("q")
        assert len(results) == 1
        assert str(results[0].url) == "https://good.com/x"


@pytest.mark.asyncio
async def test_search_respects_max_results():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {"url": f"https://example.com/{i}", "title": str(i), "content": ""}
                    for i in range(20)
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        client = SearxNGClient(client=c)
        results = await client.search("q", max_results=3)
        assert len(results) == 3
