from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
)
from actants.llm.client import LLM

from deepdive.analysis.claims import ClaimExtractor, _parse_lines, cross_reference
from deepdive.models import Citation, Claim, ScrapedPage


def _claim(text: str, url: str) -> Claim:
    return Claim(text=text, citations=[Citation(url=url, title="t")], confidence=0.5)


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        content = self._responses.pop(0)
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(
        self, messages: list[ChatMessage], model, temperature=0.7, max_tokens=None, **kwargs
    ) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_claim_extractor_parses_structured_output():
    provider = ScriptedProvider(['{"claims": ["Fact A", "Fact B"]}'])
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        max_claims=5,
        ground=False,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text="some content")
    claims = await extractor.extract(page)
    assert [c.text for c in claims] == ["Fact A", "Fact B"]
    assert all(len(c.citations) == 1 for c in claims)


@pytest.mark.asyncio
async def test_claim_extractor_returns_empty_on_empty_text():
    provider = ScriptedProvider([])
    extractor = ClaimExtractor(llm=LLM(provider=provider, model="test", tracing=False))
    page = ScrapedPage(url="https://example.com/", title="t", text="")
    assert await extractor.extract(page) == []


@pytest.mark.asyncio
async def test_claim_extractor_returns_empty_on_parse_failure():
    # All attempts return garbage (no list markers) → empty claim list.
    # extract() calls LLM.extract internally which may retry, then falls back
    # to LLM.complete() once more. Four slots is plenty.
    provider = ScriptedProvider(["garbage"] * 6)
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        ground=False,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text="real text")
    assert await extractor.extract(page) == []


@pytest.mark.asyncio
async def test_claim_extractor_falls_back_to_numbered_list():
    # Two JSON-parse attempts fail, then plain-text completion returns a numbered
    # list — fallback should parse those lines into claims.
    numbered = (
        "Here are the claims:\n"
        "1. Rust was created in 2006.\n"
        "2. Rust emphasizes memory safety.\n"
        "3. too short\n"
        "- Graydon Hoare started it at Mozilla.\n"
    )
    provider = ScriptedProvider(["not json", "still not json", numbered])
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        ground=False,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text="real text")
    claims = await extractor.extract(page)
    texts = [c.text for c in claims]
    assert "Rust was created in 2006." in texts
    assert "Rust emphasizes memory safety." in texts
    assert "Graydon Hoare started it at Mozilla." in texts
    # preamble + too-short lines should be filtered
    assert not any("Here are" in t for t in texts)
    assert not any(t == "too short" for t in texts)


def test_parse_lines_requires_list_marker():
    text = "This is prose without a marker.\n1. Real claim with enough length.\n"
    out = _parse_lines(text)
    assert out == ["Real claim with enough length."]


def test_cross_reference_merges_duplicates():
    claims = [
        _claim("The Earth is round.", "https://a.com/x"),
        _claim("the earth is round.", "https://b.com/y"),
        _claim("Pineapples are fruit.", "https://c.com/z"),
    ]
    merged = cross_reference(claims)
    assert len(merged) == 2
    earth = next(c for c in merged if "earth" in c.text.lower())
    assert len(earth.citations) == 2
    assert earth.confidence > 0.5


def test_cross_reference_preserves_distinct_claims():
    claims = [
        _claim("A", "https://a.com/"),
        _claim("B", "https://b.com/"),
    ]
    merged = cross_reference(claims)
    assert len(merged) == 2


def test_cross_reference_dedupes_citations_on_same_claim():
    claims = [
        _claim("Same fact", "https://a.com/x"),
        _claim("Same fact", "https://a.com/x"),
    ]
    merged = cross_reference(claims)
    assert len(merged) == 1
    assert len(merged[0].citations) == 1
