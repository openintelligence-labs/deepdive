"""Tests for the citation-honesty wedge: span-grounded claims."""

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

from deepdive.analysis.claims import ClaimExtractor
from deepdive.analysis.grounding import (
    filter_grounded,
    find_excerpt_offsets,
    ground_citation,
)
from deepdive.models import Citation, Claim, ScrapedPage


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        content = self._responses.pop(0) if self._responses else ""
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


# ──────────────────────────────────────────────────────────────────────
# find_excerpt_offsets
# ──────────────────────────────────────────────────────────────────────


def test_find_excerpt_exact_match():
    body = "The quick brown fox jumps over the lazy dog."
    offsets = find_excerpt_offsets(body, "brown fox jumps")
    assert offsets == (10, 25)
    assert body[10:25] == "brown fox jumps"


def test_find_excerpt_returns_none_for_missing():
    body = "The quick brown fox."
    assert find_excerpt_offsets(body, "purple elephant") is None


def test_find_excerpt_returns_none_for_empty():
    assert find_excerpt_offsets("", "anything") is None
    assert find_excerpt_offsets("anything", "") is None


def test_find_excerpt_normalized_whitespace_match():
    """LLM excerpt has single spaces; source has newlines + multiple spaces."""
    body = "Rust\nis  a   systems\tprogramming language."
    offsets = find_excerpt_offsets(body, "Rust is a systems programming language.")
    assert offsets is not None
    start, end = offsets
    # The matched span in the original body should cover from "Rust" through "."
    assert body[start:end].startswith("Rust")
    assert body[start:end].endswith(".")


def test_find_excerpt_unicode_normalization():
    """NFKC-equivalent forms (e.g. compatibility ligatures) match.

    Note: typographically distinct chars like curly vs ASCII quotes are NOT
    folded — that's intentional; NFKC preserves them and so do we.
    """
    # Compatibility "ﬁ" (U+FB01) decomposes to "fi" under NFKC.
    body = "The eﬃcient algorithm runs fast."  # contains "ﬃ" ligature
    offsets = find_excerpt_offsets(body, "efficient algorithm")
    assert offsets is not None


def test_find_excerpt_case_insensitive_fallback():
    """Source uses one case, LLM extracts another. Should still ground.

    This is common when the LLM rewords or when the body is lowercased during
    HTML rendering. Without the case-insensitive fallback, legitimate claims
    would be silently dropped.
    """
    body = "the apollo 11 mission landed on the moon in july 1969."
    offsets = find_excerpt_offsets(body, "Apollo 11 mission landed on the Moon")
    assert offsets is not None
    start, end = offsets
    assert "apollo 11" in body[start:end].lower()


# ──────────────────────────────────────────────────────────────────────
# ground_citation
# ──────────────────────────────────────────────────────────────────────


def test_ground_citation_marks_grounded_when_excerpt_found():
    body = "Python was created by Guido van Rossum in 1991."
    citation = Citation(
        url="https://example.com/", title="t", excerpt="created by Guido van Rossum"
    )
    grounded = ground_citation(body, citation)
    assert grounded.grounded is True
    assert grounded.offset_start is not None
    assert grounded.offset_end is not None
    assert body[grounded.offset_start : grounded.offset_end] == "created by Guido van Rossum"


def test_ground_citation_marks_ungrounded_when_excerpt_missing():
    body = "Python was created by Guido van Rossum in 1991."
    citation = Citation(
        url="https://example.com/", title="t",
        excerpt="invented by aliens",  # not in source
    )
    grounded = ground_citation(body, citation)
    assert grounded.grounded is False
    assert grounded.offset_start is None


def test_ground_citation_with_no_excerpt_is_ungrounded():
    citation = Citation(url="https://example.com/", title="t")  # no excerpt
    grounded = ground_citation("any body text", citation)
    assert grounded.grounded is False


# ──────────────────────────────────────────────────────────────────────
# filter_grounded
# ──────────────────────────────────────────────────────────────────────


def test_filter_grounded_drops_ungrounded_by_default():
    grounded_claim = Claim(
        text="grounded",
        citations=[Citation(url="https://x.com/", title="t", grounded=True)],
    )
    ungrounded_claim = Claim(
        text="ungrounded",
        citations=[Citation(url="https://x.com/", title="t", grounded=False)],
    )
    out = filter_grounded([grounded_claim, ungrounded_claim])
    assert [c.text for c in out] == ["grounded"]


def test_filter_grounded_keeps_all_when_include_ungrounded():
    grounded_claim = Claim(
        text="grounded",
        citations=[Citation(url="https://x.com/", title="t", grounded=True)],
    )
    ungrounded_claim = Claim(
        text="ungrounded",
        citations=[Citation(url="https://x.com/", title="t", grounded=False)],
    )
    out = filter_grounded(
        [grounded_claim, ungrounded_claim], include_ungrounded=True
    )
    assert len(out) == 2


def test_claim_grounded_property_reflects_any_citation():
    c = Claim(
        text="t",
        citations=[
            Citation(url="https://x.com/a", title="a", grounded=False),
            Citation(url="https://x.com/b", title="b", grounded=True),
        ],
    )
    assert c.grounded is True


# ──────────────────────────────────────────────────────────────────────
# ClaimExtractor with grounding (the integration)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_grounded_extractor_returns_grounded_claims_when_excerpts_match():
    body = (
        "Rust was created by Graydon Hoare at Mozilla. "
        "It emphasizes memory safety without garbage collection."
    )
    response = (
        '{"claims": ['
        '  {"text": "Rust was created by Graydon Hoare.", '
        '   "excerpt": "Rust was created by Graydon Hoare at Mozilla"},'
        '  {"text": "Rust emphasizes memory safety.", '
        '   "excerpt": "It emphasizes memory safety without garbage collection"}'
        ']}'
    )
    provider = ScriptedProvider([response])
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        max_claims=5,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text=body)
    claims = await extractor.extract(page)
    assert len(claims) == 2
    assert all(c.grounded for c in claims)
    # Both citations have offsets pointing into the body
    for claim in claims:
        cit = claim.citations[0]
        assert cit.offset_start is not None
        assert body[cit.offset_start : cit.offset_end] == cit.excerpt


@pytest.mark.asyncio
async def test_grounded_extractor_marks_invented_excerpts_ungrounded():
    body = "Rust was created by Graydon Hoare at Mozilla."
    response = (
        '{"claims": ['
        '  {"text": "Rust was created by Graydon Hoare.", '
        '   "excerpt": "Rust was created by Graydon Hoare at Mozilla"},'
        '  {"text": "Rust is fastest language.", '
        '   "excerpt": "Rust is the absolute fastest language ever"}'
        ']}'
    )
    provider = ScriptedProvider([response])
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        max_claims=5,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text=body)
    claims = await extractor.extract(page)
    grounded = [c for c in claims if c.grounded]
    ungrounded = [c for c in claims if not c.grounded]
    assert len(grounded) == 1
    assert len(ungrounded) == 1
    assert grounded[0].text == "Rust was created by Graydon Hoare."


@pytest.mark.asyncio
async def test_grounded_extractor_falls_through_to_legacy_when_grounded_fails():
    """If the LLM can't return the grounding schema at all, fall back to legacy
    extraction so the page isn't dropped silently."""
    # First call: invalid JSON for grounding schema (caught by extract retry).
    # extract() retries once, so two invalid grounding responses are consumed.
    # Then legacy: extract() retries again (1 schema attempt), then complete().
    bad_grounding = "totally not json"
    legacy_json = '{"claims": ["Fact A", "Fact B"]}'
    provider = ScriptedProvider([bad_grounding, bad_grounding, legacy_json])
    extractor = ClaimExtractor(
        llm=LLM(provider=provider, model="test", tracing=False),
        max_claims=5,
    )
    page = ScrapedPage(url="https://example.com/", title="t", text="some body")
    claims = await extractor.extract(page)
    # Legacy claims have no grounding — should still return them
    assert len(claims) == 2
    assert claims[0].text == "Fact A"
