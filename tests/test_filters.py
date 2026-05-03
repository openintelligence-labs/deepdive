"""Source-restriction filter tests."""

from __future__ import annotations

import pytest

from deepdive.models import SearchResult
from deepdive.search.filters import (
    FilteringSearch,
    SourceFilter,
)

# ──────────────────────────────────────────────────────────────────────
# Hostname matching
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "entry,host,expected",
    [
        ("arxiv.org", "arxiv.org", True),
        ("arxiv.org", "www.arxiv.org", True),  # subdomain match
        ("arxiv.org", "arxiv.org.evil.com", False),  # no fake-suffix attack
        ("ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov", True),
        ("*.gov", "nih.gov", True),
        ("*.gov", "www.nih.gov", True),
        ("*.gov", "example.com", False),
        ("gov", "nih.gov", True),  # bare TLD = same as *.gov
        (".gov", "nih.gov", True),  # leading dot OK
        ("nature.com", "scientificamerican.com", False),
        ("medium.com", "stackoverflow.com", False),
    ],
)
def test_hostname_match(entry, host, expected):
    f = SourceFilter(allow=(entry,))
    assert f.keep(f"https://{host}/path") is expected


def test_keep_returns_false_for_unparseable_url():
    f = SourceFilter(allow=("arxiv.org",))
    assert f.keep("not-a-url") is False


def test_no_allow_no_block_passes_everything():
    f = SourceFilter()
    assert f.keep("https://anything.com/") is True


def test_block_overrides_allow():
    f = SourceFilter(allow=("medium.com",), block=("medium.com",))
    assert f.keep("https://medium.com/post") is False


def test_block_only_drops_listed_hosts():
    f = SourceFilter(block=("substack.com",))
    assert f.keep("https://writer.substack.com/x") is False
    assert f.keep("https://anything-else.com/") is True


# ──────────────────────────────────────────────────────────────────────
# FilteringSearch wrapper
# ──────────────────────────────────────────────────────────────────────


class FakeSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_max = 0

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.last_max = max_results
        return self.results[:max_results]


@pytest.mark.asyncio
async def test_filtering_search_drops_disallowed_results():
    inner = FakeSearch(
        [
            SearchResult(url="https://arxiv.org/a", title="a", snippet="..."),
            SearchResult(url="https://medium.com/b", title="b", snippet="..."),
            SearchResult(url="https://nih.gov/c", title="c", snippet="..."),
        ]
    )
    f = SourceFilter(allow=("arxiv.org", "nih.gov"))
    fs = FilteringSearch(inner, f)
    out = await fs.search("q", max_results=5)
    urls = [str(r.url) for r in out]
    assert "https://arxiv.org/a" in urls
    assert "https://nih.gov/c" in urls
    assert all("medium.com" not in u for u in urls)


@pytest.mark.asyncio
async def test_filtering_search_oversamples_to_compensate():
    """When the filter is aggressive, the wrapper asks the backend for more."""
    # Interleave so a small max_results=3 fetch doesn't hit only-mediums first.
    interleaved: list[SearchResult] = []
    for i in range(15):
        interleaved.append(SearchResult(url=f"https://medium.com/{i}", title="x", snippet="x"))
        interleaved.append(SearchResult(url=f"https://arxiv.org/{i}", title="x", snippet="x"))
    inner = FakeSearch(interleaved)
    f = SourceFilter(allow=("arxiv.org",))
    fs = FilteringSearch(inner, f, oversample=5)
    out = await fs.search("q", max_results=3)
    assert len(out) == 3
    # Wrapper requested at least max_results * oversample
    assert inner.last_max >= 15
    # All kept results are arxiv
    assert all("arxiv.org" in str(r.url) for r in out)


@pytest.mark.asyncio
async def test_filtering_search_records_rejected_for_audit():
    inner = FakeSearch(
        [
            SearchResult(url="https://medium.com/x", title="x", snippet="x"),
            SearchResult(url="https://arxiv.org/y", title="y", snippet="y"),
        ]
    )
    f = SourceFilter(allow=("arxiv.org",))
    fs = FilteringSearch(inner, f)
    await fs.search("q", max_results=5)
    assert "https://medium.com/x" in fs.rejected
    assert "https://arxiv.org/y" not in fs.rejected
