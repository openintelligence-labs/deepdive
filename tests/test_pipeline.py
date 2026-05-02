from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from agentic_kit.llm.base import BaseLLMProvider, CompletionResult, TokenUsage
from agentic_kit.llm.client import LLM

from deepdive.config import DeepDiveConfig
from deepdive.models import ScrapedPage, SearchResult
from deepdive.pipeline import ResearchPipeline


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> CompletionResult:
        content = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, *a, **kw) -> AsyncIterator[str]:
        yield "x"

    async def health(self) -> bool:
        return True


class FakeSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        self.calls.append(query)
        return self.results[:max_results]


class FakeScraper:
    def __init__(self, pages: dict[str, ScrapedPage]) -> None:
        self.pages = pages

    async def fetch(self, url: str) -> ScrapedPage | None:
        return self.pages.get(url)


@pytest.mark.asyncio
async def test_pipeline_emits_full_event_sequence():
    responses = [
        '{"queries": ["query one", "query two"]}',  # query generation (structured)
        '{"claims": ["Fact A from page 1."]}',  # claim extraction page 1 (structured)
        '{"claims": ["Fact A from page 1.", "Fact B from page 2."]}',  # page 2 (structured)
        "One paragraph summary.",  # summary (plain text)
        "Background text",  # section 1
        "Findings text",  # section 2
        "Contradictions text",  # section 3
    ]
    provider = ScriptedProvider(responses)
    llm = LLM(provider=provider, model="test")

    search_results = [
        SearchResult(url="https://a.com/1", title="Page 1", snippet="..."),
        SearchResult(url="https://a.com/2", title="Page 2", snippet="..."),
    ]
    fake_search = FakeSearch(search_results)

    pages = {
        "https://a.com/1": ScrapedPage(url="https://a.com/1", title="Page 1", text="Content 1"),
        "https://a.com/2": ScrapedPage(url="https://a.com/2", title="Page 2", text="Content 2"),
    }
    fake_scraper = FakeScraper(pages)

    config = DeepDiveConfig(
        queries_per_question=2,
        results_per_query=5,
        max_pages_per_research=5,
    )
    pipeline = ResearchPipeline(
        config=config,
        llm=llm,
        search_client=fake_search,  # type: ignore[arg-type]
        scraper=fake_scraper,  # type: ignore[arg-type]
        ground=False,
    )

    events = [e async for e in pipeline.run("What is X?")]
    types = [e.type for e in events]

    assert types[0] == "start"
    assert "queries_generated" in types
    assert types.count("search_result") == 2
    assert types.count("page_scraped") == 2
    assert types.count("claims_extracted") == 2
    assert types.count("report_section") == 3
    assert types[-1] == "done"

    done_event = events[-1]
    report = done_event.data["report"]
    assert report["question"] == "What is X?"
    assert report["summary"] == "One paragraph summary."
    assert len(report["sections"]) == 3


class FailingSearch:
    async def search(self, query: str, *, max_results: int = 10):
        raise RuntimeError("searxng unreachable")


@pytest.mark.asyncio
async def test_pipeline_emits_error_event_on_search_failure():
    provider = ScriptedProvider(['{"queries": ["q1"]}'])
    llm = LLM(provider=provider, model="test")
    pipeline = ResearchPipeline(
        config=DeepDiveConfig(queries_per_question=1),
        llm=llm,
        search_client=FailingSearch(),  # type: ignore[arg-type]
        scraper=FakeScraper({}),  # type: ignore[arg-type]
    )
    events = [e async for e in pipeline.run("Q?")]
    types = [e.type for e in events]
    assert "error" in types
    err = next(e for e in events if e.type == "error")
    assert err.data["stage"] == "search"
    assert "searxng unreachable" in err.data["message"]


@pytest.mark.asyncio
async def test_pipeline_produces_graceful_report_when_no_claims():
    # Query gen succeeds but all claim extractions return empty — report should
    # still be produced with the no-claims summary and zero cost.
    responses = [
        '{"queries": ["q1"]}',
        '{"claims": []}',
    ]
    provider = ScriptedProvider(responses)
    llm = LLM(provider=provider, model="test")

    search_results = [SearchResult(url="https://a.com/1", title="P1", snippet="...")]
    pages = {
        "https://a.com/1": ScrapedPage(url="https://a.com/1", title="P1", text="hi"),
    }
    pipeline = ResearchPipeline(
        config=DeepDiveConfig(queries_per_question=1, results_per_query=1),
        llm=llm,
        search_client=FakeSearch(search_results),  # type: ignore[arg-type]
        scraper=FakeScraper(pages),  # type: ignore[arg-type]
        ground=False,
    )
    events = [e async for e in pipeline.run("Q?")]
    done = next(e for e in events if e.type == "done")
    report = done.data["report"]
    assert report["sections"] == []
    assert report["sources"] == []
    assert report["total_cost_usd"] == 0.0
    assert "No verifiable claims" in report["summary"]


class CostedProvider(BaseLLMProvider):
    name = "costed"

    def __init__(self, responses: list[str], cost_per_call: float) -> None:
        self._responses = list(responses)
        self._cost = cost_per_call
        self._i = 0

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> CompletionResult:
        content = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_usd=self._cost,
        )

    async def stream(self, *a, **kw) -> AsyncIterator[str]:
        yield "x"

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_report_builder_populates_total_cost():
    # 1 summary + 3 sections = 4 LLM calls in the builder; at $0.01 each, total $0.04.
    responses = [
        '{"queries": ["q1"]}',
        '{"claims": ["Fact A."]}',
        "summary",
        "background",
        "findings",
        "contradictions",
    ]
    provider = CostedProvider(responses, cost_per_call=0.01)
    llm = LLM(provider=provider, model="test")

    search_results = [SearchResult(url="https://a.com/1", title="P1", snippet="...")]
    pages = {
        "https://a.com/1": ScrapedPage(url="https://a.com/1", title="P1", text="hi"),
    }
    pipeline = ResearchPipeline(
        config=DeepDiveConfig(queries_per_question=1, results_per_query=1),
        llm=llm,
        search_client=FakeSearch(search_results),  # type: ignore[arg-type]
        scraper=FakeScraper(pages),  # type: ignore[arg-type]
        ground=False,
    )
    events = [e async for e in pipeline.run("Q?")]
    done = next(e for e in events if e.type == "done")
    report = done.data["report"]
    assert report["total_cost_usd"] == pytest.approx(0.04)
    assert len(report["sources"]) == 1
