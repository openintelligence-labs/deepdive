from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal

from agentic_kit import LLM

from deepdive.analysis.claims import ClaimExtractor, cross_reference
from deepdive.config import DeepDiveConfig
from deepdive.models import Claim, ResearchReport, ScrapedPage, SearchResult
from deepdive.report.builder import ReportBuilder
from deepdive.scraper.fetch import Scraper
from deepdive.search.query_gen import generate_queries

EventType = Literal[
    "start",
    "queries_generated",
    "search_result",
    "page_scraped",
    "claims_extracted",
    "report_section",
    "done",
    "error",
]


@dataclass
class ResearchEvent:
    type: EventType
    data: dict


def _make_search_client(config: DeepDiveConfig):
    if config.search_backend == "searxng":
        from deepdive.search.searxng import SearxNGClient

        return SearxNGClient(base_url=config.searxng_base_url)
    from deepdive.search.duckduckgo import DuckDuckGoClient

    return DuckDuckGoClient()


class ResearchPipeline:
    """Runs the full question → queries → search → scrape → analyze → report loop."""

    def __init__(
        self,
        config: DeepDiveConfig | None = None,
        llm: LLM | None = None,
        search_client: object | None = None,
        scraper: Scraper | None = None,
    ) -> None:
        self.config = config or DeepDiveConfig()
        self.llm = llm or LLM(model=self.config.llm_model)
        self.search_client = search_client or _make_search_client(self.config)
        self.scraper = scraper or Scraper(timeout=self.config.scrape_timeout_seconds)

    async def run(self, question: str) -> AsyncIterator[ResearchEvent]:
        yield ResearchEvent("start", {"question": question})

        try:
            queries = await generate_queries(
                question, n=self.config.queries_per_question, llm=self.llm
            )
        except Exception as exc:
            yield ResearchEvent(
                "error",
                {"stage": "query_generation", "message": f"{type(exc).__name__}: {exc}"},
            )
            return
        yield ResearchEvent(
            "queries_generated",
            {"queries": [q.text for q in queries]},
        )

        seen_urls: set[str] = set()
        all_results: list[SearchResult] = []
        for q in queries:
            try:
                results = await self.search_client.search(
                    q.text, max_results=self.config.results_per_query
                )
            except Exception as exc:
                yield ResearchEvent(
                    "error",
                    {"stage": "search", "query": q.text, "message": str(exc)},
                )
                continue
            for r in results:
                url = str(r.url)
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                all_results.append(r)
                yield ResearchEvent(
                    "search_result",
                    {"url": url, "title": r.title, "query": q.text},
                )
            if len(all_results) >= self.config.max_pages_per_research:
                break

        urls_to_scrape = [str(r.url) for r in all_results[: self.config.max_pages_per_research]]
        pages: list[ScrapedPage] = []
        for url in urls_to_scrape:
            page = await self.scraper.fetch(url)
            if page is not None:
                pages.append(page)
                yield ResearchEvent("page_scraped", {"url": url, "chars": len(page.text)})

        extractor = ClaimExtractor(llm=self.llm)
        all_claims: list[Claim] = []
        for page in pages:
            claims = await extractor.extract(page)
            all_claims.extend(claims)
            yield ResearchEvent(
                "claims_extracted",
                {"url": str(page.url), "count": len(claims)},
            )

        merged = cross_reference(all_claims)

        builder = ReportBuilder(llm=self.llm)
        try:
            report: ResearchReport = await builder.build(question, merged)
        except Exception as exc:
            yield ResearchEvent(
                "error",
                {"stage": "report", "message": f"{type(exc).__name__}: {exc}"},
            )
            return
        for section in report.sections:
            yield ResearchEvent(
                "report_section",
                {"heading": section.heading, "body": section.body},
            )

        yield ResearchEvent(
            "done",
            {"report": report.model_dump(mode="json")},
        )
