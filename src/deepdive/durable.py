"""Durable research runs: the pipeline as an actants StateGraph.

A research run is expensive in the two ways that matter — wall-clock (dozens of
scrapes) and tokens (one LLM call per page, plus four for the report). Losing it
to a Ctrl-C at the report stage means paying for all of it again. This module
re-expresses the pipeline's stages as graph nodes over a persisted state model,
so a killed run resumes at the stage it died in and every completed stage is
skipped.

The stage boundaries are the checkpoint boundaries: query generation, search,
scrape, claim extraction, and report. They were chosen because each is where the
state genuinely changes shape, and because a crash between two of them loses at
most one stage's work.

Durability is opt-in and orthogonal to `TraceRecorder`. The recorder captures
*external calls* for byte-identical replay; the checkpointer captures *stage
results* so a run need not repeat them. A durable run that also records writes a
fresh trace per process, so a resumed run's trace covers only the resumed
portion — see the README for why replaying a resumed run needs the original.
"""

from __future__ import annotations

from typing import Annotated, Any

from actants import END, Append, SqliteCheckpointer, StateGraph
from pydantic import BaseModel, Field

from deepdive.models import Claim, ResearchReport, ScrapedPage, SearchResult

#: Node names. Public because `GraphResult.executed` reports them and the resume
#: path asserts against them — a test proving "did not re-run" needs the names.
QUERY_GEN = "query_gen"
SEARCH = "search"
SCRAPE = "scrape"
EXTRACT = "extract"
REPORT = "report"

STAGES = (QUERY_GEN, SEARCH, SCRAPE, EXTRACT, REPORT)


class ResearchState(BaseModel):
    """State one durable research run carries between stages.

    The accumulating fields use `Append` so a stage that runs in several passes
    (search over N queries, scrape over N urls) extends rather than replaces —
    and so a resumed run keeps what earlier passes already found.

    Events are part of the state, not a side channel: a resumed run must be able
    to replay the events of stages it skipped, otherwise a consumer of `run()`
    would see a truncated stream after a resume.
    """

    question: str
    queries: list[str] = Field(default_factory=list)
    results: Annotated[list[SearchResult], Append] = Field(default_factory=list)
    pages: Annotated[list[ScrapedPage], Append] = Field(default_factory=list)
    claims: Annotated[list[Claim], Append] = Field(default_factory=list)
    report: ResearchReport | None = None
    #: Emitted events, as (type, data) pairs, in order. Replayed on resume so the
    #: caller's event stream is identical whether or not the run was interrupted.
    events: Annotated[list[tuple[str, dict[str, Any]]], Append] = Field(default_factory=list)
    #: Set when a stage failed terminally; the router reads it to short-circuit to
    #: END rather than running later stages against half-built state.
    failed: bool = False


def build_graph(
    pipeline: Any,
    *,
    checkpointer: SqliteCheckpointer | None = None,
    interrupt_before_scrape: bool = False,
):
    """Compile the research graph against ``pipeline``'s collaborators.

    Takes the pipeline rather than the individual clients so the nodes see
    exactly the wrapped (filtered, recording, offline-guarded) objects the
    non-durable path uses — durability must not change which client runs.
    """
    from deepdive.analysis.claims import ClaimExtractor, cross_reference
    from deepdive.report.builder import ReportBuilder
    from deepdive.search.query_gen import generate_queries

    config = pipeline.config

    async def query_gen(state: ResearchState) -> dict[str, Any]:
        try:
            queries = await generate_queries(
                state.question, n=config.queries_per_question, llm=pipeline.llm
            )
        except Exception as exc:
            return _fail("query_generation", exc)
        texts = [q.text for q in queries]
        return {
            "queries": texts,
            "events": [("queries_generated", {"queries": texts})],
        }

    async def search(state: ResearchState) -> dict[str, Any]:
        seen: set[str] = set()
        found: list[SearchResult] = []
        events: list[tuple[str, dict[str, Any]]] = []
        for text in state.queries:
            try:
                results = await pipeline.search_client.search(
                    text, max_results=config.results_per_query
                )
            except Exception as exc:
                # Per-query isolation, matching the non-durable path: one dead
                # backend query must not lose the whole run.
                events.append(("error", {"stage": "search", "query": text, "message": str(exc)}))
                continue
            for r in results:
                url = str(r.url)
                if url in seen:
                    continue
                seen.add(url)
                found.append(r)
                events.append(("search_result", {"url": url, "title": r.title, "query": text}))
            if len(found) >= config.max_pages_per_research:
                break
        return {"results": found, "events": events}

    async def scrape(state: ResearchState) -> dict[str, Any]:
        urls = [str(r.url) for r in state.results[: config.max_pages_per_research]]
        pages: list[ScrapedPage] = []
        events: list[tuple[str, dict[str, Any]]] = []
        for url in urls:
            page = await pipeline.scraper.fetch(url)
            if page is not None:
                pages.append(page)
                events.append(("page_scraped", {"url": url, "chars": len(page.text)}))
        return {"pages": pages, "events": events}

    async def extract(state: ResearchState) -> dict[str, Any]:
        extractor = ClaimExtractor(llm=pipeline.llm, ground=pipeline.ground)
        claims: list[Claim] = []
        events: list[tuple[str, dict[str, Any]]] = []
        for page in state.pages:
            try:
                page_claims = await extractor.extract(page)
            except Exception as exc:
                events.append(
                    (
                        "error",
                        {
                            "stage": "claim_extraction",
                            "url": str(page.url),
                            "message": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )
                continue
            if pipeline.ground and not config.include_ungrounded:
                page_claims = [c for c in page_claims if c.grounded]
            claims.extend(page_claims)
            events.append(("claims_extracted", {"url": str(page.url), "count": len(page_claims)}))
        return {"claims": claims, "events": events}

    async def report(state: ResearchState) -> dict[str, Any]:
        merged = cross_reference(list(state.claims))
        builder = ReportBuilder(llm=pipeline.llm)
        try:
            built = await builder.build(state.question, merged)
        except Exception as exc:
            return _fail("report", exc)
        events: list[tuple[str, dict[str, Any]]] = [
            ("report_section", {"heading": s.heading, "body": s.body}) for s in built.sections
        ]
        events.append(("done", {"report": built.model_dump(mode="json")}))
        return {"report": built, "events": events}

    graph: StateGraph[ResearchState] = StateGraph(ResearchState)
    graph.add_node(QUERY_GEN, query_gen)
    graph.add_node(SEARCH, search)
    graph.add_node(SCRAPE, scrape)
    graph.add_node(EXTRACT, extract)
    graph.add_node(REPORT, report)
    graph.set_entry_point(QUERY_GEN)

    # Every stage can bail to END, because a terminal failure in any of them
    # leaves nothing for the later stages to work on.
    for stage, nxt in ((QUERY_GEN, SEARCH), (SEARCH, SCRAPE), (SCRAPE, EXTRACT), (EXTRACT, REPORT)):
        graph.add_conditional_edges(stage, _continue_unless_failed, {"go": nxt, "stop": END})
    graph.add_edge(REPORT, END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=[SCRAPE] if interrupt_before_scrape else None,
    )


def _continue_unless_failed(state: ResearchState) -> str:
    return "stop" if state.failed else "go"


def _fail(stage: str, exc: Exception) -> dict[str, Any]:
    return {
        "failed": True,
        "events": [("error", {"stage": stage, "message": f"{type(exc).__name__}: {exc}"})],
    }
