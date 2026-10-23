"""Durable-pipeline tests: checkpointed runs, and resume skipping completed stages."""

from __future__ import annotations

import pytest
from actants import SqliteCheckpointer
from actants.llm.base import BaseLLMProvider, CompletionResult, TokenUsage
from actants.llm.client import LLM

from deepdive.config import DeepDiveConfig
from deepdive.durable import EXTRACT, QUERY_GEN, REPORT, SCRAPE, SEARCH, ResearchState
from deepdive.models import ScrapedPage, SearchResult
from deepdive.pipeline import ResearchPipeline

RESPONSES = [
    '{"queries": ["query one"]}',
    '{"claims": ["Fact A from page one."]}',
    "One paragraph summary.",
    "Background text",
    "Findings text",
    "Contradictions text",
]


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._i = 0
        self.calls = 0

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> CompletionResult:
        content = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        self.calls += 1
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def health(self) -> bool:
        return True


class CountingSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.calls = 0

    async def search(self, query: str, *, max_results: int = 10) -> list[SearchResult]:
        self.calls += 1
        return self.results[:max_results]


class CountingScraper:
    def __init__(self, pages: dict[str, ScrapedPage]) -> None:
        self.pages = pages
        self.calls = 0

    async def fetch(self, url: str) -> ScrapedPage | None:
        self.calls += 1
        return self.pages.get(url)


class _HardCrash(BaseException):
    """Stands in for a process death (SIGKILL / os._exit).

    Deliberately a BaseException: an ordinary Exception would be caught by the
    graph and flip the thread to ``failed``, which actants refuses to resume —
    correctly, since a raised error means the run decided to stop. Only a run
    whose process vanished leaves the thread ``running``, and that is the case
    resume exists for, so that is the case these tests must reproduce.
    """


class CrashingScraper(CountingScraper):
    """Dies partway through the scrape stage, leaving the thread mid-run."""

    async def fetch(self, url: str) -> ScrapedPage | None:
        raise _HardCrash


def _fixtures():
    results = [SearchResult(url="https://a.com/1", title="Page 1", snippet="...")]
    pages = {"https://a.com/1": ScrapedPage(url="https://a.com/1", title="Page 1", text="Content")}
    return results, pages


def _config() -> DeepDiveConfig:
    return DeepDiveConfig(queries_per_question=1, results_per_query=1, max_pages_per_research=5)


def _pipeline(provider, search, scraper, checkpointer, thread_id):
    return ResearchPipeline(
        config=_config(),
        llm=LLM(provider=provider, model="test", tracing=False),
        search_client=search,  # type: ignore[arg-type]
        scraper=scraper,  # type: ignore[arg-type]
        ground=False,
        checkpointer=checkpointer,
        thread_id=thread_id,
    )


@pytest.mark.asyncio
async def test_durable_run_emits_same_events_as_plain_run(tmp_path):
    """The durable path must be event-for-event identical to the default one."""
    results, pages = _fixtures()

    plain = ResearchPipeline(
        config=_config(),
        llm=LLM(provider=ScriptedProvider(RESPONSES), model="test", tracing=False),
        search_client=CountingSearch(results),  # type: ignore[arg-type]
        scraper=CountingScraper(pages),  # type: ignore[arg-type]
        ground=False,
    )
    plain_events = [(e.type, e.data) async for e in plain.run("What is X?")]

    durable = _pipeline(
        ScriptedProvider(RESPONSES),
        CountingSearch(results),
        CountingScraper(pages),
        SqliteCheckpointer(tmp_path / "runs.db"),
        "thread-1",
    )
    durable_events = [(e.type, e.data) async for e in durable.run("What is X?")]

    assert [t for t, _ in plain_events] == [t for t, _ in durable_events]
    # Payloads match too, modulo the report's generated_at timestamp.
    for (pt, pd), (dt, dd) in zip(plain_events, durable_events, strict=True):
        assert pt == dt
        if pt == "done":
            pd["report"].pop("generated_at")
            dd["report"].pop("generated_at")
        assert pd == dd


@pytest.mark.asyncio
async def test_resume_skips_completed_stages(tmp_path):
    """The core guarantee: stages that finished before the crash never re-run."""
    results, pages = _fixtures()
    db = tmp_path / "runs.db"

    crash_provider = ScriptedProvider(RESPONSES)
    crash_search = CountingSearch(results)
    crashing = _pipeline(
        crash_provider, crash_search, CrashingScraper({}), SqliteCheckpointer(db), "thread-x"
    )
    with pytest.raises(_HardCrash):
        async for _ in crashing.run("What is X?"):
            pass

    # query_gen (1 LLM call) and search (1 call) completed before the crash.
    assert crash_provider.calls == 1
    assert crash_search.calls == 1

    resume_provider = ScriptedProvider(RESPONSES[1:])
    resume_search = CountingSearch(results)
    resume_scraper = CountingScraper(pages)
    resumed = _pipeline(
        resume_provider, resume_search, resume_scraper, SqliteCheckpointer(db), "thread-x"
    )
    events = [e async for e in resumed.resume()]

    # The proof: search never ran again, and the completed stages' LLM calls were
    # not repeated — the resumed process only paid for extract + report.
    assert resume_search.calls == 0
    assert resume_scraper.calls == 1
    assert resume_provider.calls == 5  # 1 extract + 4 report, no query generation

    types = [e.type for e in events]
    assert types.count("report_section") == 3
    assert types[-1] == "done"


@pytest.mark.asyncio
async def test_resume_replays_events_of_skipped_stages(tmp_path):
    """A resumed run yields the whole stream, not just the part it re-ran."""
    results, pages = _fixtures()
    db = tmp_path / "runs.db"

    crashing = _pipeline(
        ScriptedProvider(RESPONSES),
        CountingSearch(results),
        CrashingScraper({}),
        SqliteCheckpointer(db),
        "thread-y",
    )
    with pytest.raises(_HardCrash):
        async for _ in crashing.run("What is X?"):
            pass

    resumed = _pipeline(
        ScriptedProvider(RESPONSES[1:]),
        CountingSearch(results),
        CountingScraper(pages),
        SqliteCheckpointer(db),
        "thread-y",
    )
    types = [e.type async for e in resumed.resume()]

    # queries_generated and search_result came from stages that did not re-run.
    assert "queries_generated" in types
    assert "search_result" in types


@pytest.mark.asyncio
async def test_resume_without_checkpointer_raises():
    pipeline = ResearchPipeline(
        config=_config(),
        llm=LLM(provider=ScriptedProvider(RESPONSES), model="test", tracing=False),
        search_client=CountingSearch([]),  # type: ignore[arg-type]
        scraper=CountingScraper({}),  # type: ignore[arg-type]
    )
    with pytest.raises(ValueError, match="needs a checkpointer"):
        async for _ in pipeline.resume():
            pass


@pytest.mark.asyncio
async def test_durable_run_survives_process_boundary(tmp_path):
    """State is read back from SQLite, not from memory carried across the resume."""
    results, pages = _fixtures()
    db = tmp_path / "runs.db"

    crashing = _pipeline(
        ScriptedProvider(RESPONSES),
        CountingSearch(results),
        CrashingScraper({}),
        SqliteCheckpointer(db),
        "thread-z",
    )
    with pytest.raises(_HardCrash):
        async for _ in crashing.run("What is X?"):
            pass

    # A fresh checkpointer instance — nothing shared with the crashed run.
    stored = await SqliteCheckpointer(db).get("thread-z")
    assert stored is not None
    state = ResearchState.model_validate_json(
        __import__("json").loads(stored.messages[0].content)["state_json"]
    )
    assert state.queries == ["query one"]
    assert len(state.results) == 1


def test_state_declares_expected_stages():
    assert (QUERY_GEN, SEARCH, SCRAPE, EXTRACT, REPORT) == (
        "query_gen",
        "search",
        "scrape",
        "extract",
        "report",
    )


def test_cli_rejects_resume_with_a_question():
    """--resume takes the question from the checkpoint; passing both is ambiguous."""
    from click.testing import CliRunner

    from deepdive.cli import main

    result = CliRunner().invoke(main, ["research", "a question", "--resume", "t1"])
    assert result.exit_code == 2
    assert "from its checkpoint" in result.output


def test_cli_requires_a_question_or_resume():
    from click.testing import CliRunner

    from deepdive.cli import main

    result = CliRunner().invoke(main, ["research"])
    assert result.exit_code == 2
    assert "--resume" in result.output


@pytest.mark.asyncio
async def test_approve_scrape_pauses_before_any_fetch(tmp_path):
    """The approval gate must stop the run before it spends anything on scraping."""
    results, pages = _fixtures()
    db = tmp_path / "runs.db"
    scraper = CountingScraper(pages)

    def build(provider, checkpointer):
        return ResearchPipeline(
            config=_config(),
            llm=LLM(provider=provider, model="test", tracing=False),
            search_client=CountingSearch(results),  # type: ignore[arg-type]
            scraper=scraper,  # type: ignore[arg-type]
            ground=False,
            checkpointer=checkpointer,
            thread_id="gated",
            approve_scrape=True,
        )

    gated = build(ScriptedProvider(RESPONSES), SqliteCheckpointer(db))
    types = [e.type async for e in gated.run("What is X?")]
    assert types[-1] == "error"  # paused, surfaced as a resumable notice
    assert scraper.calls == 0

    approved = build(ScriptedProvider(RESPONSES[1:]), SqliteCheckpointer(db))
    resumed = [e.type async for e in approved.resume()]
    assert scraper.calls == 1
    assert resumed[-1] == "done"
