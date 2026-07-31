"""Trace recording + replay tests.

The wedge claim: a recorded research run replays byte-identically offline.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from actants.llm.base import (
    BaseLLMProvider,
    ChatMessage,
    CompletionResult,
    TokenUsage,
)
from actants.llm.client import LLM

from deepdive.config import DeepDiveConfig
from deepdive.models import ScrapedPage, SearchResult
from deepdive.pipeline import ResearchPipeline
from deepdive.report.markdown import report_to_markdown
from deepdive.trace import (
    RecordingScraper,
    RecordingSearch,
    TraceRecorder,
    record_provider,
    replay_provider,
)
from deepdive.trace.replayer import ReplayingScraper, ReplayingSearch, ReplayMiss


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


class FakeSearch:
    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results

    async def search(self, query: str, max_results: int) -> list[SearchResult]:
        return self.results[:max_results]


class FakeScraper:
    def __init__(self, pages: dict[str, ScrapedPage]) -> None:
        self.pages = pages

    async def fetch(self, url: str) -> ScrapedPage | None:
        return self.pages.get(url)


def test_recorder_writes_jsonl(tmp_path):
    recorder = TraceRecorder(tmp_path / "x.trace.jsonl")
    result = CompletionResult(
        content="hi",
        model="m",
        provider="p",
        usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    recorder.record_llm_call([ChatMessage(role="user", content="q")], "m", result)
    recorder.record_event("start", {"question": "Q"})
    recorder.close()

    lines = (tmp_path / "x.trace.jsonl").read_text().strip().splitlines()
    types = [json.loads(line)["type"] for line in lines]
    assert types[0] == "trace_start"
    assert "llm_call" in types
    assert "event" in types
    assert types[-1] == "trace_end"


@pytest.mark.asyncio
async def test_recording_provider_records_and_passes_through(tmp_path):
    inner = ScriptedProvider(["hello"])
    recorder = TraceRecorder(tmp_path / "p.trace.jsonl")
    wrapped = record_provider(inner, recorder)

    result = await wrapped.complete([ChatMessage(role="user", content="say hi")], "m")
    assert result.content == "hello"
    recorder.close()

    lines = (tmp_path / "p.trace.jsonl").read_text().strip().splitlines()
    llm_lines = [json.loads(line) for line in lines if json.loads(line)["type"] == "llm_call"]
    assert len(llm_lines) == 1
    assert llm_lines[0]["result"]["content"] == "hello"


@pytest.mark.asyncio
async def test_recording_search_records_results(tmp_path):
    results = [SearchResult(url="https://a.com/", title="A", snippet="...")]
    recorder = TraceRecorder(tmp_path / "s.trace.jsonl")
    wrapped = RecordingSearch(FakeSearch(results), recorder)
    out = await wrapped.search("query", max_results=3)
    assert out == results
    recorder.close()

    lines = (tmp_path / "s.trace.jsonl").read_text().strip().splitlines()
    search_lines = [json.loads(line) for line in lines if json.loads(line)["type"] == "search"]
    assert len(search_lines) == 1
    assert search_lines[0]["results"][0]["title"] == "A"


@pytest.mark.asyncio
async def test_recording_scraper_records_page(tmp_path):
    page = ScrapedPage(url="https://a.com/", title="A", text="body")
    recorder = TraceRecorder(tmp_path / "r.trace.jsonl")
    wrapped = RecordingScraper(FakeScraper({"https://a.com/": page}), recorder)
    out = await wrapped.fetch("https://a.com/")
    assert out is not None
    assert out.text == "body"
    recorder.close()

    lines = (tmp_path / "r.trace.jsonl").read_text().strip().splitlines()
    scrape_lines = [json.loads(line) for line in lines if json.loads(line)["type"] == "scrape"]
    assert len(scrape_lines) == 1
    assert scrape_lines[0]["page"]["text"] == "body"


@pytest.mark.asyncio
async def test_replay_provider_returns_recorded_results(tmp_path):
    recorder = TraceRecorder(tmp_path / "round.trace.jsonl")
    inner = ScriptedProvider(["first answer", "second answer"])
    wrapped = record_provider(inner, recorder)
    msgs1 = [ChatMessage(role="user", content="q1")]
    msgs2 = [ChatMessage(role="user", content="q2")]
    a1 = await wrapped.complete(msgs1, "m")
    a2 = await wrapped.complete(msgs2, "m")
    recorder.close()

    replay, _ = replay_provider(tmp_path / "round.trace.jsonl")
    r1 = await replay.complete(msgs1, "m")
    r2 = await replay.complete(msgs2, "m")
    assert r1.content == a1.content == "first answer"
    assert r2.content == a2.content == "second answer"


@pytest.mark.asyncio
async def test_replay_miss_raises_when_unknown_call(tmp_path):
    recorder = TraceRecorder(tmp_path / "miss.trace.jsonl")
    recorder.close()
    replay, _ = replay_provider(tmp_path / "miss.trace.jsonl")
    with pytest.raises(ReplayMiss):
        await replay.complete([ChatMessage(role="user", content="never recorded")], "m")


@pytest.mark.asyncio
async def test_replay_search_and_scraper(tmp_path):
    results = [SearchResult(url="https://a.com/", title="A", snippet="s")]
    page = ScrapedPage(url="https://a.com/", title="A", text="body")
    recorder = TraceRecorder(tmp_path / "ss.trace.jsonl")
    rec_search = RecordingSearch(FakeSearch(results), recorder)
    rec_scraper = RecordingScraper(FakeScraper({"https://a.com/": page}), recorder)
    await rec_search.search("q", max_results=5)
    await rec_scraper.fetch("https://a.com/")
    recorder.close()

    _, index = replay_provider(tmp_path / "ss.trace.jsonl")
    rep_search = ReplayingSearch(index)
    rep_scraper = ReplayingScraper(index)
    out = await rep_search.search("q", max_results=5)
    assert out[0].title == "A"
    out_page = await rep_scraper.fetch("https://a.com/")
    assert out_page is not None and out_page.text == "body"


@pytest.mark.asyncio
async def test_pipeline_record_then_replay_identical_markdown(tmp_path):
    """End-to-end: run a pipeline with recording, replay it, assert the
    rendered Markdown is byte-identical."""
    responses = [
        '{"queries": ["q1"]}',
        '{"claims": ["Fact A."]}',
        "summary",
        "background",
        "findings",
        "contradictions",
    ]
    search_results = [SearchResult(url="https://a.com/1", title="P1", snippet="...")]
    pages = {"https://a.com/1": ScrapedPage(url="https://a.com/1", title="P1", text="hi")}

    trace_path = tmp_path / "run.trace.jsonl"

    pipeline = ResearchPipeline(
        config=DeepDiveConfig(queries_per_question=1, results_per_query=1),
        llm=LLM(provider=ScriptedProvider(responses), model="test"),
        search_client=FakeSearch(search_results),  # type: ignore[arg-type]
        scraper=FakeScraper(pages),  # type: ignore[arg-type]
        ground=False,
        trace_path=trace_path,
    )
    events = [e async for e in pipeline.run("Q?")]
    done = next(e for e in events if e.type == "done")
    from deepdive.models import ResearchReport

    report1 = ResearchReport.model_validate(done.data["report"])
    md1 = report_to_markdown(report1)

    # Replay run — no network, no scripted provider, nothing live.
    replay_p, index = replay_provider(trace_path)
    replay_pipeline = ResearchPipeline(
        config=DeepDiveConfig(queries_per_question=1, results_per_query=1),
        llm=LLM(provider=replay_p, model="test"),
        search_client=ReplayingSearch(index),  # type: ignore[arg-type]
        scraper=ReplayingScraper(index),  # type: ignore[arg-type]
        ground=False,
    )
    events2 = [e async for e in replay_pipeline.run("Q?")]
    done2 = next(e for e in events2 if e.type == "done")
    report2 = ResearchReport.model_validate(done2.data["report"])
    md2 = report_to_markdown(report2)

    # Strip generation timestamp + cost lines (depend on wall-clock); the rest
    # must match exactly.
    def _scrub(md: str) -> str:
        return "\n".join(line for line in md.splitlines() if "Generated by DeepDive" not in line)

    assert _scrub(md1) == _scrub(md2)
