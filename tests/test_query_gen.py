from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from actants.llm.base import BaseLLMProvider, CompletionResult, TokenUsage
from actants.llm.client import LLM

from deepdive.search.query_gen import generate_queries


class ScriptedProvider(BaseLLMProvider):
    name = "scripted"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    async def complete(
        self, messages, model, temperature=0.7, max_tokens=None, **kwargs
    ) -> CompletionResult:
        content = self._responses.pop(0) if self._responses else ""
        return CompletionResult(
            content=content,
            model=model,
            provider=self.name,
            usage=TokenUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
        )

    async def stream(self, *a, **kw) -> AsyncIterator[str]:
        yield ""

    async def health(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_generate_queries_parses_structured_output():
    provider = ScriptedProvider(['{"queries": ["q1", "q2", "q3"]}'])
    llm = LLM(provider=provider, model="test", tracing=False)
    queries = await generate_queries("what is X?", n=3, llm=llm)
    assert [q.text for q in queries] == ["q1", "q2", "q3"]


@pytest.mark.asyncio
async def test_generate_queries_fallback_on_bad_output():
    # extract() retries once then raises — fallback to line splitting runs
    provider = ScriptedProvider(
        [
            "nope",
            "still nope",
            "- first query\n- second query\n- third query",
        ]
    )
    llm = LLM(provider=provider, model="test", tracing=False)
    queries = await generate_queries("what?", n=3, llm=llm)
    assert len(queries) == 3
    assert queries[0].text.startswith("first")


@pytest.mark.asyncio
async def test_generate_queries_respects_n():
    provider = ScriptedProvider(['{"queries": ["a", "b", "c", "d", "e"]}'])
    llm = LLM(provider=provider, model="test", tracing=False)
    queries = await generate_queries("q", n=2, llm=llm)
    assert len(queries) == 2
