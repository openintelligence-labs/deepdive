"""Local corpus indexing + offline-mode tests."""

from __future__ import annotations

import pytest
from actants.embeddings.base import EmbeddingResult

from deepdive.corpus.chunker import chunk_text
from deepdive.corpus.indexer import CorpusIndex
from deepdive.offline import OfflineViolation, assert_loopback, is_loopback
from deepdive.search.local_corpus import LocalCorpusClient, LocalCorpusScraper

# ──────────────────────────────────────────────────────────────────────
# Chunker
# ──────────────────────────────────────────────────────────────────────


def test_chunker_returns_empty_for_blank_input():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_chunker_keeps_short_text_as_one_chunk():
    chunks = chunk_text("Short sentence. Another short one.", max_chars=1000)
    assert len(chunks) == 1
    assert "Short sentence" in chunks[0].text


def test_chunker_splits_when_exceeding_max_chars():
    text = ". ".join(f"Sentence number {i}" for i in range(50)) + "."
    chunks = chunk_text(text, max_chars=200, overlap_chars=30)
    assert len(chunks) > 1
    # Every chunk's recorded slice should be findable in the original text
    # (allowing some slack for whitespace handling around boundaries)
    for c in chunks:
        assert c.offset_start >= 0
        assert c.offset_end <= len(text)


def test_chunker_offsets_are_monotonic():
    text = ". ".join(f"S{i}" for i in range(40)) + "."
    chunks = chunk_text(text, max_chars=100, overlap_chars=20)
    starts = [c.offset_start for c in chunks]
    assert starts == sorted(starts)


def test_chunker_text_equals_source_slice():
    # Span-grounding invariant: chunk.text must equal source[start:end]
    # exactly so excerpt offsets translate back to the original document.
    text = ". ".join(f"Sentence number {i}" for i in range(60)) + "."
    chunks = chunk_text(text, max_chars=200, overlap_chars=40)
    assert len(chunks) > 1, "test needs multi-chunk output to exercise overlap"
    for c in chunks:
        assert text[c.offset_start : c.offset_end] == c.text


# ──────────────────────────────────────────────────────────────────────
# Offline helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://localhost:11434/api/chat", True),
        ("http://127.0.0.1:8000/", True),
        ("http://127.5.5.5/", True),
        ("http://[::1]/", True),
        ("http://[0:0:0:0:0:0:0:1]/", True),  # IPv6 long form
        ("https://api.openai.com/v1/chat", False),
        ("https://arxiv.org/abs/x", False),
        ("not-a-url", False),
        # SSRF guard regressions: prefix-spoofers must NOT pass.
        ("http://127.evil.com/", False),
        ("http://127.0.0.1.evil.com/", False),
        ("http://localhost.evil.com/", False),
        # 0.0.0.0 is a wildcard bind, not a loopback — must be rejected.
        ("http://0.0.0.0/", False),
    ],
)
def test_is_loopback(url, expected):
    assert is_loopback(url) is expected


def test_assert_loopback_raises_for_remote():
    with pytest.raises(OfflineViolation, match="not a loopback"):
        assert_loopback("https://api.openai.com/")


def test_assert_loopback_passes_for_localhost():
    # No exception
    assert_loopback("http://localhost:11434/")


# ──────────────────────────────────────────────────────────────────────
# Indexer + local-corpus search (with FakeEmbeddingProvider)
# ──────────────────────────────────────────────────────────────────────


class FakeEmbeddings:
    """Wrap the actants fake provider as the higher-level Embeddings client."""

    def __init__(self, dim: int = 8) -> None:
        from actants.testing import FakeEmbeddingProvider

        self.provider = FakeEmbeddingProvider(dimensions=dim)

    async def embed(self, texts: list[str]) -> EmbeddingResult:
        return await self.provider.embed(texts)


@pytest.mark.asyncio
async def test_indexer_indexes_and_searches_text_file(tmp_path):
    fake = FakeEmbeddings(dim=8)
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "Apollo 11 landed on the Moon on July 20, 1969. "
        "Neil Armstrong was the first person to step onto the lunar surface. "
        "The mission was launched from Kennedy Space Center.",
        encoding="utf-8",
    )
    db_path = tmp_path / "corpus.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        added = await index.index_file(doc)
        assert added >= 1
        stats = index.stats()
        assert stats["documents"] == 1
        assert stats["chunks"] >= 1
        hits = await index.search("Apollo Moon landing")
        assert len(hits) >= 1
        assert "Apollo" in hits[0].text


@pytest.mark.asyncio
async def test_indexer_skips_unchanged_files(tmp_path):
    fake = FakeEmbeddings(dim=8)
    doc = tmp_path / "doc.txt"
    doc.write_text("Some content here. Another sentence.", encoding="utf-8")
    db_path = tmp_path / "corpus.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        added1 = await index.index_file(doc)
        added2 = await index.index_file(doc)  # should be no-op
        assert added1 >= 1
        assert added2 == 0


@pytest.mark.asyncio
async def test_indexer_reindexes_changed_files(tmp_path):
    import os
    import time

    fake = FakeEmbeddings(dim=8)
    doc = tmp_path / "doc.txt"
    doc.write_text("First version of the content.", encoding="utf-8")
    db_path = tmp_path / "corpus.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        await index.index_file(doc)
        # Modify file + bump mtime
        time.sleep(0.05)
        doc.write_text("Completely new content here.", encoding="utf-8")
        os.utime(doc, None)
        added = await index.index_file(doc)
        assert added >= 1


@pytest.mark.asyncio
async def test_local_corpus_client_returns_search_results(tmp_path):
    fake = FakeEmbeddings(dim=8)
    doc = tmp_path / "x.txt"
    doc.write_text(
        "Local first software keeps data on the device. " * 10,
        encoding="utf-8",
    )
    db_path = tmp_path / "corpus.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        await index.index_file(doc)
        client = LocalCorpusClient(index)
        results = await client.search("local first", max_results=3)
        assert len(results) >= 1
        # URL is loopback so it passes the offline gate
        assert "localhost" in str(results[0].url)
        assert results[0].title == "x.txt"
        assert results[0].source == "corpus"


@pytest.mark.asyncio
async def test_local_corpus_scraper_returns_chunk_text(tmp_path):
    fake = FakeEmbeddings(dim=8)
    doc = tmp_path / "y.txt"
    body = "Sentence one. Sentence two. Sentence three."
    doc.write_text(body, encoding="utf-8")
    db_path = tmp_path / "corpus.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        await index.index_file(doc)
        client = LocalCorpusClient(index)
        scraper = LocalCorpusScraper(index, offline=True)
        results = await client.search("Sentence", max_results=1)
        assert results
        page = await scraper.fetch(str(results[0].url))
        assert page is not None
        assert "Sentence" in page.text


@pytest.mark.asyncio
async def test_local_corpus_scraper_blocks_remote_in_offline(tmp_path):
    fake = FakeEmbeddings(dim=8)
    db_path = tmp_path / "empty.db"
    async with CorpusIndex(db_path, embeddings=fake) as index:
        scraper = LocalCorpusScraper(index, offline=True)
        # Non-corpus URL → blocked
        page = await scraper.fetch("https://example.com/")
        assert page is None


# ──────────────────────────────────────────────────────────────────────
# Pipeline offline-mode enforcement
# ──────────────────────────────────────────────────────────────────────


def test_pipeline_offline_raises_when_llm_endpoint_is_remote():
    from actants import LLM

    from deepdive.config import DeepDiveConfig
    from deepdive.pipeline import ResearchPipeline

    config = DeepDiveConfig()
    config.offline = True
    config.llm_base_url = "https://api.openai.com"
    with pytest.raises(OfflineViolation):
        ResearchPipeline(config=config, llm=LLM(model="x"))


def test_pipeline_offline_passes_for_localhost_endpoint():
    from actants import LLM

    from deepdive.config import DeepDiveConfig
    from deepdive.pipeline import ResearchPipeline

    config = DeepDiveConfig()
    config.offline = True
    config.llm_base_url = "http://localhost:11434"
    # Should not raise — Ollama default endpoint is loopback.
    pipeline = ResearchPipeline(config=config, llm=LLM(model="x"))
    assert pipeline.config.offline is True
