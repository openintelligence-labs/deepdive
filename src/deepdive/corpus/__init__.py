"""Local corpus indexing — `deepdive index` then research with `--corpus`.

Indexes PDF / Markdown / HTML / plain-text documents into a sqlite-vec database
under your control. The local-corpus search backend treats those chunks as
search results that the rest of the pipeline (scrape → claim extraction →
report) handles like any web result.

Combined with ``--offline``, this gives a fully air-gapped research path:
no LLM-cloud, no web search, no scraping the open internet — only your
documents and a local Ollama model.
"""

from __future__ import annotations

from deepdive.corpus.chunker import Chunk, chunk_text
from deepdive.corpus.indexer import CorpusIndex, extract_text

__all__ = ["Chunk", "CorpusIndex", "chunk_text", "extract_text"]
