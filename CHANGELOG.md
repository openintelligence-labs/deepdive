# Changelog

## [Unreleased]

### Fixed
- Pin `mcp>=1.0,<2` in the `mcp`, `all`, and `dev` extras: the MCP Python SDK 2.0 (released after v0.3.0 was verified) restructured its API and breaks `actants.mcp` / `deepdive serve-mcp`. Installs that pulled `mcp==2.0.0` failed the in-memory MCP roundtrip and could not build the server.

## [0.3.0] - 2026-07-29

### Added
- Span-grounded citations — every claim carries the verbatim excerpt that supports it (`Citation.excerpt`, `offset_start`, `offset_end`, `grounded`). A post-extraction validator re-checks each excerpt against the fetched source; ungrounded claims are dropped from the report by default. CLI: `--ground/--no-ground`, `--include-ungrounded`. Markdown reports gain an Evidence appendix listing every claim with its verbatim source excerpt.
- Trace recording + replay — `--trace FILE` records every LLM call, search, and scrape to a `.jsonl` audit trace (auto-recorded next to `-o` output). `deepdive replay` reconstructs the report offline, byte-for-byte; `deepdive inspect` pretty-prints a trace; `deepdive trace verify` re-validates every recorded excerpt.
- Source restriction — `--allow-domains` / `--block-domains` hostname filters wrap any search backend (subdomain-aware, `*.gov`-style wildcards, no fake-suffix bypass).
- Local corpus — `deepdive index` builds a sqlite-vec database from PDF / Markdown / HTML / TXT with offset-preserving chunking; `--corpus` researches it instead of the web. Install with the `corpus` extra.
- Offline mode — `--offline` enforces loopback-only LLM endpoints (cloud providers raise `OfflineViolation`) and drops non-loopback URLs at the scraper level.
- MCP server — `deepdive serve-mcp` exposes the `research` tool over the Model Context Protocol (stdio or Streamable HTTP). Install with the `mcp` extra.
- Multi-format export — `--export latex|bibtex|json|obsidian|notion` alongside the default Markdown; LaTeX also writes a sibling `references.bib`.
- CLI polish — `--plan-only` dry-run, and a `--force` guard so `research`/`replay` refuse to overwrite an existing output file by default.

### Changed
- Dependency renamed: `agentic-kit` is now published as `actants`; pin bumped to `actants>=0.5.0`.
- Default Ollama model is now `llama3.2` (tool-capable), matching the documented quick start; previously `llama2`.
- Grounded claims boost confidence to 0.8 (was indistinguishable from ungrounded at 0.6); cross-referencing still adds +0.1 per corroborating source.
- Per-page claim-extraction failures no longer abort the whole research run.

### Fixed
- `offline.is_loopback` now uses `ipaddress.ip_address().is_loopback` instead of string prefixes, closing spoofs like `127.evil.com` / `localhost.evil.com` / long-form IPv6, and no longer treats `0.0.0.0` as loopback.
- SSRF guard in the scraper: URLs whose host resolves to loopback / private / link-local / reserved addresses are rejected, redirects are followed manually so every hop is re-validated, and `fetch_many` survives per-URL exceptions instead of poisoning the batch.
- Excerpt grounding adds a case-folded fallback so excerpts differing only in capitalization still ground.
- DuckDuckGo search runs in a thread (`asyncio.to_thread`) instead of blocking the event loop.
- Corpus chunker enforces `chunk.text == source[start:end]` so span-grounding offsets always map back to the original document.
- Corpus indexer casts the embedding dimension to `int` before interpolating into DDL.
- Trace replayer guards its response queues with a lock for parallel extraction.

## [0.2.0] - 2026-04-24

Shipped as part of the initial public release.

### Added
- `report_to_markdown()` — render any `ResearchReport` as a self-contained Markdown document with inline links, timestamp, and cost footer.
- `POST /api/research/markdown` — non-streaming endpoint that returns the final report as Markdown.
- `deepdive research --output FILE` / `-o FILE` — save the final report to a Markdown file.
- Query generation and claim extraction now use `actants.LLM.extract()` for provider-agnostic structured output with automatic self-repair.

### Changed
- Bumped `actants` dependency to >=0.3.0 to pick up streaming tool calls and `extract`.
- Dropped private regex-based JSON array parsing in favor of `LLM.extract()`.

## [0.1.0]

### Added
- Query generation via actants
- SearxNG + DuckDuckGo search clients
- Parallel web scraper (selectolax, httpx)
- Claim extraction + cross-reference with confidence boosting
- Structured `ResearchReport` with Background / Findings / Contradictions sections
- FastAPI server with `/api/research` SSE endpoint and `/health` probe
- Full pipeline with progress events
- `deepdive` CLI with rich live progress UI
