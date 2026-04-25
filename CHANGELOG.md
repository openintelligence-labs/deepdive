# Changelog

## [0.2.0] - unreleased

### Added
- `report_to_markdown()` — render any `ResearchReport` as a self-contained Markdown document with inline links, timestamp, and cost footer.
- `POST /api/research/markdown` — non-streaming endpoint that returns the final report as Markdown.
- `deepdive research --output FILE` / `-o FILE` — save the final report to a Markdown file.
- Query generation and claim extraction now use `agentic_kit.LLM.extract()` for provider-agnostic structured output with automatic self-repair.

### Changed
- Bumped `agentic-kit` dependency to >=0.3.0 to pick up streaming tool calls and `extract`.
- Dropped private regex-based JSON array parsing in favor of `LLM.extract()`.

## [0.1.0]

### Added
- Query generation via agentic-kit
- SearxNG + DuckDuckGo search clients
- Parallel web scraper (selectolax, httpx)
- Claim extraction + cross-reference with confidence boosting
- Structured `ResearchReport` with Background / Findings / Contradictions sections
- FastAPI server with `/api/research` SSE endpoint and `/health` probe
- Full pipeline with progress events
- `deepdive` CLI with rich live progress UI
