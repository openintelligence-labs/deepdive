# DeepDive

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-116%20passing-brightgreen)]()
[![Powered by agentic-kit](https://img.shields.io/badge/powered%20by-agentic--kit-7c3aed)](https://github.com/openintelligence-labs/agentic-kit)

A local-first AI research agent. Asks a question, searches the web (or your own documents), reads the sources, and writes a cited report. Every claim is checked against a verbatim excerpt from its source — claims the validator can't verify are dropped.

```bash
pip install deepdive
ollama pull llama3.2
deepdive research "When did Apollo 11 land on the Moon?" -o report.md
```

No API key required. Runs on Ollama by default.

---

## Features

### Span-grounded citations

The claim extractor returns each claim alongside the verbatim excerpt that supports it. A post-extraction validator re-checks every excerpt against the fetched source page via substring match. Claims that don't ground get dropped from the report.

The Markdown report ends with an Evidence appendix listing every claim and its verbatim source excerpt:

```markdown
**1.** Apollo 11 landed on the Moon on July 20, 1969.
   - [✓] [Apollo 11 - Wikipedia](https://en.wikipedia.org/wiki/Apollo_11)
     > Apollo 11 (July 16-24, 1969) was the American spaceflight that
     > first landed humans on the Moon, and the fifth crewed mission of
     > NASA's Apollo program.
```

### Trace and replay

Every research run can record a trace of every LLM call, search, and scrape. Replay reconstructs the report offline, byte-for-byte.

```bash
deepdive research "..." -o report.md         # auto-records report.md.trace.jsonl
deepdive replay report.md.trace.jsonl        # produces an identical report, no network
deepdive trace verify report.md.trace.jsonl  # re-validates every excerpt
deepdive inspect report.md.trace.jsonl       # event counts
```

### Source restriction

Restrict search to your own allow/block lists of hostnames. Subdomains match; wildcards like `*.gov` are supported.

```bash
deepdive research "..." --allow-domains arxiv.org,nih.gov
deepdive research "..." --block-domains medium.com,*.substack.com
```

### Local corpus and offline mode

Index your own documents into a sqlite-vec database, then research them with no outbound network calls.

```bash
deepdive index ~/Documents/papers -o ~/.local/share/deepdive/index.db
deepdive research "..." --corpus ~/.local/share/deepdive/index.db --offline
```

In `--offline` mode the LLM endpoint must be loopback (Ollama is fine; cloud APIs raise) and non-loopback URLs are dropped at the scraper level.

### Multi-format export

```bash
deepdive research "..." -o report.md                      # markdown (default)
deepdive research "..." -o paper.tex --export latex       # +references.bib
deepdive research "..." -o data.json --export json
deepdive research "..." -o note.md --export obsidian      # YAML + [[wikilinks]]
deepdive research "..." -o page.md --export notion
deepdive research "..." -o refs.bib --export bibtex
```

### Use as an MCP server

Expose DeepDive's `research` tool to any MCP-compatible client.

```bash
deepdive serve-mcp                         # stdio
deepdive serve-mcp --http --port 8765      # Streamable HTTP
```

---

## Install

```bash
pip install deepdive[all]      # everything: MCP server, corpus indexing
pip install deepdive[mcp]      # MCP server only
pip install deepdive[corpus]   # local corpus indexing only

ollama pull llama3.2           # tool-capable LLM (default)
ollama pull nomic-embed-text   # embeddings (only needed for --corpus)
```

---

## Architecture

DeepDive is built on [agentic-kit](https://github.com/openintelligence-labs/agentic-kit). The pipeline:

```
question
   │
   ▼  query generation              (LLM)
search backend                       (DuckDuckGo / SearxNG / local corpus)
   │                                 │  optional: --allow / --block filter
   ▼  scraper                        (web / corpus / offline)
claim extractor                      (LLM, returns excerpts)
   │
   ▼  span-grounding validator       (drops claims without verbatim excerpts)
cross-reference
   │
   ▼  report builder                 (LLM)
Markdown / LaTeX / BibTeX / JSON / Obsidian / Notion
```

Every external call is wrapped in optional recorders, which is how trace and replay work.

---

## What's not in scope

- Hosted SaaS or paid tier — DeepDive is open-source and runs on your hardware
- Multi-modal output (charts, infographics)
- Conversational follow-up on a single report — single-shot research only
- Browser automation
- Code execution as a research step

---

## Status

- **Version:** 0.3.0 (in development)
- **License:** MIT
- **Python:** 3.12+
- **Framework:** [agentic-kit ≥ 0.5.0](https://github.com/openintelligence-labs/agentic-kit)

Part of [Open Intelligence Labs](https://github.com/openintelligence-labs).
