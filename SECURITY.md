# Security policy

## Reporting a vulnerability

If you find a security issue in DeepDive — the research pipeline, the search-backend adapters, the corpus indexer, the MCP server, or the report renderer — please **do not open a public issue**.

File a private report through GitHub Security Advisories:

> **<https://github.com/openintelligence-labs/deepdive/security/advisories/new>**

Include:

1. The component affected (e.g. "MCP server", "SearxNG adapter", "corpus indexer").
2. A reproduction case — minimal input that demonstrates the issue.
3. The impact you've observed or believe is possible.
4. Whether you've already disclosed the issue elsewhere.

We aim to acknowledge within 48 hours and to publish a fix (or a detailed mitigation) within 30 days. For high-severity issues we'll request a coordinated disclosure window — typically 90 days from first report.

## Supported versions

| Version | Status |
|---|---|
| 0.3.x (current) | Supported. Fixes land in the latest patch. |
| < 0.3 | Unsupported. Please upgrade. |

## Scope

DeepDive fetches attacker-controlled content by design — it retrieves and summarizes arbitrary web pages. That makes the boundary between "fetched content" and "our process" the interesting part of the threat model.

**In scope:**

- Code execution or path traversal triggered by fetched page content, a search-backend response, or a corpus document.
- Prompt injection in fetched content that escalates into tool calls, filesystem access, or exfiltration of local corpus material into a report or an outbound request.
- SSRF: a search result or redirect steering a fetch at localhost, link-local metadata endpoints, or other internal addresses.
- Credential leakage — provider API keys or SearxNG credentials appearing in logs, reports, traces, or cached artifacts.
- Corpus indexer flaws allowing reads outside the directory the user pointed it at, or writes outside the configured index location.
- MCP server flaws allowing unauthorized tool invocation or filesystem mutation beyond documented behaviour.
- Report rendering that yields an executable artifact (e.g. script injection in HTML output opened in a browser).

**Out of scope** (working as designed):

- DeepDive fetching a URL you asked it to research, or a link a search backend returned for your query. Retrieval is the product.
- A model summarizing a page inaccurately, or a source being low-quality. Research quality is not a security boundary.
- Secrets you place in a query reaching the LLM provider or search backend you configured.

## Data flow

DeepDive is local-by-default with pluggable transports and ships zero telemetry. It makes outbound requests to exactly three things you configure: your search backend (default: your own SearxNG instance), the pages that backend returns, and your LLM provider (default: local Ollama). Your corpus, index, and generated reports stay on disk. Nothing is phoned home — an outbound request to a host you did not configure is a security bug.
