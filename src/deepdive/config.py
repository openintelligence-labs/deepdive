from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepDiveConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPDIVE_", extra="ignore")

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2"
    llm_base_url: str = "http://localhost:11434"

    # Search
    search_backend: str = "duckduckgo"  # "duckduckgo" or "searxng"
    searxng_base_url: str = "http://localhost:8888"
    queries_per_question: int = 3
    results_per_query: int = 3
    scrape_timeout_seconds: float = 15.0

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Safety
    max_pages_per_research: int = 30
    max_tokens_per_report: int = 4096

    # Citation honesty
    ground_citations: bool = True
    """Require span-grounded citations (verbatim excerpts validated against source).
    Drops claims where the LLM-supplied excerpt isn't found in the fetched body.
    Set False to fall back to v0.2 plain extraction."""

    include_ungrounded: bool = False
    """When grounded extraction is on, also include claims where the validator
    couldn't confirm the excerpt. Useful for debugging; risky for publishing."""

    # Offline mode — strict air-gapped operation
    offline: bool = False
    """When True, every outbound network destination must be loopback. Cloud LLM
    providers raise OfflineViolation; non-loopback URLs are dropped from the
    scraper. The local corpus + Ollama is the only legal data path."""

    corpus_path: str | None = None
    """Path to a sqlite-vec corpus index. When set, the local-corpus search
    backend is used (in addition to or instead of the web backend)."""
