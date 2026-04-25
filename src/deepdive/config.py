from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class DeepDiveConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DEEPDIVE_", extra="ignore")

    # LLM
    llm_provider: str = "ollama"
    llm_model: str = "llama2"
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
