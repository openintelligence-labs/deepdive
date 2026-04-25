from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, HttpUrl


class SearchQuery(BaseModel):
    text: str
    rationale: str | None = None


class SearchResult(BaseModel):
    url: HttpUrl
    title: str
    snippet: str
    source: str = "searxng"
    score: float = 0.0


class ScrapedPage(BaseModel):
    url: HttpUrl
    title: str
    text: str
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Citation(BaseModel):
    url: HttpUrl
    title: str
    quote: str | None = None


class Claim(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    citations: list[Citation] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    body: str
    claims: list[Claim] = Field(default_factory=list)


class ResearchReport(BaseModel):
    question: str
    summary: str
    sections: list[ReportSection] = Field(default_factory=list)
    sources: list[Citation] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    total_cost_usd: float = 0.0
