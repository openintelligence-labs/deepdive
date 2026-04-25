from __future__ import annotations

from agentic_kit import LLM
from pydantic import BaseModel, Field

from deepdive.models import SearchQuery

_INSTRUCTIONS = """You are a research assistant. Given a user question, produce {n}
diverse search engine queries that would help answer it.

Rules:
- Each query should target a different angle or sub-question.
- Prefer specific over general. Include relevant technical terms.

Question: {question}"""


class _QueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1)


async def generate_queries(
    question: str, *, n: int = 5, llm: LLM | None = None
) -> list[SearchQuery]:
    """Generate N diverse search queries for a research question.

    Uses agentic-kit's ``LLM.extract`` for robust JSON parsing across any provider.
    """
    llm = llm or LLM()
    prompt = _INSTRUCTIONS.format(n=n, question=question)
    try:
        plan = await llm.extract(prompt, _QueryPlan, temperature=0.3)
        cleaned = [q.strip() for q in plan.queries if q.strip()][:n]
    except Exception:
        # Fall back to raw completion + line splitting on older/weaker models
        result = await llm.complete(prompt, temperature=0.3)
        cleaned = [
            line.strip("-*• 0123456789.") for line in result.content.splitlines() if line.strip()
        ][:n]
    return [SearchQuery(text=q) for q in cleaned if q]
