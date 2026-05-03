from __future__ import annotations

from actants import LLM

from deepdive.models import Citation, Claim, ReportSection, ResearchReport

_SUMMARY_PROMPT = """Write a one-paragraph summary (3-5 sentences) that
answers the question using the claims below.

Question: {question}

Claims:
{claims}

Summary:"""

_SECTION_PROMPT = """Write a {heading} section for a research report.
Use the claims below as factual basis.

Rules:
- 2-4 short paragraphs
- Use inline citations in the form [n] referencing the numbered claims
- Do not invent facts

Question: {question}
Section: {heading}

Claims:
{claims}

Section body:"""

_NO_CLAIMS_MESSAGE = (
    "No verifiable claims could be extracted from the searched sources. "
    "This can happen when search returned no results, when pages failed to "
    "scrape, or when the local model could not parse the content. Try "
    "rephrasing the question, switching search backend, or using a larger model."
)


class ReportBuilder:
    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm or LLM()

    async def build(self, question: str, claims: list[Claim]) -> ResearchReport:
        if not claims:
            return ResearchReport(
                question=question,
                summary=_NO_CLAIMS_MESSAGE,
                sections=[],
                sources=[],
                total_cost_usd=0.0,
            )

        numbered = [f"[{i + 1}] {c.text}" for i, c in enumerate(claims)]
        claims_block = "\n".join(numbered)

        total_cost = 0.0

        summary_result = await self.llm.complete(
            _SUMMARY_PROMPT.format(question=question, claims=claims_block),
            temperature=0.3,
        )
        total_cost += summary_result.cost_usd

        sections: list[ReportSection] = []
        for heading in ("Background", "Key findings", "Contradictions & open questions"):
            r = await self.llm.complete(
                _SECTION_PROMPT.format(question=question, heading=heading, claims=claims_block),
                temperature=0.3,
            )
            total_cost += r.cost_usd
            sections.append(ReportSection(heading=heading, body=r.content, claims=claims))

        sources = self._collect_sources(claims)

        return ResearchReport(
            question=question,
            summary=summary_result.content.strip(),
            sections=sections,
            sources=sources,
            total_cost_usd=total_cost,
        )

    @staticmethod
    def _collect_sources(claims: list[Claim]) -> list[Citation]:
        seen: dict[str, Citation] = {}
        for claim in claims:
            for c in claim.citations:
                seen.setdefault(str(c.url), c)
        return list(seen.values())
