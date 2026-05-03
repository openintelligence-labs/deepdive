"""Span-grounded claim extraction and validation.

A grounded claim records the exact substring of the source page that supports
it. The validator re-checks every excerpt by substring search on the original
fetched body — if the excerpt isn't there verbatim (or near-verbatim after a
small whitespace normalization), the citation is marked ``grounded=False``
and the claim is dropped from the report by default.

This is the wedge: every other deep-research tool publishes "citation 3"
without proving the cited text actually says what we claim it says. We do.
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from deepdive.models import Citation, Claim


class _GroundedClaim(BaseModel):
    """LLM-facing schema: each claim ships with the verbatim excerpt that supports it."""

    text: str
    excerpt: str = Field(default="", description="Verbatim substring from the source")


class _GroundedClaimList(BaseModel):
    claims: list[_GroundedClaim] = Field(default_factory=list)


GROUNDING_INSTRUCTIONS = """\
Extract the {max_claims} most important verifiable factual claims from the text.

For EVERY claim, you MUST return BOTH fields:
  - ``text``: a self-contained factual statement (1 sentence)
  - ``excerpt``: a SHORT verbatim substring (1-3 sentences) copied EXACTLY from
    the SOURCE TEXT — character-for-character, including spaces and punctuation.
    Do NOT paraphrase. Do NOT add ellipses or brackets. The excerpt must be
    findable in the source via plain string search.

Skip any claim whose supporting excerpt you cannot copy verbatim. Returning
fewer high-quality grounded claims is BETTER than padding with ungrounded ones.

EXAMPLE
-------
SOURCE TEXT: "The Apollo 11 mission landed on the Moon on July 20, 1969. Neil
Armstrong was the first person to step onto the lunar surface."

Correct response:
{{
  "claims": [
    {{
      "text": "Apollo 11 landed on the Moon on July 20, 1969.",
      "excerpt": "The Apollo 11 mission landed on the Moon on July 20, 1969."
    }},
    {{
      "text": "Neil Armstrong was the first person to walk on the Moon.",
      "excerpt": "Neil Armstrong was the first person to step onto the lunar surface."
    }}
  ]
}}

SOURCE TEXT:
{text}
"""


def _normalize_for_match(s: str) -> str:
    """Collapse whitespace + Unicode-normalize so minor formatting doesn't break matching.

    LLMs sometimes emit excerpts with single spaces where the source has tabs
    or newlines, or NFKD-decomposed quotes. We accept those.
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def find_excerpt_offsets(body: str, excerpt: str) -> tuple[int, int] | None:
    """Locate ``excerpt`` in ``body``. Tries exact match, then whitespace-and-case
    normalized match.

    Returns (start, end) char offsets into ``body``, or None if not found.
    Normalized search uses a sliding-window strategy: look for the normalized
    excerpt as a substring of the normalized body, then map back to original
    coordinates by walking the body.
    """
    if not excerpt or not body:
        return None

    # Fast path: exact substring match.
    idx = body.find(excerpt)
    if idx >= 0:
        return (idx, idx + len(excerpt))

    # Fallback: whitespace + Unicode + case normalized. LLMs sometimes emit
    # excerpts with different capitalization than the source (e.g. title-cased
    # "The Moon" vs body "the moon" after lowercasing in HTML rendering).
    # Casefolding is correct for cross-locale matching (handles Turkish ß etc.).
    norm_excerpt = _normalize_for_match(excerpt).casefold()
    if not norm_excerpt:
        return None

    # Build a "normalized body" + a mapping from normalized index → original index.
    # Each char in the original body either contributes one char (after collapsing)
    # to the normalized body, or is squashed (whitespace runs).
    norm_chars: list[str] = []
    norm_to_orig: list[int] = []
    in_ws = False
    for i, ch in enumerate(body):
        nch = unicodedata.normalize("NFKC", ch).casefold()
        for sub in nch:
            if sub.isspace():
                if not in_ws:
                    norm_chars.append(" ")
                    norm_to_orig.append(i)
                    in_ws = True
            else:
                norm_chars.append(sub)
                norm_to_orig.append(i)
                in_ws = False
    norm_body = "".join(norm_chars).strip()
    # Recompute offsets after strip — track leading-strip count.
    leading = len("".join(norm_chars)) - len("".join(norm_chars).lstrip())
    if leading:
        norm_to_orig = norm_to_orig[leading:]

    j = norm_body.find(norm_excerpt)
    if j < 0:
        return None
    if j >= len(norm_to_orig):
        return None
    start = norm_to_orig[j]
    end_idx = j + len(norm_excerpt) - 1
    if end_idx >= len(norm_to_orig):
        return None
    end = norm_to_orig[end_idx] + 1
    return (start, end)


def ground_citation(body: str, citation: Citation) -> Citation:
    """Try to ground a citation against the source body. Returns a NEW Citation."""
    if not citation.excerpt:
        return citation.model_copy(update={"grounded": False})
    offsets = find_excerpt_offsets(body, citation.excerpt)
    if offsets is None:
        return citation.model_copy(update={"grounded": False})
    start, end = offsets
    return citation.model_copy(
        update={
            "offset_start": start,
            "offset_end": end,
            "grounded": True,
        }
    )


def ground_claim(body: str, claim: Claim) -> Claim:
    """Re-validate every citation on a claim against the source body."""
    new_citations = [ground_citation(body, c) for c in claim.citations]
    return claim.model_copy(update={"citations": new_citations})


def filter_grounded(claims: list[Claim], *, include_ungrounded: bool = False) -> list[Claim]:
    """Drop claims with no grounded citation by default."""
    if include_ungrounded:
        return claims
    return [c for c in claims if c.grounded]
