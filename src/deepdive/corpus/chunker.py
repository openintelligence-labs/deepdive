"""Token-aware chunking with character-offset preservation.

Each chunk records the byte offsets back into the source document so the
span-grounding validator can verify excerpts against the original file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    offset_start: int  # char offset in the source document
    offset_end: int


# Sentence-ending punctuation followed by whitespace + a capital. A heuristic:
# good enough for prose, imprecise on code.
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


def _split_sentences(text: str) -> list[tuple[str, int]]:
    """Return [(sentence, start_offset)]. Offsets are into ``text``."""
    out: list[tuple[str, int]] = []
    start = 0
    for m in _SENT_RE.finditer(text):
        end = m.start()
        sent = text[start:end].strip()
        if sent:
            actual_start = start + (len(text[start:end]) - len(text[start:end].lstrip()))
            out.append((sent, actual_start))
        start = m.end()
    tail = text[start:].strip()
    if tail:
        actual_start = start + (len(text[start:]) - len(text[start:].lstrip()))
        out.append((tail, actual_start))
    return out


def chunk_text(
    text: str,
    *,
    max_chars: int = 1500,
    overlap_chars: int = 200,
) -> list[Chunk]:
    """Split ``text`` into chunks of at most ``max_chars`` with sentence-aligned cuts.

    Each chunk overlaps the previous by ``overlap_chars`` so a relevant claim
    near a chunk boundary is still findable. Both bounds are character-based;
    we don't tokenize because the embedder will do that downstream.

    Invariant: every chunk's ``text`` equals ``source_text[offset_start:offset_end]``.
    Span-grounding relies on this — without it, an excerpt match against the
    chunk text wouldn't translate back to a slice of the original document.
    """
    if not text or not text.strip():
        return []
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    cur_start: int | None = None
    cur_end: int = 0

    def flush() -> None:
        nonlocal cur_start, cur_end
        if cur_start is not None and cur_end > cur_start:
            # Chunk text is the literal slice of the source — keeps the
            # `text == source[start:end]` invariant intact even after overlap.
            chunks.append(
                Chunk(
                    text=text[cur_start:cur_end],
                    offset_start=cur_start,
                    offset_end=cur_end,
                )
            )
        cur_start = None
        cur_end = 0

    for sent, start in sentences:
        sent_len = len(sent)
        sent_end = start + sent_len
        if cur_start is None:
            cur_start = start
            cur_end = sent_end
            continue
        prospective = sent_end - cur_start
        if prospective > max_chars:
            flush()
            # Start a new chunk; back-track ``overlap_chars`` worth of text so
            # claims near the boundary are still findable in this chunk too.
            tail_start = max(0, start - overlap_chars)
            cur_start = tail_start
            cur_end = sent_end
        else:
            cur_end = sent_end
    flush()
    return chunks
