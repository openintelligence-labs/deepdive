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


# Sentence boundary heuristic — good enough for prose, not perfect for code.
# We cut at sentence-ending punctuation followed by whitespace + capital letter.
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
    """
    if not text or not text.strip():
        return []
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    cur_text: list[str] = []
    cur_start: int | None = None
    cur_end: int = 0

    def flush() -> None:
        nonlocal cur_text, cur_start, cur_end
        if cur_text and cur_start is not None:
            joined = " ".join(cur_text)
            chunks.append(Chunk(text=joined, offset_start=cur_start, offset_end=cur_end))
        cur_text = []
        cur_start = None
        cur_end = 0

    for sent, start in sentences:
        sent_len = len(sent)
        if cur_start is None:
            cur_start = start
            cur_end = start + sent_len
            cur_text.append(sent)
            continue
        # Would this sentence overflow the chunk?
        prospective = cur_end - cur_start + 1 + sent_len
        if prospective > max_chars and cur_text:
            flush()
            # Start a new chunk; back-track ``overlap_chars`` worth of text by
            # re-including the tail of the prior chunk's last sentence.
            tail_start = max(0, start - overlap_chars)
            cur_start = tail_start
            cur_end = start + sent_len
            # Pull the prefix between tail_start and start into the new chunk
            prefix = text[tail_start:start].strip()
            if prefix:
                cur_text.append(prefix)
            cur_text.append(sent)
        else:
            cur_text.append(sent)
            cur_end = start + sent_len
    flush()
    return chunks
