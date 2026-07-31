"""sqlite-vec corpus indexer.

Extracts text from PDF/MD/HTML/TXT, chunks it, embeds each chunk via Ollama
(default ``nomic-embed-text``), and stores everything in a sqlite database
with a vec0 virtual table for cosine search.

Table layout:
  documents(doc_id INTEGER PRIMARY KEY, source_path TEXT UNIQUE, indexed_at TEXT)
  chunks(chunk_id INTEGER PRIMARY KEY, doc_id INTEGER REFERENCES documents,
         text TEXT, offset_start INT, offset_end INT)
  vec_chunks(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[N])  -- vec0 virtual

The indexer is async because the embedder is. Indexing is incremental — files
already in ``documents`` are skipped unless their mtime is newer.
"""

from __future__ import annotations

import os
import sqlite3
import struct
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import sqlite_vec
from actants import Embeddings

from deepdive.corpus.chunker import chunk_text


@dataclass
class IndexHit:
    text: str
    source_path: str
    offset_start: int
    offset_end: int
    distance: float


def extract_text(path: Path) -> str:
    """Extract plain text from PDF / Markdown / HTML / plain text.

    Returns an empty string for unrecognized formats. Formatting is not
    preserved; the goal is searchable text.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "PDF indexing requires `pypdf`. Install with `pip install pypdf`."
            ) from exc
        reader = PdfReader(str(path))
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    if suffix in {".md", ".markdown", ".txt", ".rst"}:
        return path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm"}:
        try:
            from selectolax.parser import HTMLParser
        except ImportError:
            return path.read_text(encoding="utf-8", errors="replace")
        tree = HTMLParser(path.read_text(encoding="utf-8", errors="replace"))
        for tag in tree.css("script, style, nav, header, footer"):
            tag.decompose()
        body = tree.body
        return body.text(separator=" ", strip=True) if body else ""
    return ""


def _pack(vector: list[float]) -> bytes:
    """Pack a float vector as little-endian f32 — sqlite-vec's wire format."""
    return struct.pack(f"{len(vector)}f", *vector)


class CorpusIndex:
    """Async sqlite-vec corpus index. Use as an async context manager."""

    def __init__(
        self,
        db_path: str | Path,
        *,
        embeddings: Embeddings | None = None,
        embed_model: str = "nomic-embed-text",
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings = embeddings or Embeddings(model=embed_model)
        self.embed_model = embed_model
        self._dim: int | None = None  # set on first embed
        self._conn: sqlite3.Connection | None = None

    async def __aenter__(self) -> CorpusIndex:
        self._conn = self._open()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = self._open()
        return self._conn

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS documents ("
            " doc_id INTEGER PRIMARY KEY,"
            " source_path TEXT UNIQUE NOT NULL,"
            " indexed_at TEXT NOT NULL,"
            " mtime REAL NOT NULL"
            ")"
        )
        conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks ("
            " chunk_id INTEGER PRIMARY KEY,"
            " doc_id INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,"
            " text TEXT NOT NULL,"
            " offset_start INTEGER NOT NULL,"
            " offset_end INTEGER NOT NULL"
            ")"
        )
        return conn

    async def _ensure_vec_table(self, dim: int) -> None:
        """Create the vec0 virtual table once the embedder dimension is known."""
        if self._dim is not None:
            return
        # ``dim`` is f-stringed into SQL below, so coerce explicitly: a non-int
        # must raise here rather than reach the statement.
        safe_dim = int(dim)
        if safe_dim <= 0:
            raise ValueError(f"embedding dimension must be positive, got {dim!r}")
        self._dim = safe_dim
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"chunk_id INTEGER PRIMARY KEY, embedding FLOAT[{safe_dim}])"
        )

    async def index_file(self, path: str | Path) -> int:
        """Index one file. Returns the number of chunks added (0 if up-to-date)."""
        path = Path(path).resolve()
        if not path.exists() or not path.is_file():
            return 0
        text = extract_text(path)
        if not text.strip():
            return 0

        mtime = path.stat().st_mtime
        cur = self.conn.execute(
            "SELECT doc_id, mtime FROM documents WHERE source_path = ?", (str(path),)
        )
        row = cur.fetchone()
        if row is not None and row[1] >= mtime:
            return 0  # up-to-date
        if row is not None:
            # vec_chunks is a virtual table, so FK cascade does not apply; it
            # must be cleaned explicitly before the chunks rows are dropped.
            old_chunk_ids = [
                r[0]
                for r in self.conn.execute(
                    "SELECT chunk_id FROM chunks WHERE doc_id = ?", (row[0],)
                )
            ]
            for chunk_id in old_chunk_ids:
                self.conn.execute("DELETE FROM vec_chunks WHERE chunk_id = ?", (chunk_id,))
            self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (row[0],))
            self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (row[0],))

        chunks = chunk_text(text)
        if not chunks:
            return 0

        embed_result = await self.embeddings.embed([c.text for c in chunks])
        vectors = embed_result.vectors
        if not vectors:
            return 0
        await self._ensure_vec_table(len(vectors[0]))

        self.conn.execute(
            "INSERT INTO documents (source_path, indexed_at, mtime) VALUES (?, ?, ?)",
            (str(path), datetime.now(UTC).isoformat(), mtime),
        )
        doc_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        for chunk, vec in zip(chunks, vectors, strict=True):
            cur = self.conn.execute(
                "INSERT INTO chunks (doc_id, text, offset_start, offset_end) VALUES (?, ?, ?, ?)",
                (doc_id, chunk.text, chunk.offset_start, chunk.offset_end),
            )
            chunk_id = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            self.conn.execute(
                "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, _pack(vec)),
            )
        self.conn.commit()
        return len(chunks)

    async def index_directory(
        self,
        root: str | Path,
        *,
        extensions: tuple[str, ...] = (".pdf", ".md", ".markdown", ".txt", ".rst", ".html", ".htm"),
    ) -> AsyncIterator[tuple[Path, int]]:
        """Walk ``root`` and index every file with a supported extension.

        Yields ``(path, chunks_added)`` per file so callers can show progress.
        """
        root = Path(root).resolve()
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if not name.lower().endswith(extensions):
                    continue
                fpath = Path(dirpath) / name
                added = await self.index_file(fpath)
                yield fpath, added

    async def search(self, query: str, *, k: int = 5) -> list[IndexHit]:
        """Cosine-search the corpus. Returns the top-k hits (smaller distance = closer)."""
        if self._dim is None:
            # No embeddings yet — try to detect from existing vec table
            cur = self.conn.execute("SELECT name FROM sqlite_master WHERE name = 'vec_chunks'")
            if cur.fetchone() is None:
                return []
        embed_result = await self.embeddings.embed([query])
        if not embed_result.vectors:
            return []
        qvec = _pack(embed_result.vectors[0])
        rows = self.conn.execute(
            "SELECT vc.chunk_id, c.text, d.source_path, c.offset_start, c.offset_end, vc.distance "
            "FROM vec_chunks AS vc "
            "JOIN chunks AS c ON c.chunk_id = vc.chunk_id "
            "JOIN documents AS d ON d.doc_id = c.doc_id "
            "WHERE vc.embedding MATCH ? AND k = ? "
            "ORDER BY vc.distance",
            (qvec, k),
        ).fetchall()
        return [
            IndexHit(
                text=row[1],
                source_path=row[2],
                offset_start=row[3],
                offset_end=row[4],
                distance=row[5],
            )
            for row in rows
        ]

    def stats(self) -> dict[str, int]:
        n_docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_chunks = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {"documents": n_docs, "chunks": n_chunks}
