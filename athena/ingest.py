"""Document ingestion: text extraction, chunking, topic tagging, BM25 indexing."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .db import connect, now
from .retrieval import BM25Index, tokenize

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120
SUPPORTED_TEXT_EXTS = {".txt", ".md", ".markdown"}


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in SUPPORTED_TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="replace")
    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError(
                "PDF support needs the 'pypdf' package (pip install pypdf).") from exc
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    raise ValueError(f"Unsupported file type: {ext} (use .txt, .md, or .pdf)")


def chunk_text(text: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer to break at sentence end.
        if end < len(text):
            cut = text.rfind(". ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        start = max(end - overlap, start + 1)
        if end >= len(text):
            break
    return chunks


def simple_topic(text: str, fallback: str) -> str:
    """Cheap fallback topic: most frequent content word bigram-ish signal.

    Real topic labels come from the LLM tagger at ingest; this keeps the app
    functional with zero API access."""
    toks = tokenize(text)
    if not toks:
        return fallback
    freq: dict[str, int] = {}
    for t in toks:
        if len(t) > 3:
            freq[t] = freq.get(t, 0) + 1
    top = sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:2]
    label = "-".join(w for w, _ in top) or fallback
    return label[:40]


def ingest_file(db_path: str | Path, src: Path, title: str | None = None,
                tagger=None) -> int:
    """Ingest a text document. Returns the document id.

    tagger: optional callable(list[str]) -> list[str] returning one topic per chunk.
    """
    src = Path(src)
    text = extract_text(src)
    pieces = chunk_text(text)
    if not pieces:
        raise ValueError("No readable text found in file.")
    title = title or src.stem.replace("_", " ").replace("-", " ").strip() or "Untitled"
    topics = tagger(pieces) if tagger else [simple_topic(p, "general") for p in pieces]
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO documents (title, filename, kind, created_at) VALUES (?,?,?,?)",
            (title, src.name, "text", now()))
        doc_id = cur.lastrowid
        for i, (piece, topic) in enumerate(zip(pieces, topics)):
            conn.execute(
                "INSERT INTO chunks (doc_id, ord, text, topic, tokens) VALUES (?,?,?,?,?)",
                (doc_id, i, piece, topic, " ".join(tokenize(piece))))
        conn.commit()
    finally:
        conn.close()
    return doc_id


def ingest_image(db_path: str | Path, src: Path, title: str | None = None,
                 caption: str = "") -> int:
    """Register an image (diagram/photo) as study material with a caption."""
    import shutil
    src = Path(src)
    title = title or src.stem.replace("_", " ").strip() or "Untitled image"
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO documents (title, filename, kind, created_at) VALUES (?,?,?,?)",
            (title, src.name, "image", now()))
        doc_id = cur.lastrowid
        text = f"[Diagram: {title}] {caption}".strip()
        conn.execute(
            "INSERT INTO chunks (doc_id, ord, text, topic, tokens) VALUES (?,?,?,?,?)",
            (doc_id, 0, text, simple_topic(text, "diagrams"), " ".join(tokenize(text))))
        conn.commit()
    finally:
        conn.close()
    return doc_id


def build_index(db_path: str | Path) -> BM25Index:
    conn = connect(db_path)
    try:
        rows = conn.execute(
            "SELECT c.id, c.doc_id, c.tokens, c.text, c.topic, d.title "
            "FROM chunks c JOIN documents d ON d.id = c.doc_id").fetchall()
    finally:
        conn.close()
    index = BM25Index()
    for r in rows:
        index.add(r[0], r[1], r[2].split(), r[3], r[4], r[5])
    index.finalize()
    return index
