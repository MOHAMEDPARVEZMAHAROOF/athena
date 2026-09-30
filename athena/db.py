"""SQLite persistence for Athena. Standard library only."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    kind TEXT NOT NULL,            -- text | image
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord INTEGER NOT NULL,
    text TEXT NOT NULL,
    topic TEXT NOT NULL DEFAULT 'general',
    tokens TEXT NOT NULL           -- space-joined normalized tokens for BM25
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    difficulty INTEGER NOT NULL,   -- 800..2000 Elo scale
    qtype TEXT NOT NULL,           -- mcq | cloze | short
    prompt TEXT NOT NULL,
    choices TEXT,                  -- JSON array or NULL
    answer TEXT NOT NULL,          -- correct choice index (mcq) or text
    explanation TEXT NOT NULL,
    source_chunk_ids TEXT NOT NULL,-- JSON array of chunk ids
    mode TEXT NOT NULL,            -- llm | template
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id INTEGER NOT NULL REFERENCES questions(id),
    correct INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS topic_ratings (
    topic TEXT PRIMARY KEY,
    rating REAL NOT NULL DEFAULT 1200,
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS reviews (
    question_id INTEGER PRIMARY KEY REFERENCES questions(id),
    box INTEGER NOT NULL DEFAULT 1,   -- Leitner box 1..5
    due_at REAL NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def now() -> float:
    return time.time()
