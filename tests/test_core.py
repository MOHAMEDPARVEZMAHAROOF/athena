"""Tests for Athena core: retrieval, chunking, adaptive engine, ingest."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from athena import ingest, quiz
from athena.retrieval import BM25Index, tokenize


def test_tokenize_drops_stopwords():
    toks = tokenize("The quick brown fox jumps over the lazy dog")
    assert "the" not in toks and "over" not in toks
    assert "quick" in toks and "fox" in toks


def test_bm25_ranks_relevant_chunk_first():
    idx = BM25Index()
    docs = [
        "photosynthesis converts light energy into glucose in chloroplasts",
        "the constitution guarantees fundamental rights to citizens",
        "photosynthesis releases oxygen through photolysis of water",
    ]
    for i, d in enumerate(docs):
        idx.add(i, 0, tokenize(d), d, "general", "doc")
    idx.finalize()
    res = idx.search("how do plants release oxygen", top_k=3)
    assert res, "expected results"
    assert "oxygen" in res[0].text


def test_bm25_empty_query():
    idx = BM25Index()
    idx.add(0, 0, tokenize("hello world"), "hello world", "g", "d")
    idx.finalize()
    assert idx.search("", top_k=3) == []
    assert idx.search("zzzqqq", top_k=3) == []


def test_chunking_sentence_aware():
    text = "First sentence here. " * 40
    chunks = ingest.chunk_text(text, size=120, overlap=20)
    assert len(chunks) > 1
    assert all(c.strip() for c in chunks)


def test_ingest_roundtrip(tmp_path):
    src = tmp_path / "notes.md"
    src.write_text("# Test\n\nMitochondria produce ATP through cellular respiration. " * 10)
    db = tmp_path / "t.db"
    doc_id = ingest.ingest_file(db, src, title="Test Doc")
    assert doc_id == 1
    index = ingest.build_index(db)
    res = index.search("what produces ATP", top_k=3)
    assert res and res[0].doc_title == "Test Doc"


def test_elo_moves_right_direction(tmp_path):
    import sqlite3
    from athena.db import connect
    conn = connect(tmp_path / "t.db")
    r0 = quiz.get_rating(conn, "biology")
    assert r0 == 1200.0
    r1 = quiz.update_rating(conn, "biology", True, 1200)
    assert r1 > r0
    r2 = quiz.update_rating(conn, "biology", False, 2000)
    assert r2 < r1
    conn.close()


def test_leitner_scheduling(tmp_path):
    from athena.db import connect
    conn = connect(tmp_path / "t.db")
    qid = conn.execute(
        "INSERT INTO questions (topic,difficulty,qtype,prompt,answer,explanation,"
        "source_chunk_ids,mode,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        ("biology", 1200, "mcq", "Q?", "0", "E", "[]", "template", 0.0)).lastrowid
    conn.commit()
    assert quiz.due_reviews(conn) == []
    quiz.schedule_review(conn, qid, False)  # wrong -> box 1 -> due in 1 day
    assert quiz.due_reviews(conn) == []
    # force due
    conn.execute("UPDATE reviews SET due_at=0 WHERE question_id=?", (qid,))
    conn.commit()
    assert quiz.due_reviews(conn) == [qid]
    quiz.schedule_review(conn, qid, True)  # box 2 -> due in 2 days
    row = conn.execute("SELECT box FROM reviews WHERE question_id=?", (qid,)).fetchone()
    assert row[0] == 2
    conn.close()


def test_grade_mcq_variants(tmp_path):
    from athena.quiz import Question, grade
    q = Question(id=1, topic="t", difficulty=1200, qtype="mcq", prompt="p",
                 choices=["Alpha", "Beta", "Gamma", "Delta"], answer="1",
                 explanation="e", source_chunk_ids=[], mode="template")
    assert grade(q, "1") and grade(q, "B") and grade(q, "b") and grade(q, "Beta")
    assert not grade(q, "0") and not grade(q, "Gamma")


def test_cloze_generation():
    from athena.retrieval import ScoredChunk
    p = ScoredChunk(chunk_id=1, doc_id=1, topic="t", doc_title="d", score=1.0,
                    text="Chlorophyll absorbs light energy mainly in the red and blue "
                         "wavelengths, while reflecting green light back to our eyes.")
    obj = quiz.generate_cloze([p])
    assert obj is not None
    assert "_____" in obj["prompt"]
    assert len(obj["choices"]) == 4
