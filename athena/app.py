"""Athena web server: library, cited study chat, adaptive quizzes, progress."""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import ingest, llm, quiz, tutor
from .db import connect
from .retrieval import BM25Index

BASE = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("ATHENA_DATA", BASE / "data"))
DB = DATA / "athena.db"
UPLOADS = DATA / "uploads"
WEB = BASE / "web"

app = FastAPI(title="Athena")
_index: BM25Index | None = None
_lock = threading.Lock()


def get_index() -> BM25Index:
    global _index
    with _lock:
        if _index is None:
            _index = ingest.build_index(DB)
        return _index


def invalidate_index() -> None:
    global _index
    with _lock:
        _index = None


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _retopic_chunks(doc_id: int) -> None:
    """Ask the LLM for clean per-chunk topic labels (batched, one call)."""
    import json as _json
    conn = connect(DB)
    try:
        rows = conn.execute("SELECT id, text FROM chunks WHERE doc_id=? ORDER BY ord",
                            (doc_id,)).fetchall()
        if not rows:
            return
        numbered = "\n".join(f"[{r[0]}] {r[1][:300]}" for r in rows)
        raw = llm.complete([
            {"role": "system", "content": (
                "Label each numbered passage with a short topic (2-4 words, "
                "lowercase, e.g. 'photosynthesis', 'fundamental rights'). "
                "Reply with ONLY a JSON object mapping id to topic.")},
            {"role": "user", "content": numbered}], temperature=0.2, max_tokens=600)
        m = raw[raw.find("{"):raw.rfind("}") + 1]
        labels = _json.loads(m)
        for r in rows:
            label = str(labels.get(str(r[0]), "")).strip().lower()[:40]
            if label:
                conn.execute("UPDATE chunks SET topic=? WHERE id=?", (label, r[0]))
        conn.commit()
    except Exception:  # noqa: BLE001
        pass
    finally:
        conn.close()


# ---------------------------------------------------------------- pages/static
@app.get("/")
def home():
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/api/status")
def status():
    conn = connect(DB)
    try:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        atts = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()[0]
        due = len(quiz.due_reviews(conn))
    finally:
        conn.close()
    return {"docs": docs, "chunks": chunks, "attempts": atts, "due_reviews": due,
            "llm": llm.configured(), "backend": llm.backend_label()}


# ---------------------------------------------------------------- library
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), title: str = Form(""),
                 caption: str = Form("")):
    UPLOADS.mkdir(parents=True, exist_ok=True)
    dest = UPLOADS / file.filename
    i = 1
    while dest.exists():
        dest = UPLOADS / f"{Path(file.filename).stem}-{i}{Path(file.filename).suffix}"
        i += 1
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    ext = dest.suffix.lower()
    try:
        if ext in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            doc_id = ingest.ingest_image(DB, dest, title or None, caption)
        else:
            doc_id = ingest.ingest_file(DB, dest, title or None)
    except ValueError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, str(exc))
    if llm.configured() and ext not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        try:
            _retopic_chunks(doc_id)
        except Exception:  # noqa: BLE001
            pass
    invalidate_index()
    return {"doc_id": doc_id}


@app.get("/api/documents")
def documents():
    conn = connect(DB)
    try:
        rows = conn.execute(
            "SELECT d.id, d.title, d.filename, d.kind, COUNT(c.id) AS chunks "
            "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
            "GROUP BY d.id ORDER BY d.created_at DESC").fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int):
    conn = connect(DB)
    try:
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))
        conn.commit()
    finally:
        conn.close()
    invalidate_index()
    return {"ok": True}


# ---------------------------------------------------------------- study chat
@app.post("/api/chat")
async def chat(body: dict):
    question = (body.get("question") or "").strip()
    if not question:
        raise HTTPException(400, "Empty question.")
    index = get_index()
    passages = index.search(question, top_k=6)

    def gen():
        import time
        yield sse("sources", {"passages": [
            {"n": i + 1, "doc": p.doc_title, "text": p.text[:400]}
            for i, p in enumerate(passages)]})
        if not llm.configured():
            text, cited = tutor.retrieval_only(passages)
        else:
            try:
                text, cited = tutor.answer(question, passages)
            except Exception as exc:  # noqa: BLE001
                yield sse("error", {"message": f"LLM error: {exc}"})
                return
        # Paced delivery for a live typing feel; text is verbatim the model's.
        words = text.split(" ")
        for i in range(0, len(words), 6):
            yield sse("delta", {"text": (" " if i else "") + " ".join(words[i:i + 6])})
            time.sleep(0.03)
        yield sse("done", {"cited": [
            {"n": passages.index(p) + 1, "doc": p.doc_title, "text": p.text}
            for p in cited]})

    return StreamingResponse(gen(), media_type="text/event-stream")


# ---------------------------------------------------------------- adaptive quiz
def _pick_topic(conn, want: str | None) -> str:
    if want:
        return want
    stats = quiz.topic_stats(conn)
    if not stats:
        # derive topics from chunks
        rows = conn.execute("SELECT DISTINCT topic FROM chunks").fetchall()
        topics = [r[0] for r in rows]
        if not topics:
            raise HTTPException(400, "Upload study material first.")
        import random
        return random.choice(topics)
    # weight toward weakest topics
    import random
    inv = [1.0 / max(1, s["rating"] - 700) for s in stats]
    total = sum(inv)
    r = random.random() * total
    for s, w in zip(stats, inv):
        r -= w
        if r <= 0:
            return s["topic"]
    return stats[0]["topic"]


@app.post("/api/quiz/next")
async def quiz_next(body: dict):
    topic = (body.get("topic") or "").strip() or None
    conn = connect(DB)
    try:
        # serve due reviews first
        due = quiz.due_reviews(conn, limit=5)
        if due:
            q = quiz.load_question(conn, due[0])
            if q:
                return _public_question(q, review=True)
        topic = _pick_topic(conn, topic)
        rating = quiz.get_rating(conn, topic)
        index = get_index()
        passages = index.search(topic.replace("-", " "), top_k=8)
        if not passages:
            raise HTTPException(400, "No material for this topic yet.")
        use_llm = llm.configured()
        try:
            if use_llm:
                obj = quiz.generate_mcq(passages, topic, rating)
                mode, qtype = "llm", "mcq"
            else:
                raise RuntimeError("no llm")
        except Exception:  # noqa: BLE001
            obj = quiz.generate_cloze(passages)
            if obj is None:
                raise HTTPException(500, "Could not generate a question.")
            use_llm = False
            mode, qtype = "template", "mcq"
        difficulty = max(800, min(2000, int(rating + (0 if use_llm else -100))))
        qid = quiz.store_question(conn, topic, difficulty, qtype, obj, passages, mode)
        q = quiz.load_question(conn, qid)
        return _public_question(q, review=False)
    finally:
        conn.close()


def _public_question(q: quiz.Question, review: bool) -> dict:
    return {"id": q.id, "topic": q.topic, "difficulty": q.difficulty,
            "prompt": q.prompt, "choices": q.choices, "mode": q.mode,
            "review": review}


@app.post("/api/quiz/answer")
async def quiz_answer(body: dict):
    qid = body.get("question_id")
    given = body.get("given", "")
    conn = connect(DB)
    try:
        q = quiz.load_question(conn, qid)
        if not q:
            raise HTTPException(404, "Unknown question.")
        correct = quiz.grade(q, given)
        new_rating = quiz.update_rating(conn, q.topic, correct, q.difficulty)
        quiz.schedule_review(conn, q.id, correct)
        conn.execute("INSERT INTO attempts (question_id, correct, created_at)"
                     " VALUES (?,?,?)", (q.id, int(correct), __import__("time").time()))
        conn.commit()
        rows = conn.execute("SELECT text FROM chunks WHERE id IN (%s)" %
                            ",".join("?" * len(q.source_chunk_ids)),
                            q.source_chunk_ids).fetchall() if q.source_chunk_ids else []
        return {"correct": correct, "explanation": q.explanation,
                "rating": round(new_rating),
                "sources": [r[0][:300] for r in rows]}
    finally:
        conn.close()


@app.get("/api/topics")
def topics():
    conn = connect(DB)
    try:
        return quiz.topic_stats(conn)
    finally:
        conn.close()


@app.get("/api/quiz/due")
def quiz_due():
    conn = connect(DB)
    try:
        return {"due": len(quiz.due_reviews(conn))}
    finally:
        conn.close()
