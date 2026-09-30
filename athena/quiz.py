"""Adaptive assessment engine.

- Per-topic Elo ratings (start 1200). Correct answers raise the rating,
  wrong answers lower it; next questions target difficulty near the rating.
- Leitner boxes (1..5) schedule reviews: box n is due after 2^(n-1) days.
- Question generation: LLM-grounded MCQs with citations; template-mode
  cloze questions when no LLM key is configured (clearly labeled).
"""
from __future__ import annotations

import json
import math
import random
import re
import time
from dataclasses import dataclass

from . import llm
from .db import connect, now
from .retrieval import ScoredChunk

K_FACTOR = 32
MIN_RATING, MAX_RATING = 800, 2000

QUESTION_SYSTEM = (
    "You generate quiz questions for a student. Given source passages, write ONE "
    "multiple-choice question as strict JSON: "
    '{{"prompt": "...", "choices": ["A...", "B...", "C...", "D..."], '
    '"answer": 0-3 (index of correct choice), "explanation": "why, citing [n]"}} '
    "Ground everything in the passages; do not invent facts. "
    "Target difficulty: {difficulty} (easy<1000, medium~1200, hard>1400). "
    "Output ONLY the JSON object."
)


@dataclass
class Question:
    id: int
    topic: str
    difficulty: int
    qtype: str
    prompt: str
    choices: list[str] | None
    answer: str
    explanation: str
    source_chunk_ids: list[int]
    mode: str


# ---------------------------------------------------------------- ratings

def get_rating(conn, topic: str) -> float:
    row = conn.execute("SELECT rating FROM topic_ratings WHERE topic=?", (topic,)).fetchone()
    return float(row[0]) if row else 1200.0


def update_rating(conn, topic: str, correct: bool, question_difficulty: int) -> float:
    rating = get_rating(conn, topic)
    expected = 1.0 / (1.0 + 10 ** ((question_difficulty - rating) / 400.0))
    actual = 1.0 if correct else 0.0
    new_rating = max(MIN_RATING, min(MAX_RATING, rating + K_FACTOR * (actual - expected)))
    row = conn.execute("SELECT attempts FROM topic_ratings WHERE topic=?", (topic,)).fetchone()
    attempts = (row[0] if row else 0) + 1
    conn.execute(
        "INSERT INTO topic_ratings (topic, rating, attempts) VALUES (?,?,?) "
        "ON CONFLICT(topic) DO UPDATE SET rating=excluded.rating, attempts=excluded.attempts",
        (topic, new_rating, attempts))
    conn.commit()
    return new_rating


def topic_stats(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT topic, rating, attempts FROM topic_ratings ORDER BY rating ASC").fetchall()
    return [{"topic": r[0], "rating": round(r[1]), "attempts": r[2]} for r in rows]


# ---------------------------------------------------------------- reviews (Leitner)

def schedule_review(conn, question_id: int, correct: bool) -> None:
    row = conn.execute("SELECT box FROM reviews WHERE question_id=?", (question_id,)).fetchone()
    box = row[0] if row else 1
    box = min(5, box + 1) if correct else 1
    due = now() + (2 ** (box - 1)) * 86400
    conn.execute(
        "INSERT INTO reviews (question_id, box, due_at) VALUES (?,?,?) "
        "ON CONFLICT(question_id) DO UPDATE SET box=excluded.box, due_at=excluded.due_at",
        (question_id, box, due))
    conn.commit()


def due_reviews(conn, limit: int = 10) -> list[int]:
    rows = conn.execute(
        "SELECT question_id FROM reviews WHERE due_at <= ? ORDER BY due_at ASC LIMIT ?",
        (now(), limit)).fetchall()
    return [r[0] for r in rows]


# ---------------------------------------------------------------- generation

def _difficulty_label(rating: float) -> str:
    if rating < 1050:
        return "easy"
    if rating < 1350:
        return "medium"
    return "hard"


def generate_mcq(passages: list[ScoredChunk], topic: str, rating: float) -> dict:
    numbered = "\n\n".join(
        f"[{i+1}] {p.text}" for i, p in enumerate(passages))
    messages = [
        {"role": "system", "content": QUESTION_SYSTEM.format(difficulty=_difficulty_label(rating))},
        {"role": "user", "content": f"Passages:\n{numbered}\n\nTopic: {topic}"},
    ]
    raw = llm.complete(messages, temperature=0.6, max_tokens=800)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("LLM did not return JSON.")
    obj = json.loads(m.group(0))
    assert isinstance(obj["choices"], list) and len(obj["choices"]) == 4
    assert 0 <= int(obj["answer"]) <= 3
    return obj


def generate_cloze(passages: list[ScoredChunk]) -> dict | None:
    """Template-mode question: blank out a key term from a factual sentence."""
    for p in passages:
        sentences = re.split(r"(?<=[.!?])\s+", p.text)
        for s in sentences:
            words = re.findall(r"[A-Za-z][A-Za-z\-]{4,}", s)
            if len(words) >= 8:
                target = max(words, key=len)
                prompt = s.replace(target, "_____", 1)
                distractors = [w for w in words if w != target][:3]
                if len(distractors) < 3:
                    continue
                choices = [target] + distractors
                random.shuffle(choices)
                return {
                    "prompt": f"Fill in the blank:\n{prompt}",
                    "choices": choices,
                    "answer": choices.index(target),
                    "explanation": f"The sentence comes from your notes: “{s[:160]}…”",
                }
    return None


def store_question(conn, topic: str, difficulty: int, qtype: str, obj: dict,
                   passages: list[ScoredChunk], mode: str) -> int:
    cur = conn.execute(
        "INSERT INTO questions (topic, difficulty, qtype, prompt, choices, answer,"
        " explanation, source_chunk_ids, mode, created_at)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (topic, difficulty, qtype, obj["prompt"], json.dumps(obj.get("choices")),
         str(obj["answer"]), obj.get("explanation", ""),
         json.dumps([p.chunk_id for p in passages]), mode, now()))
    conn.commit()
    return cur.lastrowid


def load_question(conn, qid: int) -> Question | None:
    r = conn.execute("SELECT * FROM questions WHERE id=?", (qid,)).fetchone()
    if not r:
        return None
    return Question(id=r["id"], topic=r["topic"], difficulty=r["difficulty"],
                    qtype=r["qtype"], prompt=r["prompt"],
                    choices=json.loads(r["choices"]) if r["choices"] else None,
                    answer=r["answer"], explanation=r["explanation"],
                    source_chunk_ids=json.loads(r["source_chunk_ids"]), mode=r["mode"])


def grade(question: Question, given: str) -> bool:
    given = (given or "").strip()
    if question.qtype == "mcq" and question.choices:
        # accept index ("2"), letter ("C"), or the choice text
        if given.isdigit() and int(given) == int(question.answer):
            return True
        letters = "ABCD"
        if len(given) == 1 and given.upper() in letters:
            return int(question.answer) == letters.index(given.upper())
        try:
            return question.choices[int(question.answer)].strip().lower() == given.lower()
        except (IndexError, ValueError):
            return False
    return given.lower() in question.answer.lower() or question.answer.lower() in given.lower()
