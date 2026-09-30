"""Pure-Python BM25 retrieval over indexed chunks. No dependencies."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

_TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = frozenset(
    "a an the and or but of to in on for with as at by from is are was were be been "
    "being it its this that these those he she they we you i my his her their our your "
    "not no yes do does did will would can could should shall may might must have has "
    "had having what when where which who whom how why if then than so such into over "
    "under between through during".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class ScoredChunk:
    chunk_id: int
    doc_id: int
    text: str
    topic: str
    doc_title: str
    score: float


class BM25Index:
    """In-memory BM25 built from rows of (chunk_id, doc_id, tokens, text, topic, doc_title)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []
        self.df: dict[str, int] = {}
        self.avgdl = 0.0

    def add(self, chunk_id: int, doc_id: int, tokens: list[str], text: str,
            topic: str, doc_title: str) -> None:
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            self.df[t] = self.df.get(t, 0) + 1
        self.docs.append({
            "chunk_id": chunk_id, "doc_id": doc_id, "tf": tf,
            "len": len(tokens), "text": text, "topic": topic,
            "doc_title": doc_title,
        })

    def finalize(self) -> None:
        n = len(self.docs)
        self.avgdl = sum(d["len"] for d in self.docs) / n if n else 0.0
        self._n = n

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 6) -> list[ScoredChunk]:
        qterms = tokenize(query)
        if not qterms or not self.docs:
            return []
        scored: list[ScoredChunk] = []
        for d in self.docs:
            score = 0.0
            dl = d["len"] or 1
            for t in qterms:
                tf = d["tf"].get(t, 0)
                if not tf:
                    continue
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                score += self.idf(t) * tf * (self.k1 + 1) / denom
            if score > 0:
                scored.append(ScoredChunk(
                    chunk_id=d["chunk_id"], doc_id=d["doc_id"], text=d["text"],
                    topic=d["topic"], doc_title=d["doc_title"], score=score))
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]
