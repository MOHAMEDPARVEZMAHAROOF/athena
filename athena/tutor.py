"""Study companion: retrieval-augmented answers with inline [n] citations.

Every factual claim the model makes must cite a retrieved passage; the API
returns the answer plus the cited sources so the UI can show them verbatim.
"""
from __future__ import annotations

import re

from . import llm
from .retrieval import ScoredChunk

SYSTEM = (
    "You are Athena, a precise study tutor. Answer ONLY from the provided source "
    "passages. Every factual statement must end with a citation like [1] or [2] "
    "referring to the passage numbers. If the passages do not contain the answer, "
    "say so plainly and do not invent facts. Keep answers focused and student-friendly. "
    "Write citations with plain ASCII square brackets exactly like [1], [2]."
)

CIT_RE = re.compile(r"[\[【](\d+)[\]】]")


def build_messages(question: str, passages: list[ScoredChunk]) -> list[dict]:
    numbered = "\n\n".join(
        f"[{i+1}] (from '{p.doc_title}') {p.text}" for i, p in enumerate(passages))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Source passages:\n{numbered}\n\nQuestion: {question}"},
    ]


def extract_cited(passages: list[ScoredChunk], answer: str) -> list[ScoredChunk]:
    cited: list[ScoredChunk] = []
    for m in CIT_RE.finditer(answer):
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(passages) and passages[idx] not in cited:
            cited.append(passages[idx])
    return cited


def answer(question: str, passages: list[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
    """Non-streaming answer. Returns (answer_text, cited_passages)."""
    if not passages:
        return ("I couldn't find anything about that in your study materials. "
                "Try uploading more notes or rephrasing the question."), []
    text = llm.complete(build_messages(question, passages))
    return text, extract_cited(passages, text)


def answer_stream(question: str, passages: list[ScoredChunk]):
    """Yield (delta, done, cited) tuples for SSE streaming."""
    if not passages:
        msg = ("I couldn't find anything about that in your study materials. "
               "Try uploading more notes or rephrasing the question.")
        yield ("delta", msg)
        yield ("done", msg, [])
        return
    messages = build_messages(question, passages)
    full: list[str] = []
    for delta in llm.complete_stream(messages):
        full.append(delta)
        yield ("delta", delta)
    text = "".join(full)
    yield ("done", text, extract_cited(passages, text))


def retrieval_only(passages: list[ScoredChunk]) -> tuple[str, list[ScoredChunk]]:
    """Honest fallback when no LLM key is configured: show top passages verbatim."""
    if not passages:
        return ("No matching passages in your library."), []
    lines = [f"[{i+1}] {p.text}\n— {p.doc_title}" for i, p in enumerate(passages[:5])]
    return ("AI answering is unavailable (no API key configured). "
            "Here are the most relevant passages from your materials:\n\n"
            + "\n\n".join(lines)), passages[:5]
