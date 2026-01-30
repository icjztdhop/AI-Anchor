# txt_knowledge.py
from __future__ import annotations

"""Lightweight TXT knowledge retrieval (offline, deterministic).

Design goals
- Works well for Chinese TXT FAQs (e.g. "营业时间：每天 10:00 - 22:00").
- Prefer exact/keyword matches first; embedding similarity is only a fallback.
- No llama-index / OpenAI dependency.

Usage
- Put your knowledge in knowledge.txt (UTF-8).
- Call retrieve(query). If hits:
    - Use answer_from_hits(query, hits) to build a deterministic grounded snippet.
  Else:
    - Fall back to normal LLM.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import os
import math
import re


def _normalize(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("：", ":")
    # remove common punctuation and spaces
    s = re.sub(r"[\s\t\r\n]+", "", s)
    s = re.sub(r"[，。！？!?,.;；、】【()（）\[\]{}<>《》\"'“”‘’]", "", s)
    return s


def _char_ngrams(text: str, n: int) -> List[str]:
    if not text:
        return []
    if len(text) <= n:
        return [text]
    return [text[i : i + n] for i in range(0, len(text) - n + 1)]


def _hash_embed(text: str, dim: int = 512) -> List[float]:
    """A tiny, offline embedding using character n-grams.

    This is not as good as real embeddings, but it's stable and works reasonably
    for short Chinese queries without spaces.
    """

    t = _normalize(text)
    if not t:
        return [0.0] * dim

    vec = [0.0] * dim
    grams: List[str] = []
    grams.extend(_char_ngrams(t, 2))
    grams.extend(_char_ngrams(t, 3))

    for g in grams:
        # FNV-1a
        h = 2166136261
        for ch in g:
            h ^= ord(ch)
            h = (h * 16777619) & 0xFFFFFFFF
        vec[h % dim] += 1.0

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 0.0
    nb = math.sqrt(sum(y * y for y in b)) or 0.0
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class KnowledgeHit:
    text: str
    score: float
    kind: str = "similarity"  # "exact" | "keyword" | "similarity"


class TxtKnowledgeBase:
    def __init__(
        self,
        txt_path: str,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        dim: int = 512,
    ) -> None:
        self.txt_path = txt_path
        self.chunk_size = max(200, int(chunk_size))
        self.chunk_overlap = max(0, int(chunk_overlap))
        self.dim = int(dim)

        self.enabled = False
        self._chunks: List[str] = []
        self._embs: List[List[float]] = []

        # Key-value like lines: "营业时间: 每天 10:00-22:00"
        self._kv: List[Tuple[str, str]] = []  # (key, full_line)
        self._keys_norm: List[str] = []

        self.reload()

    def reload(self) -> bool:
        if not self.txt_path or not os.path.exists(self.txt_path):
            self.enabled = False
            self._chunks, self._embs, self._kv, self._keys_norm = [], [], [], []
            return False

        text = open(self.txt_path, "r", encoding="utf-8").read().strip()
        if not text:
            self.enabled = False
            self._chunks, self._embs, self._kv, self._keys_norm = [], [], [], []
            return False

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

        # Parse kv lines
        kv: List[Tuple[str, str]] = []
        for ln in lines:
            ln2 = ln.replace("：", ":")
            if ":" in ln2:
                left = ln2.split(":", 1)[0].strip()
                if left:
                    kv.append((left, ln.strip()))

        self._kv = kv
        self._keys_norm = [_normalize(k) for k, _ in kv]

        # Build chunks for general retrieval
        joined = "\n".join(lines)
        chunks: List[str] = []
        i = 0
        while i < len(joined):
            chunk = joined[i : i + self.chunk_size].strip()
            if chunk:
                chunks.append(chunk)
            if i + self.chunk_size >= len(joined):
                break
            i += max(1, self.chunk_size - self.chunk_overlap)

        self._chunks = chunks
        self._embs = [_hash_embed(c, self.dim) for c in chunks]
        self.enabled = True
        return True

    def detect_topics(self, query: str) -> List[str]:
        """Return KB topic keys that appear in the query.

        Useful for guardrails: if user asks about a known topic (like 营业时间) but
        retrieval fails, we can avoid hallucination.
        """
        qn = _normalize(query)
        if not qn:
            return []
        topics: List[str] = []
        for (key, _line), kn in zip(self._kv, self._keys_norm):
            if kn and kn in qn:
                topics.append(key)
        return topics

    def _kv_match(self, query: str) -> List[KnowledgeHit]:
        """Strong matching for key-value FAQ lines."""
        if not self._kv:
            return []
        qn = _normalize(query)
        if not qn:
            return []

        hits: List[KnowledgeHit] = []

        # 1) direct key inclusion: if query contains the normalized key
        for (key, line), kn in zip(self._kv, self._keys_norm):
            if not kn:
                continue
            if kn in qn:
                hits.append(KnowledgeHit(text=line, score=1.0, kind="exact"))

        if hits:
            return hits

        # 2) keyword overlap scoring (use 2-gram overlap between query and key)
        q2 = set(_char_ngrams(qn, 2))
        best: List[Tuple[float, str]] = []
        for (key, line), kn in zip(self._kv, self._keys_norm):
            if not kn:
                continue
            k2 = set(_char_ngrams(kn, 2))
            inter = len(q2 & k2)
            if inter <= 0:
                continue
            score = inter / max(1, len(k2))
            best.append((score, line))
        best.sort(key=lambda x: x[0], reverse=True)
        for s, line in best[:3]:
            if s >= 0.25:
                hits.append(KnowledgeHit(text=line, score=float(s), kind="keyword"))
        return hits

    def retrieve(self, query: str, top_k: int = 4, score_threshold: float = 0.18) -> List[KnowledgeHit]:
        """Retrieve relevant knowledge chunks.

        Order:
        - kv match first (high precision)
        - similarity match as fallback
        """
        if not self.enabled or not (query or "").strip():
            return []

        # High-precision FAQ matching
        kv_hits = self._kv_match(query)
        if kv_hits:
            return kv_hits[: max(1, int(top_k))]

        # Similarity fallback
        qv = _hash_embed(query, self.dim)
        scored = [(_cosine(qv, e), c) for c, e in zip(self._chunks, self._embs)]
        scored.sort(key=lambda x: x[0], reverse=True)

        hits: List[KnowledgeHit] = []
        for s, c in scored[: max(1, int(top_k))]:
            if float(s) >= float(score_threshold):
                hits.append(KnowledgeHit(text=c, score=float(s), kind="similarity"))
        return hits

    def answer_from_hits(self, query: str, hits: List[KnowledgeHit], max_chars: int = 260) -> str:
        """Deterministic grounded snippet.

        - If hit is a kv line, return that line directly.
        - Else choose lines within the chunk that share 2-gram overlap with query.
        """
        if not hits:
            return ""

        best_text = (hits[0].text or "").strip().replace("\r", "")
        if not best_text:
            return ""

        # If it's a KV line (contains colon), keep it as-is.
        if ":" in best_text or "：" in best_text:
            out = best_text
        else:
            qn = _normalize(query)
            lines = [ln.strip() for ln in best_text.splitlines() if ln.strip()]
            if len(lines) <= 1 or not qn:
                out = best_text
            else:
                q2 = set(_char_ngrams(qn, 2))
                picked: List[Tuple[int, str]] = []
                for ln in lines:
                    ln_n = _normalize(ln)
                    l2 = set(_char_ngrams(ln_n, 2))
                    inter = len(q2 & l2)
                    if inter > 0:
                        picked.append((inter, ln))
                picked.sort(key=lambda x: x[0], reverse=True)
                out_lines = [ln for _s, ln in picked[:3]] or lines[:2]
                out = "\n".join(out_lines)

        out = out.strip()
        if len(out) > max_chars:
            out = out[: max_chars - 1].rstrip() + "…"
        return out
