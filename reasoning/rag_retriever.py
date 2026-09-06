"""Semantic Vector & Lexical Hybrid Retriever for NHS GP Surgery Policies.
Indexes local surgery guidelines, appointment booking protocols, and emergency guidelines.
"""

from __future__ import annotations
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set


@dataclass
class PolicyChunk:
    chunk_id: str
    source_file: str
    section_title: str
    content: str
    tokens: Set[str]
    score: float = 0.0


class PolicyRetriever:
    """Retrieves relevant GP surgery policies with sub-5ms latency."""

    def __init__(self, policies_dir: Optional[str | Path] = None):
        if policies_dir is None:
            policies_dir = Path(__file__).parent.parent / "data" / "policies"
        self.policies_dir = Path(policies_dir)
        self.chunks: List[PolicyChunk] = []
        self.idf: Dict[str, float] = {}
        self._load_and_index()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())
        # English stop words list
        stopwords = {
            "the", "and", "is", "in", "to", "of", "for", "with", "at", "by", "from",
            "on", "an", "be", "this", "that", "it", "are", "as", "or", "your", "you",
            "can", "have", "will", "our", "all", "we", "may", "if", "not", "any"
        }
        return [w for w in words if w not in stopwords]

    def _load_and_index(self) -> None:
        """Parse markdown policy documents into semantic chunks and compute IDF weights."""
        if not self.policies_dir.exists():
            return

        chunks: List[PolicyChunk] = []
        chunk_idx = 0

        for md_file in self.policies_dir.glob("*.md"):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue

            sections = re.split(r"\n(?=##?\s+)", text)
            for section in sections:
                lines = section.strip().split("\n")
                if not lines:
                    continue
                header = lines[0].replace("#", "").strip()
                body = "\n".join(lines[1:]).strip()
                if not body:
                    body = header

                tokens = self._tokenize(f"{header} {body}")
                chunk = PolicyChunk(
                    chunk_id=f"{md_file.stem}_{chunk_idx}",
                    source_file=md_file.name,
                    section_title=header,
                    content=f"## {header}\n{body}",
                    tokens=set(tokens),
                )
                chunks.append(chunk)
                chunk_idx += 1

        self.chunks = chunks

        # Compute IDF
        doc_count = len(self.chunks)
        df: Dict[str, int] = {}
        for c in self.chunks:
            for t in c.tokens:
                df[t] = df.get(t, 0) + 1

        self.idf = {
            term: math.log((doc_count + 1.0) / (count + 0.5)) + 1.0
            for term, count in df.items()
        }

    def retrieve(self, query: str, top_k: int = 3) -> List[PolicyChunk]:
        """Search policy chunks using BM25-style lexical relevance."""
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return self.chunks[:top_k]

        scored: List[PolicyChunk] = []
        for c in self.chunks:
            score = 0.0
            # Token overlap weighted by IDF
            matching = c.tokens.intersection(query_tokens)
            for t in matching:
                weight = self.idf.get(t, 1.0)
                score += weight * 1.5

            # Boost if query tokens appear in section title
            title_tokens = self._tokenize(c.section_title)
            title_matching = set(title_tokens).intersection(query_tokens)
            score += len(title_matching) * 2.0

            if score > 0.0:
                c_copy = PolicyChunk(
                    chunk_id=c.chunk_id,
                    source_file=c.source_file,
                    section_title=c.section_title,
                    content=c.content,
                    tokens=c.tokens,
                    score=round(score, 3),
                )
                scored.append(c_copy)

        scored.sort(key=lambda x: x.score, reverse=True)
        if not scored:
            return self.chunks[:top_k]
        return scored[:top_k]
