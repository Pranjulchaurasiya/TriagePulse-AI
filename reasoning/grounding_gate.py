"""Grounding & Auditor Gate for Hallucination Prevention.
Audits generated LLM responses against retrieved surgery policies and medical safety rules.
Rejects ungrounded clinical advice and falls back safely to human receptionist handoff.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Optional
from .rag_retriever import PolicyChunk


@dataclass
class GroundingResult:
    is_grounded: bool
    confidence_score: float
    violations: List[str]
    sanitized_response: str
    fallback_applied: bool = False


class GroundingGate:
    """Verifies factual alignment between LLM generation and retrieved GP policies."""

    SAFE_FALLBACK_HANDOFF = (
        "I want to make sure you receive the most accurate and safe information. "
        "Please hold while I transfer you directly to our reception team at St. Jude Medical Centre."
    )

    # Disallowed clinical prescriptions or hallucinated medical advice
    UNAUTHORIZED_CLINICAL_PATTERNS = [
        re.compile(r"\b(take|prescribe|administer)\s+\d+\s*(mg|milligram|tablets?|capsules?|ml)\b", re.IGNORECASE),
        re.compile(r"\b(diagnos(e|ed|is)|you have (cancer|pneumonia|diabetes|a fracture))\b", re.IGNORECASE),
        re.compile(r"\b(stop taking your|increase your dose of)\b", re.IGNORECASE),
        re.compile(r"\b(open on (saturday|sunday)|open 24 hours)\b", re.IGNORECASE),
    ]

    def __init__(self, min_confidence_threshold: float = 0.65):
        self.min_confidence_threshold = min_confidence_threshold

    def audit(
        self,
        llm_response: str,
        retrieved_chunks: List[PolicyChunk],
    ) -> GroundingResult:
        """Inspect generated response for policy violations or clinical hallucinations."""
        violations: List[str] = []

        # 1. Check for unauthorized medical prescription / diagnosis claims
        for pattern in self.UNAUTHORIZED_CLINICAL_PATTERNS:
            match = pattern.search(llm_response)
            if match:
                violations.append(f"Unauthorized clinical/medical assertion detected: '{match.group(0)}'")

        # 2. Check semantic overlap with retrieved policy chunks
        combined_policy_text = " ".join([c.content.lower() for c in retrieved_chunks])
        response_words = re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", llm_response.lower())

        # Filter out common conversational words
        stop_words = {
            "the", "and", "you", "for", "with", "that", "this", "have", "are", "our",
            "can", "will", "please", "thank", "hello", "would", "like", "sure", "help",
            "medical", "centre", "jude", "calling", "today", "assist", "your", "could"
        }
        content_words = [w for w in response_words if w not in stop_words]

        grounded_count = 0
        if content_words:
            for w in content_words:
                if w in combined_policy_text:
                    grounded_count += 1
            confidence_score = round(grounded_count / len(content_words), 2)
        else:
            confidence_score = 1.0

        if confidence_score < 0.2 and len(content_words) > 5:
            violations.append(f"Response lacks sufficient policy grounding (overlap: {confidence_score})")

        # 3. Decision
        if violations:
            return GroundingResult(
                is_grounded=False,
                confidence_score=confidence_score,
                violations=violations,
                sanitized_response=self.SAFE_FALLBACK_HANDOFF,
                fallback_applied=True,
            )

        return GroundingResult(
            is_grounded=True,
            confidence_score=confidence_score,
            violations=[],
            sanitized_response=llm_response,
            fallback_applied=False,
        )
