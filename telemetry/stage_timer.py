"""High-precision stage timing instrumentation for TriagePulse-AI.
Tracks voice-in to voice-out latency across perception, safety, and reasoning layers.
"""

from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TurnTelemetry:
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_speech_end_ns: Optional[int] = None
    stt_final_ns: Optional[int] = None
    safety_check_start_ns: Optional[int] = None
    safety_check_end_ns: Optional[int] = None
    retrieval_start_ns: Optional[int] = None
    retrieval_end_ns: Optional[int] = None
    llm_start_ns: Optional[int] = None
    llm_first_token_ns: Optional[int] = None
    llm_end_ns: Optional[int] = None
    tts_start_ns: Optional[int] = None
    tts_first_audio_ns: Optional[int] = None
    tts_end_ns: Optional[int] = None

    # Metadata & contextual audit flags
    safety_triggered: bool = False
    safety_matched_pattern: Optional[str] = None
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    grounding_confidence: float = 1.0
    booking_extracted: bool = False
    booking_details: Optional[Dict[str, Any]] = None

    def mark_user_speech_end(self) -> None:
        self.user_speech_end_ns = time.perf_counter_ns()

    def mark_stt_final(self) -> None:
        self.stt_final_ns = time.perf_counter_ns()

    def start_safety(self) -> None:
        self.safety_check_start_ns = time.perf_counter_ns()

    def end_safety(self, triggered: bool = False, pattern: Optional[str] = None) -> None:
        self.safety_check_end_ns = time.perf_counter_ns()
        self.safety_triggered = triggered
        self.safety_matched_pattern = pattern

    def start_retrieval(self) -> None:
        self.retrieval_start_ns = time.perf_counter_ns()

    def end_retrieval(self, chunk_ids: Optional[list[str]] = None) -> None:
        self.retrieval_end_ns = time.perf_counter_ns()
        if chunk_ids:
            self.retrieved_chunk_ids = chunk_ids

    def start_llm(self) -> None:
        self.llm_start_ns = time.perf_counter_ns()

    def mark_llm_first_token(self) -> None:
        if self.llm_first_token_ns is None:
            self.llm_first_token_ns = time.perf_counter_ns()

    def end_llm(self) -> None:
        self.llm_end_ns = time.perf_counter_ns()

    def start_tts(self) -> None:
        self.tts_start_ns = time.perf_counter_ns()

    def mark_tts_first_audio(self) -> None:
        if self.tts_first_audio_ns is None:
            self.tts_first_audio_ns = time.perf_counter_ns()

    def end_tts(self) -> None:
        self.tts_end_ns = time.perf_counter_ns()

    @staticmethod
    def _ns_to_ms(start_ns: Optional[int], end_ns: Optional[int]) -> Optional[float]:
        if start_ns is None or end_ns is None:
            return None
        return round((end_ns - start_ns) / 1_000_000.0, 2)

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert recorded nanosecond timestamps into actionable stage latency metrics (in ms)."""
        stt_latency = self._ns_to_ms(self.user_speech_end_ns, self.stt_final_ns)
        safety_latency = self._ns_to_ms(self.safety_check_start_ns, self.safety_check_end_ns)
        rag_latency = self._ns_to_ms(self.retrieval_start_ns, self.retrieval_end_ns)
        llm_ttft = self._ns_to_ms(self.llm_start_ns, self.llm_first_token_ns)
        llm_total = self._ns_to_ms(self.llm_start_ns, self.llm_end_ns)
        tts_ttfb = self._ns_to_ms(self.tts_start_ns, self.tts_first_audio_ns)
        tts_total = self._ns_to_ms(self.tts_start_ns, self.tts_end_ns)

        # Total voice-in to voice-out: from end of user speech to first byte of audio output
        voice_in_to_voice_out = self._ns_to_ms(self.user_speech_end_ns, self.tts_first_audio_ns)

        return {
            "turn_id": self.turn_id,
            "voice_in_to_voice_out_ms": voice_in_to_voice_out,
            "stt_latency_ms": stt_latency,
            "safety_latency_ms": safety_latency,
            "rag_latency_ms": rag_latency,
            "llm_ttft_ms": llm_ttft,
            "llm_total_ms": llm_total,
            "tts_ttfb_ms": tts_ttfb,
            "tts_total_ms": tts_total,
            "safety_triggered": self.safety_triggered,
            "safety_matched_pattern": self.safety_matched_pattern,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "grounding_confidence": self.grounding_confidence,
            "booking_extracted": self.booking_extracted,
            "booking_details": self.booking_details,
            "meets_target_sub_800ms": (voice_in_to_voice_out is not None and voice_in_to_voice_out < 800.0),
        }
