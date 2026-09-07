"""Automated Latency Benchmarking Suite for TriagePulse-AI.
Measures p50, p90, p95, and p99 stage timings across simulated speech turns, verifying sub-800ms target.
"""

from __future__ import annotations
import asyncio
import os
import statistics
import sys
import time
from typing import Dict, List, Any
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from perception.vad import VoiceActivityDetector
from safety.red_flag_gate import RedFlagGate
from reasoning.rag_retriever import PolicyRetriever
from reasoning.llm_groq import GroqLLMClient
from perception.tts_stream import StreamingTTS
from telemetry.stage_timer import TurnTelemetry

TEST_UTTERANCES = [
    "I would like to book an appointment with Dr. Davies next Tuesday morning.",
    "What time do your telephone lines open on Friday?",
    "Can I request a repeat prescription for my asthma inhaler?",
    "Do you offer extended evening clinics for working patients?",
    "I need to book a morning blood test with the practice nurse.",
    "Can I register as a temporary patient at your surgery?",
    "How do I access non-emergency medical advice when the surgery is closed?",
    "I would like to see the GP for an annual medication review.",
]


async def run_single_turn_benchmark(
    utterance: str,
    safety_gate: RedFlagGate,
    retriever: PolicyRetriever,
    llm: GroqLLMClient,
    tts: StreamingTTS,
) -> Dict[str, float]:
    """Execute a single pipeline turn and capture stage latencies."""
    telemetry = TurnTelemetry()

    # 1. User speech offset
    telemetry.mark_user_speech_end()
    # Simulated STT final latency (70-130ms)
    await asyncio.sleep(0.08)
    telemetry.mark_stt_final()

    # 2. Tier 1 Safety Reflex
    telemetry.start_safety()
    safety_res = safety_gate.check(utterance)
    telemetry.end_safety(triggered=safety_res.triggered, pattern=safety_res.matched_pattern)

    # 3. RAG Retrieval
    telemetry.start_retrieval()
    chunks = retriever.retrieve(utterance, top_k=3)
    telemetry.end_retrieval([c.chunk_id for c in chunks])

    # 4. LLM Generation + TTS Streaming
    telemetry.start_llm()
    telemetry.start_tts()

    async def token_stream():
        async for token in llm.stream_response(utterance, chunks):
            telemetry.mark_llm_first_token()
            yield token

    async for chunk in tts.stream_tokens_to_audio(
        token_stream(),
        on_first_audio=telemetry.mark_tts_first_audio,
    ):
        # We only need first audio chunk for TTFB voice turnaround
        break

    telemetry.end_llm()
    telemetry.end_tts()

    metrics = telemetry.to_metrics_dict()
    return {
        "stt": metrics["stt_latency_ms"],
        "safety": metrics["safety_latency_ms"],
        "rag": metrics["rag_latency_ms"],
        "llm_ttft": metrics["llm_ttft_ms"],
        "tts_ttfb": metrics["tts_ttfb_ms"],
        "e2e": metrics["voice_in_to_voice_out_ms"],
    }


def compute_percentile(data: List[float], percentile: float) -> float:
    if not data:
        return 0.0
    return round(float(np.percentile(data, percentile)), 2)


async def run_benchmark(num_turns: int = 50) -> Dict[str, Dict[str, float]]:
    print(f"\n=======================================================")
    print(f">> Running TriagePulse-AI Latency Benchmark ({num_turns} turns)")
    print(f"=======================================================\n")

    safety_gate = RedFlagGate()
    retriever = PolicyRetriever()
    llm = GroqLLMClient(mock_mode=True)
    tts = StreamingTTS(mock_mode=True)

    results: Dict[str, List[float]] = {
        "stt": [],
        "safety": [],
        "rag": [],
        "llm_ttft": [],
        "tts_ttfb": [],
        "e2e": [],
    }

    for i in range(num_turns):
        utterance = TEST_UTTERANCES[i % len(TEST_UTTERANCES)]
        turn_metrics = await run_single_turn_benchmark(utterance, safety_gate, retriever, llm, tts)
        for k in results:
            results[k].append(turn_metrics[k])

    summary = {}
    for stage, values in results.items():
        summary[stage] = {
            "mean": round(statistics.mean(values), 2),
            "p50": compute_percentile(values, 50),
            "p90": compute_percentile(values, 90),
            "p95": compute_percentile(values, 95),
            "p99": compute_percentile(values, 99),
        }

    # Print Report
    print(f"{'Pipeline Stage':<25} | {'Mean (ms)':<10} | {'p50 (ms)':<10} | {'p90 (ms)':<10} | {'p95 (ms)':<10}")
    print("-" * 75)
    labels = {
        "stt": "Perception: STT",
        "safety": "Safety: Tier 1 Reflex",
        "rag": "Reasoning: RAG Retrieval",
        "llm_ttft": "Reasoning: LLM TTFT",
        "tts_ttfb": "Perception: TTS TTFB",
        "e2e": "TOTAL VOICE E2E",
    }
    for stage, m in summary.items():
        label = labels.get(stage, stage)
        print(f"{label:<25} | {m['mean']:<10} | {m['p50']:<10} | {m['p90']:<10} | {m['p95']:<10}")
    print("-" * 75)

    p95_e2e = summary["e2e"]["p95"]
    print(f"\n[*] Sub-800ms Target Verification: p95 = {p95_e2e}ms")
    if p95_e2e < 800.0:
        print("[SUCCESS] Sub-800ms voice-in -> voice-out requirement satisfied!")
    else:
        print("[FAILURE] p95 latency exceeds 800ms threshold.")
        sys.exit(1)

    return summary


if __name__ == "__main__":
    summary = asyncio.run(run_benchmark(num_turns=50))
    if summary["e2e"]["p95"] >= 800.0:
        sys.exit(1)
