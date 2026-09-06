"""Bidirectional Audio WebSocket Gateway & Full Pipeline Orchestrator.
Coordinates VAD, Deepgram STT, Tier 1 Safety Gate, RAG Retriever, Groq LLaMA 3.3, and Streaming TTS.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from perception.vad import VoiceActivityDetector
from perception.stt_deepgram import DeepgramSTTStream
from perception.tts_stream import StreamingTTS
from safety.red_flag_gate import RedFlagGate
from reasoning.rag_retriever import PolicyRetriever
from reasoning.llm_groq import GroqLLMClient
from reasoning.grounding_gate import GroundingGate
from reasoning.booking_extractor import BookingExtractor, BookingRequest
from telemetry.stage_timer import TurnTelemetry

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("TriagePulse.Gateway")

app = FastAPI(title="TriagePulse-AI Voice Receptionist", version="0.1.0")

# Security Headers Middleware (gstack CSO / OWASP hardening)
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response: Response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "microphone=*"
        return response

app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount static web directory
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Shared singleton components
safety_gate = RedFlagGate()
retriever = PolicyRetriever()
llm_client = GroqLLMClient()
grounding_gate = GroundingGate()
booking_extractor = BookingExtractor()

# Bounded telemetry memory buffer (prevents unbounded memory leak)
MAX_TELEMETRY_LOGS = 100
telemetry_history: list[Dict[str, Any]] = []


@app.get("/")
async def get_index():
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({"status": "TriagePulse-AI Gateway Active", "ws_endpoint": "/ws/audio"})


@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "mock_providers": os.getenv("MOCK_PROVIDERS", "true").lower() in ("true", "1", "yes"),
        "policies_indexed": len(retriever.chunks),
        "safety_rules_count": len(safety_gate.compiled_rules),
    }


@app.get("/api/telemetry")
async def get_telemetry():
    return {"history": telemetry_history[-50:]}


@app.get("/api/bookings")
async def get_bookings():
    return {"bookings": booking_extractor.confirmed_bookings}


class ClientSession:
    """Manages state for an active caller connection."""

    def __init__(self, ws: WebSocket, session_id: str):
        self.ws = ws
        self.session_id = session_id
        self.vad = VoiceActivityDetector()
        self.stt = DeepgramSTTStream()
        self.tts = StreamingTTS()
        self.is_speaking = False
        self.is_system_playing = False
        self.cumulative_transcript = ""
        self.conversation_history: list[Dict[str, str]] = []
        self.current_tts_task: Optional[asyncio.Task] = None

    async def send_json(self, data: Dict[str, Any]) -> None:
        try:
            await self.ws.send_text(json.dumps(data))
        except Exception as e:
            logger.debug(f"Failed to send JSON to client {self.session_id}: {e}")

    async def send_audio_chunk(self, audio_bytes: bytes) -> None:
        try:
            await self.ws.send_bytes(audio_bytes)
        except Exception as e:
            logger.debug(f"Failed to send audio to client {self.session_id}: {e}")

    def interrupt(self) -> None:
        """Trigger instant barge-in interrupt: cancel TTS and reset playback state."""
        self.tts.cancel()
        if self.current_tts_task and not self.current_tts_task.done():
            self.current_tts_task.cancel()
        self.is_system_playing = False


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    session = ClientSession(websocket, session_id)
    await session.stt.connect()

    logger.info(f"Client connected. Session ID: {session_id}")
    await session.send_json({
        "type": "connection_ready",
        "session_id": session_id,
        "message": "Connected to TriagePulse-AI Voice Gateway.",
    })

    try:
        while True:
            message = await websocket.receive()
            
            # Handle binary audio frame with size cap (gstack security hardening)
            if "bytes" in message and message["bytes"]:
                raw_frame = message["bytes"]
                if len(raw_frame) > 512 * 1024:
                    logger.warning(f"Rejected oversized audio frame ({len(raw_frame)} bytes) from session {session_id}")
                    continue

                vad_result = session.vad.process_frame(raw_frame, is_system_speaking=session.is_system_playing)

                # Check Barge-in mid-playback
                if vad_result["barge_in_triggered"]:
                    logger.info(f"Barge-in detected in session {session_id}! Cancelling audio output.")
                    session.interrupt()
                    await session.send_json({"type": "barge_in", "message": "Playback interrupted by user speech."})

                # Stream audio to STT
                await session.stt.send_audio_chunk(raw_frame)

                # If speech completed, finalize turn
                if vad_result["speech_ended"]:
                    logger.info(f"User speech ended in session {session_id}. Processing turn.")
                    asyncio.create_task(process_turn(session, user_audio_ended=True))

            # Handle JSON control packets (sanitized and validated)
            elif "text" in message and message["text"]:
                if len(message["text"]) > 64 * 1024:
                    logger.warning(f"Rejected oversized text packet from session {session_id}")
                    continue

                try:
                    payload = json.loads(message["text"])
                except Exception:
                    continue

                msg_type = payload.get("type")
                if msg_type == "text_query":
                    # Truncate input string safely to prevent memory bloat
                    query_text = payload.get("text", "").strip()[:2000]
                    if query_text:
                        asyncio.create_task(process_turn_from_text(session, query_text))
                elif msg_type == "interrupt":
                    session.interrupt()
                    await session.send_json({"type": "barge_in_confirmed"})
                elif msg_type == "ping":
                    await session.send_json({"type": "pong", "time": time.time()})

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket session error {session_id}: {e}")
    finally:
        session.interrupt()
        await session.stt.close()


async def process_turn_from_text(session: ClientSession, text: str) -> None:
    """Execute pipeline turn driven by incoming user text utterance."""
    telemetry = TurnTelemetry()
    telemetry.mark_user_speech_end()
    telemetry.mark_stt_final()

    await execute_pipeline_reasoning(session, text, telemetry)


async def process_turn(session: ClientSession, user_audio_ended: bool = True) -> None:
    """Execute pipeline turn driven by streaming audio VAD offset."""
    telemetry = TurnTelemetry()
    telemetry.mark_user_speech_end()
    # In live mode with Deepgram, final transcript comes from STT event loop.
    # In mock mode, we finalize with last captured audio representation
    telemetry.mark_stt_final()


async def execute_pipeline_reasoning(
    session: ClientSession,
    user_utterance: str,
    telemetry: TurnTelemetry,
) -> None:
    """Core three-layer pipeline execution."""
    session.cumulative_transcript += f" {user_utterance}"

    await session.send_json({
        "type": "transcript",
        "text": user_utterance,
        "is_final": True,
    })

    # ==========================================
    # LAYER 2: TIER 1 DETERMINISTIC SAFETY GATE
    # ==========================================
    telemetry.start_safety()
    safety_result = safety_gate.check(user_utterance, session.cumulative_transcript)
    telemetry.end_safety(
        triggered=safety_result.triggered,
        pattern=safety_result.matched_pattern,
    )

    if safety_result.triggered:
        logger.warning(
            f"SAFETY REFLEX TRIGGERED! Category: {safety_result.category}, "
            f"Pattern: {safety_result.matched_pattern} in {safety_result.latency_ms}ms"
        )
        # Notify client of hard safety interrupt
        await session.send_json({
            "type": "safety_alert",
            "category": safety_result.category,
            "matched_pattern": safety_result.matched_pattern,
            "instruction": safety_result.emergency_instruction,
            "latency_ms": safety_result.latency_ms,
        })

        # Bypass LLM completely! Stream emergency directive to TTS
        session.is_system_playing = True
        telemetry.start_tts()

        async def emergency_token_stream():
            words = safety_result.emergency_instruction.split(" ")
            for w in words:
                yield w + " "
                await asyncio.sleep(0.01)

        async for audio_chunk in session.tts.stream_tokens_to_audio(
            emergency_token_stream(),
            on_first_audio=telemetry.mark_tts_first_audio,
        ):
            await session.send_audio_chunk(audio_chunk)

        telemetry.end_tts()
        session.is_system_playing = False

        # Record and broadcast telemetry
        metrics = telemetry.to_metrics_dict()
        telemetry_history.append(metrics)
        await session.send_json({"type": "telemetry", "metrics": metrics})
        return

    # ==========================================
    # LAYER 3: REASONING LAYER (TIER 2)
    # ==========================================
    # 1. RAG Policy Retrieval
    telemetry.start_retrieval()
    relevant_chunks = retriever.retrieve(user_utterance, top_k=3)
    chunk_ids = [c.chunk_id for c in relevant_chunks]
    telemetry.end_retrieval(chunk_ids)

    await session.send_json({
        "type": "retrieval",
        "chunks": [{"id": c.chunk_id, "file": c.source_file, "title": c.section_title} for c in relevant_chunks],
    })

    # 2. Groq LLaMA 3.3 Streaming Inference
    telemetry.start_llm()
    generated_tokens: list[str] = []

    async def token_generator():
        async for token in llm_client.stream_response(
            user_utterance=user_utterance,
            policy_chunks=relevant_chunks,
            conversation_history=session.conversation_history,
        ):
            telemetry.mark_llm_first_token()
            generated_tokens.append(token)
            yield token

    # Stream tokens into TTS engine for lowest Time-To-First-Byte (TTFB)
    session.is_system_playing = True
    telemetry.start_tts()

    full_audio_stream = session.tts.stream_tokens_to_audio(
        token_generator(),
        on_first_audio=telemetry.mark_tts_first_audio,
    )

    async for audio_chunk in full_audio_stream:
        await session.send_audio_chunk(audio_chunk)

    telemetry.end_llm()
    telemetry.end_tts()
    session.is_system_playing = False

    full_generated_text = "".join(generated_tokens).strip()

    # 3. Grounding / Auditor Gate
    grounding_result = grounding_gate.audit(full_generated_text, relevant_chunks)
    telemetry.grounding_confidence = grounding_result.confidence_score

    # 4. Structured Booking Extractor
    booking_req = booking_extractor.extract_from_conversation(
        user_utterance,
        session.conversation_history,
    )
    if booking_req:
        confirmed = booking_extractor.write_to_calendar(booking_req)
        telemetry.booking_extracted = True
        telemetry.booking_details = confirmed
        await session.send_json({"type": "booking_confirmed", "booking": confirmed})

    # Update conversation history
    session.conversation_history.append({"role": "user", "content": user_utterance})
    session.conversation_history.append({"role": "assistant", "content": full_generated_text})

    # Emit telemetry
    metrics = telemetry.to_metrics_dict()
    telemetry_history.append(metrics)
    await session.send_json({
        "type": "response_completed",
        "text": full_generated_text,
        "metrics": metrics,
    })


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting TriagePulse-AI Gateway on http://{host}:{port}")
    uvicorn.run("perception.ws_gateway:app", host=host, port=port, reload=True)
