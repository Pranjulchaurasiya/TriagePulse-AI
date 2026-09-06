"""Deepgram Nova-2 Streaming Speech-To-Text client.
Supports live bidirectional WebSocket streaming and high-fidelity local simulation.
"""

from __future__ import annotations
import asyncio
import json
import logging
import os
from typing import AsyncGenerator, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class DeepgramSTTStream:
    """Streaming STT interface wrapping Deepgram Nova-2 with graceful mock fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        sample_rate: int = 16000,
    ):
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")
        if mock_mode is None:
            mock_env = os.getenv("MOCK_PROVIDERS", "true").lower()
            self.mock_mode = (mock_env in ("true", "1", "yes")) or not bool(self.api_key)
        else:
            self.mock_mode = mock_mode

        self.sample_rate = sample_rate
        self.is_connected = False
        self._ws = None

    async def connect(self) -> None:
        """Establish connection to Deepgram Nova-2 or initialize simulator."""
        if self.mock_mode:
            self.is_connected = True
            logger.info("DeepgramSTT initialized in simulated/mock mode.")
            return

        try:
            import websockets
            url = (
                f"wss://api.deepgram.com/v1/listen"
                f"?model=nova-2&encoding=linear16&sample_rate={self.sample_rate}"
                f"&channels=1&interim_results=true&punctuate=true&smart_format=true"
            )
            headers = {"Authorization": f"Token {self.api_key}"}
            self._ws = await websockets.connect(url, additional_headers=headers)
            self.is_connected = True
            logger.info("Connected to Deepgram Nova-2 streaming WebSocket.")
        except Exception as e:
            logger.warning(f"Failed to connect to Deepgram live API: {e}. Falling back to simulation.")
            self.mock_mode = True
            self.is_connected = True

    async def send_audio_chunk(self, chunk: bytes) -> None:
        """Send raw PCM audio chunk to Deepgram."""
        if self.mock_mode or not self._ws:
            return
        try:
            await self._ws.send(chunk)
        except Exception as e:
            logger.error(f"Error sending audio chunk to Deepgram: {e}")

    async def close(self) -> None:
        """Close connection."""
        self.is_connected = False
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

    async def simulate_stream(self, text: str) -> AsyncGenerator[Dict, None]:
        """Simulate real-time streaming STT emitting incremental partials and a final transcript."""
        words = text.split()
        partial_acc = []
        for idx, word in enumerate(words):
            partial_acc.append(word)
            is_final = (idx == len(words) - 1)
            # Simulated speech latency per word (30ms)
            await asyncio.sleep(0.03)
            yield {
                "transcript": " ".join(partial_acc),
                "is_final": is_final,
                "confidence": 0.98 if is_final else 0.85,
                "speech_final": is_final,
            }
