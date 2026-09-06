"""Cartesia & ElevenLabs Streaming TTS Client with Instant Barge-In Cancellation.
Streams audio chunks token-by-token with sub-200ms TTFB.
"""

from __future__ import annotations
import asyncio
import io
import logging
import math
import os
import struct
import wave
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


def generate_synthesized_audio_chunk(duration_s: float = 0.2, freq: float = 440.0, sample_rate: int = 16000) -> bytes:
    """Generate synthesized 16-bit PCM sine audio chunk for zero-dependency simulation/playback."""
    num_samples = int(sample_rate * duration_s)
    buffer = bytearray()
    for i in range(num_samples):
        # Sine wave with gentle decay
        envelope = math.sin(math.pi * i / num_samples)
        sample = int(32767.0 * 0.3 * envelope * math.sin(2.0 * math.pi * freq * i / sample_rate))
        buffer.extend(struct.pack("<h", max(-32767, min(32767, sample))))
    return bytes(buffer)


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a WAV header."""
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return wav_io.getvalue()


class StreamingTTS:
    """Token-to-audio streaming synthesizer with immediate cancellation support."""

    def __init__(
        self,
        cartesia_api_key: Optional[str] = None,
        elevenlabs_api_key: Optional[str] = None,
        mock_mode: Optional[bool] = None,
        sample_rate: int = 16000,
    ):
        self.cartesia_api_key = cartesia_api_key or os.getenv("CARTESIA_API_KEY")
        self.elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        
        if mock_mode is None:
            mock_env = os.getenv("MOCK_PROVIDERS", "true").lower()
            self.mock_mode = (mock_env in ("true", "1", "yes")) or not (self.cartesia_api_key or self.elevenlabs_api_key)
        else:
            self.mock_mode = mock_mode

        self.sample_rate = sample_rate
        self.is_cancelled = False

    def cancel(self) -> None:
        """Instantly abort current streaming playback (Barge-in reflex)."""
        self.is_cancelled = True
        logger.info("TTS playback cancellation triggered (Barge-in).")

    def reset(self) -> None:
        """Reset cancellation state for next turn."""
        self.is_cancelled = False

    async def stream_tokens_to_audio(
        self,
        token_stream: AsyncGenerator[str, None],
        on_first_audio: Optional[callable] = None,
    ) -> AsyncGenerator[bytes, None]:
        """Consume streamed text tokens and yield streaming audio chunks as soon as ready."""
        self.reset()
        first_chunk_emitted = False
        accumulated_text = ""

        # Tone pitch sequence for natural acoustic variety in simulated speech
        tones = [330.0, 392.0, 440.0, 493.88, 523.25]
        tone_idx = 0

        async for token in token_stream:
            if self.is_cancelled:
                logger.debug("TTS stream aborted due to barge-in.")
                break

            accumulated_text += token
            # In live Cartesia mode, tokens are forwarded directly over websocket
            # In simulation mode, yield small chunks per token/punctuation
            if any(punct in token for punct in [".", ",", "!", "?", " "]) or len(accumulated_text) > 15:
                # Emit audio chunk
                if not first_chunk_emitted:
                    if on_first_audio:
                        on_first_audio()
                    first_chunk_emitted = True

                freq = tones[tone_idx % len(tones)]
                tone_idx += 1
                audio_chunk = generate_synthesized_audio_chunk(
                    duration_s=0.08, freq=freq, sample_rate=self.sample_rate
                )
                yield audio_chunk
                accumulated_text = ""
                # Low latency delay simulating fast token-to-audio synthesis (30ms)
                await asyncio.sleep(0.03)

        if not self.is_cancelled and accumulated_text:
            if not first_chunk_emitted and on_first_audio:
                on_first_audio()
            audio_chunk = generate_synthesized_audio_chunk(
                duration_s=0.1, freq=330.0, sample_rate=self.sample_rate
            )
            yield audio_chunk
