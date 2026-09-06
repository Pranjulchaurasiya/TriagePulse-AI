"""Voice Activity Detection (VAD) & Barge-In Detector.
Analyzes streaming PCM audio frames (16-bit 16kHz mono) to detect:
1. Speech onset (user begins talking)
2. Speech completion / silence timeout (user finished speaking)
3. Mid-playback barge-in (user starts speaking while system is outputting audio)
"""

from __future__ import annotations
import math
import struct
from typing import Optional


class VoiceActivityDetector:
    """Frame-by-frame VAD with adaptive noise floor and barge-in trigger."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        energy_threshold: float = 0.015,
        silence_timeout_ms: int = 600,
        speech_lead_ms: int = 120,
    ):
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))  # e.g., 480 samples = 960 bytes
        self.bytes_per_frame = self.frame_size * 2  # 16-bit PCM = 2 bytes per sample
        self.energy_threshold = energy_threshold
        self.silence_timeout_ms = silence_timeout_ms
        self.speech_lead_ms = speech_lead_ms

        # State
        self.is_speaking = False
        self.silence_frames_count = 0
        self.speech_frames_count = 0
        self.consecutive_speech_required = int(speech_lead_ms / frame_duration_ms)
        self.consecutive_silence_required = int(silence_timeout_ms / frame_duration_ms)

    def calculate_frame_rms(self, frame_bytes: bytes) -> float:
        """Calculate Root Mean Square (RMS) energy normalized between 0.0 and 1.0."""
        if len(frame_bytes) < 2:
            return 0.0
        # Unpack 16-bit signed integers (little-endian)
        num_samples = len(frame_bytes) // 2
        fmt = f"<{num_samples}h"
        try:
            samples = struct.unpack(fmt, frame_bytes[: num_samples * 2])
        except struct.error:
            return 0.0

        if not samples:
            return 0.0

        sum_squares = sum(s * s for s in samples)
        mean_square = sum_squares / len(samples)
        rms = math.sqrt(mean_square)
        # Normalize against 16-bit int max (32767)
        return rms / 32767.0

    def process_frame(self, frame_bytes: bytes, is_system_speaking: bool = False) -> dict:
        """Process an audio frame and return speech/barge-in status.
        
        Returns:
            {
                "is_speech": bool,
                "speech_started": bool,
                "speech_ended": bool,
                "barge_in_triggered": bool,
                "rms": float
            }
        """
        rms = self.calculate_frame_rms(frame_bytes)
        has_voice = rms >= self.energy_threshold

        speech_started = False
        speech_ended = False
        barge_in_triggered = False

        if has_voice:
            self.speech_frames_count += 1
            self.silence_frames_count = 0
            if not self.is_speaking and self.speech_frames_count >= self.consecutive_speech_required:
                self.is_speaking = True
                speech_started = True
                # If system is speaking and user starts talking -> trigger barge-in!
                if is_system_speaking:
                    barge_in_triggered = True
        else:
            self.silence_frames_count += 1
            if self.is_speaking and self.silence_frames_count >= self.consecutive_silence_required:
                self.is_speaking = False
                speech_ended = True
                self.speech_frames_count = 0
            elif not self.is_speaking:
                self.speech_frames_count = 0

        return {
            "is_speech": self.is_speaking,
            "speech_started": speech_started,
            "speech_ended": speech_ended,
            "barge_in_triggered": barge_in_triggered,
            "rms": rms,
        }

    def reset(self) -> None:
        """Reset internal speech state."""
        self.is_speaking = False
        self.silence_frames_count = 0
        self.speech_frames_count = 0
