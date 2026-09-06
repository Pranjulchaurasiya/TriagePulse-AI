"""Audio Sample Generator for Automated Perception and Latency Testing.
Generates realistic 16-bit 16kHz mono PCM WAV test files.
"""

from __future__ import annotations
import math
import os
import struct
import wave
from pathlib import Path


def generate_sample_wav(
    file_path: Path,
    duration_s: float = 2.0,
    sample_rate: int = 16000,
    has_speech: bool = True,
) -> None:
    """Generate sample WAV audio file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(duration_s * sample_rate)

    with wave.open(str(file_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        buffer = bytearray()
        for i in range(num_samples):
            if has_speech:
                # Modulated harmonic signals simulating voice frequencies
                f1 = 220.0
                f2 = 440.0
                envelope = 0.5 * (1.0 + math.sin(2 * math.pi * 1.5 * i / sample_rate))
                val = int(
                    16000.0
                    * envelope
                    * (0.6 * math.sin(2 * math.pi * f1 * i / sample_rate) + 0.4 * math.sin(2 * math.pi * f2 * i / sample_rate))
                )
            else:
                val = 0  # Silence
            buffer.extend(struct.pack("<h", max(-32767, min(32767, val))))

        wav_file.writeframes(buffer)


if __name__ == "__main__":
    output_dir = Path(__file__).parent
    generate_sample_wav(output_dir / "sample_speech.wav", duration_s=1.5, has_speech=True)
    generate_sample_wav(output_dir / "sample_silence.wav", duration_s=1.0, has_speech=False)
    print(f"Generated test audio samples in {output_dir}")
