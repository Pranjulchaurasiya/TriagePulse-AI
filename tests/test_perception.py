import asyncio
import struct
import pytest
from perception.vad import VoiceActivityDetector
from perception.stt_deepgram import DeepgramSTTStream
from perception.tts_stream import StreamingTTS, generate_synthesized_audio_chunk


def generate_pcm_frame(is_loud: bool = False, num_samples: int = 480) -> bytes:
    """Generate dummy 16-bit PCM frame (30ms at 16kHz)."""
    amplitude = 15000 if is_loud else 50
    data = [amplitude if i % 2 == 0 else -amplitude for i in range(num_samples)]
    return struct.pack(f"<{num_samples}h", *data)


def test_vad_silence_vs_speech():
    vad = VoiceActivityDetector(frame_duration_ms=30, speech_lead_ms=60, silence_timeout_ms=120)
    silent_frame = generate_pcm_frame(is_loud=False)
    res_silent = vad.process_frame(silent_frame)
    assert not res_silent["is_speech"]
    assert res_silent["rms"] < 0.01

    loud_frame = generate_pcm_frame(is_loud=True)
    # Feed enough frames to pass speech lead threshold
    vad.process_frame(loud_frame)
    res_loud = vad.process_frame(loud_frame)
    assert res_loud["is_speech"]
    assert res_loud["rms"] > 0.1


def test_vad_barge_in_trigger():
    vad = VoiceActivityDetector(frame_duration_ms=30, speech_lead_ms=60)
    loud_frame = generate_pcm_frame(is_loud=True)

    # Frame 1: system is speaking
    res1 = vad.process_frame(loud_frame, is_system_speaking=True)
    # Frame 2: passes speech lead threshold, triggers barge-in!
    res2 = vad.process_frame(loud_frame, is_system_speaking=True)
    assert res2["barge_in_triggered"]


@pytest.mark.asyncio
async def test_deepgram_stt_stream_simulation():
    stt = DeepgramSTTStream(mock_mode=True)
    await stt.connect()
    assert stt.is_connected

    partials = []
    finals = []
    async for packet in stt.simulate_stream("Hello I would like to see Dr Smith"):
        if packet["is_final"]:
            finals.append(packet)
        else:
            partials.append(packet)

    assert len(partials) > 0
    assert len(finals) == 1
    assert "Dr Smith" in finals[0]["transcript"]
    await stt.close()


@pytest.mark.asyncio
async def test_streaming_tts_cancellation_barge_in():
    tts = StreamingTTS(mock_mode=True)

    async def token_gen():
        for word in ["Good", "morning", "how", "can", "I", "help", "you", "today", "at", "the", "surgery"]:
            yield word + " "
            await asyncio.sleep(0.01)

    chunks_received = 0
    async for chunk in tts.stream_tokens_to_audio(token_gen()):
        chunks_received += 1
        if chunks_received >= 2:
            tts.cancel()  # Barge-in mid stream!

    # Verify stream was cut short
    assert tts.is_cancelled
    assert chunks_received < 10


def test_sip_telephony_transcoder():
    """Verify G.711 A-law 8kHz <-> 16kHz PCM transcoding roundtrip."""
    from perception.sip_bridge import TelephonyAudioTranscoder
    
    # 160 bytes of G.711 A-law = 20ms of 8kHz telephony audio
    raw_rtp_alaw = bytes([0xD5] * 160)
    pcm16 = TelephonyAudioTranscoder.g711a_to_pcm16(raw_rtp_alaw)
    
    # 160 samples @ 8kHz -> 320 samples @ 16kHz * 2 bytes/sample = 640 bytes
    assert len(pcm16) == 640
    
    # Roundtrip back to G.711 A-law
    out_alaw = TelephonyAudioTranscoder.pcm16_to_g711a(pcm16)
    assert len(out_alaw) == 160


@pytest.mark.asyncio
async def test_sip_session_lifecycle():
    """Verify SIP call session state transitions."""
    from perception.sip_bridge import SIPCallMetadata, SIPTelephonySession, CallState
    
    metadata = SIPCallMetadata(
        call_id="SIP-TEST-9988",
        caller_cli="07891234567",
        called_did="02079460123",
    )
    session = SIPTelephonySession(metadata)
    assert session.state == CallState.IDLE
    
    session.answer_call()
    assert session.state == CallState.CONNECTED
    assert session.is_active
    
    # Test processing a 20ms inbound RTP packet
    rtp_frame = bytes([0x55] * 160)
    ai_frame = await session.process_inbound_rtp_frame(rtp_frame)
    assert len(ai_frame) == 640
    
    session.terminate_call(reason="caller_hung_up")
    assert session.state == CallState.TERMINATED
    assert not session.is_active
