"""Unit and integration tests for Reasoning Layer (Tier 2).
Tests RAG Policy Retrieval, Grounding/Auditor Gate hallucination rejection, and Structured Booking Extraction.
"""

import pytest
from reasoning.rag_retriever import PolicyRetriever, PolicyChunk
from reasoning.llm_groq import GroqLLMClient
from reasoning.grounding_gate import GroundingGate
from reasoning.booking_extractor import BookingExtractor, BookingRequest


@pytest.fixture
def retriever():
    return PolicyRetriever()


@pytest.fixture
def grounding_gate():
    return GroundingGate()


@pytest.fixture
def booking_extractor():
    return BookingExtractor()


def test_rag_retrieval_opening_hours(retriever):
    chunks = retriever.retrieve("What time does the surgery open on Monday?", top_k=2)
    assert len(chunks) > 0
    assert any("surgery_hours_and_access" in c.source_file for c in chunks)
    # Check that opening time (08:00) is in content
    assert any("08:00" in c.content for c in chunks)


def test_rag_retrieval_repeat_prescriptions(retriever):
    chunks = retriever.retrieve("How long does it take for a repeat prescription?", top_k=2)
    assert len(chunks) > 0
    assert any("repeat_prescriptions" in c.source_file for c in chunks)
    assert any("48 working hours" in c.content or "2 full working days" in c.content for c in chunks)


def test_rag_retrieval_urgent_triage(retriever):
    chunks = retriever.retrieve("Can I get an urgent same-day appointment for an earache?", top_k=2)
    assert len(chunks) > 0
    assert any("appointment_booking_rules" in c.source_file for c in chunks)
    assert any("Urgent Same-Day" in c.content for c in chunks)


@pytest.mark.asyncio
async def test_groq_llm_streaming(retriever):
    client = GroqLLMClient(mock_mode=True)
    chunks = retriever.retrieve("What are your opening hours?", top_k=2)

    tokens = []
    async for token in client.stream_response("What are your opening hours?", chunks):
        tokens.append(token)

    full_text = "".join(tokens)
    assert len(tokens) > 5
    assert "St. Jude Medical Centre" in full_text
    assert "8:00 AM" in full_text


def test_grounding_gate_accepts_valid_policy_statement(grounding_gate, retriever):
    chunks = retriever.retrieve("What are the surgery opening hours?", top_k=3)
    valid_response = (
        "St. Jude Medical Centre is open Monday to Friday from 08:00 to 18:30. "
        "We also offer evening clinics on Tuesdays and Thursdays."
    )
    result = grounding_gate.audit(valid_response, chunks)
    assert result.is_grounded
    assert not result.fallback_applied
    assert len(result.violations) == 0


def test_grounding_gate_rejects_hallucinated_medical_prescriptions(grounding_gate, retriever):
    chunks = retriever.retrieve("I have a headache", top_k=2)
    hallucinated_response = "You should immediately take 500 mg of amoxicillin tablets twice a day for your pain."

    result = grounding_gate.audit(hallucinated_response, chunks)
    assert not result.is_grounded
    assert result.fallback_applied
    assert any("clinical/medical assertion" in v for v in result.violations)
    assert "transfer you directly to our reception team" in result.sanitized_response


def test_grounding_gate_rejects_contradictory_operating_hours(grounding_gate, retriever):
    chunks = retriever.retrieve("Are you open on the weekend?", top_k=2)
    hallucinated_response = "Yes, our GP surgery is open on Sunday from 9am to 5pm."

    result = grounding_gate.audit(hallucinated_response, chunks)
    assert not result.is_grounded
    assert result.fallback_applied
    assert any("open on" in v.lower() for v in result.violations)


def test_booking_extractor_routine_appointment(booking_extractor):
    conversation = [
        {"role": "user", "content": "I would like to book an appointment with Dr. Davies for next Tuesday morning."},
        {"role": "assistant", "content": "Certainly, may I have your name and the reason for your visit?"},
    ]
    query = "My name is John Smith and I need an appointment for a routine medication review."

    req = booking_extractor.extract_from_conversation(query, conversation)
    assert req is not None
    assert req.patient_name == "John Smith"
    assert req.urgency == "routine"
    assert req.confidence >= 0.85

    confirmed = booking_extractor.write_to_calendar(req)
    assert confirmed["booking_id"].startswith("STJ-")
    assert confirmed["status"] == "CONFIRMED"
    assert len(booking_extractor.confirmed_bookings) == 1


def test_booking_extractor_urgent_same_day(booking_extractor):
    query = "I need an urgent same day appointment today please, my name is Sarah Connor and I have sudden severe earache."
    req = booking_extractor.extract_from_conversation(query)
    assert req is not None
    assert req.patient_name == "Sarah Connor"
    assert req.urgency == "urgent_same_day"


def test_booking_extractor_nurse_clinic(booking_extractor):
    query = "Hi, my name is David Miller and I need to book a blood test with the nurse tomorrow morning."
    req = booking_extractor.extract_from_conversation(query)
    assert req is not None
    assert req.patient_name == "David Miller"
    assert req.urgency == "nurse_clinic"
