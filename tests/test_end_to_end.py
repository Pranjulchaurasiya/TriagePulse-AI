"""End-to-End Integration Tests for TriagePulse-AI.
Validates WebSocket gateway lifecycle, routine triage booking flow, emergency red-flag bypass, and barge-in.
"""

import json
import pytest
from starlette.testclient import TestClient
from perception.ws_gateway import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["policies_indexed"] > 0
    assert data["safety_rules_count"] > 0


def test_telemetry_endpoint(client):
    response = client.get("/api/telemetry")
    assert response.status_code == 200
    data = response.json()
    assert "history" in data


def test_websocket_routine_booking_flow(client):
    with client.websocket_connect("/ws/audio") as ws:
        # Initial greeting packet
        ready_packet = ws.receive_json()
        assert ready_packet["type"] == "connection_ready"
        assert "session_id" in ready_packet

        # Send text query simulating user speech
        ws.send_json({
            "type": "text_query",
            "text": "Hi, my name is John Smith and I would like to book an appointment for next Tuesday morning.",
        })

        events = []
        while True:
            msg = ws.receive()
            if "text" in msg:
                event = json.loads(msg["text"])
                events.append(event)
                if event.get("type") == "response_completed":
                    break

        event_types = [e["type"] for e in events]
        assert "transcript" in event_types
        assert "retrieval" in event_types
        assert "booking_confirmed" in event_types
        assert "response_completed" in event_types

        # Check booking details
        booking_event = next(e for e in events if e["type"] == "booking_confirmed")
        assert booking_event["booking"]["patient_name"] == "John Smith"
        assert booking_event["booking"]["status"] == "CONFIRMED"

        # Check telemetry
        completed_event = next(e for e in events if e["type"] == "response_completed")
        assert "metrics" in completed_event
        metrics = completed_event["metrics"]
        assert metrics["safety_triggered"] is False
        assert metrics["safety_latency_ms"] < 5.0
        assert metrics["meets_target_sub_800ms"] is True


def test_websocket_emergency_red_flag_bypass(client):
    with client.websocket_connect("/ws/audio") as ws:
        ready_packet = ws.receive_json()
        assert ready_packet["type"] == "connection_ready"

        # Send emergency utterance
        ws.send_json({
            "type": "text_query",
            "text": "I have severe crushing chest pain and feel dizzy",
        })

        events = []
        # Receive until safety telemetry is delivered
        while True:
            msg = ws.receive()
            if "text" in msg:
                event = json.loads(msg["text"])
                events.append(event)
                if event.get("type") == "telemetry":
                    break

        event_types = [e["type"] for e in events]
        assert "safety_alert" in event_types
        assert "telemetry" in event_types
        # Crucial: verify LLM and booking are bypassed completely!
        assert "retrieval" not in event_types
        assert "booking_confirmed" not in event_types

        alert_event = next(e for e in events if e["type"] == "safety_alert")
        assert alert_event["category"] == "cardiac"
        assert "999" in alert_event["instruction"]
        assert alert_event["latency_ms"] < 3.0


def test_websocket_barge_in_control(client):
    with client.websocket_connect("/ws/audio") as ws:
        ready_packet = ws.receive_json()
        assert ready_packet["type"] == "connection_ready"

        # Trigger barge-in interrupt signal
        ws.send_json({"type": "interrupt"})
        reply = ws.receive_json()
        assert reply["type"] == "barge_in_confirmed"
