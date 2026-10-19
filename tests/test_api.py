from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from tests.conftest import FakeLLM


def start(client: TestClient, scenario_id: str = "abdominal_pain") -> dict:
    response = client.post("/session/start", json={"scenario_id": scenario_id})
    assert response.status_code == 200, response.text
    return response.json()


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Sister Vuo" in response.text
    assert "Start session" in response.text


def test_health_and_scenarios(client: TestClient) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["scenarios"] == 3
    assert health.json()["llm_configured"] is True
    assert health.json()["whatsapp_configured"] is True

    scenarios = client.get("/scenarios")
    assert scenarios.status_code == 200
    payload = scenarios.json()
    assert [item["id"] for item in payload] == [
        "abdominal_pain",
        "paediatric_dehydration",
        "chest_pain",
    ]
    body = json.dumps(payload)
    assert "ectopic" not in body.lower()
    assert "myocardial infarction" not in body.lower()


def test_start_message_end_flow(
    client: TestClient,
    fake_llm: FakeLLM,
    settings: Settings,
) -> None:
    started = start(client)
    session_id = started["session_id"]
    assert started["scenario"]["id"] == "abdominal_pain"
    assert started["student_turns"] == 0
    assert started["turns_remaining"] == 12
    assert "Sister" in started["opening_message"]

    message = client.post(
        f"/session/{session_id}/message",
        json={"message": "When did the pain start?"},
    )
    assert message.status_code == 200, message.text
    payload = message.json()
    assert payload["reply"] == fake_llm.default_reply
    assert payload["student_turns"] == 1
    assert payload["turns_remaining"] == 11
    assert payload["status"] == "active"
    assert payload["evaluation"] is None

    ended = client.post(f"/session/{session_id}/end")
    assert ended.status_code == 200, ended.text
    evaluation = ended.json()["evaluation"]
    assert evaluation["overall_score"] == 4.0
    assert set(evaluation["domain_scores"]) == {
        "history_taking",
        "clinical_reasoning",
        "communication_empathy",
        "language_accessibility",
        "safety_referral",
    }
    assert fake_llm.evaluation_calls == 1

    repeated = client.post(f"/session/{session_id}/end")
    assert repeated.status_code == 200
    assert repeated.json()["evaluation"] == evaluation
    assert fake_llm.evaluation_calls == 1

    snapshot = client.get(f"/session/{session_id}")
    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "completed"
    assert "diagnosis" not in snapshot.text

    log_path = settings.log_dir / f"{session_id}.jsonl"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "session_started" in log_text
    assert "student_message" in log_text
    assert "evaluation" in log_text


def test_auto_evaluation_after_twelve_turns(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    started = start(client)
    session_id = started["session_id"]
    last = None
    for turn in range(1, 13):
        response = client.post(
            f"/session/{session_id}/message",
            json={"message": f"Question number {turn}"},
        )
        assert response.status_code == 200, response.text
        last = response.json()

    assert last is not None
    assert last["student_turns"] == 12
    assert last["turns_remaining"] == 0
    assert last["status"] == "completed"
    assert last["auto_evaluated"] is True
    assert last["evaluation"]["overall_score"] == 4.0
    assert fake_llm.evaluation_calls == 1

    overflow = client.post(
        f"/session/{session_id}/message",
        json={"message": "One more question"},
    )
    assert overflow.status_code == 409


def test_evaluation_failure_at_turn_limit_can_be_retried(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.evaluation_failures = 1
    started = start(client)
    session_id = started["session_id"]
    last = None
    for turn in range(1, 13):
        response = client.post(
            f"/session/{session_id}/message",
            json={"message": f"Question number {turn}"},
        )
        assert response.status_code == 200, response.text
        last = response.json()

    assert last is not None
    assert last["status"] == "active"
    assert last["turns_remaining"] == 0
    assert last["evaluation"] is None
    assert last["evaluation_error"]
    assert last["auto_evaluated"] is False

    overflow = client.post(
        f"/session/{session_id}/message",
        json={"message": "One more question"},
    )
    assert overflow.status_code == 409

    retried = client.post(f"/session/{session_id}/end")
    assert retried.status_code == 200, retried.text
    assert retried.json()["evaluation"]["overall_score"] == 4.0
    assert fake_llm.evaluation_calls == 2


def test_session_can_be_ended_without_messages(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    session_id = start(client)["session_id"]
    response = client.post(f"/session/{session_id}/end")
    assert response.status_code == 200
    assert response.json()["evaluation"]["overall_score"] == 4.0
    assert fake_llm.evaluation_calls == 1


def test_unknown_scenario(client: TestClient) -> None:
    response = client.post("/session/start", json={"scenario_id": "does-not-exist"})
    assert response.status_code == 404


def test_unknown_session(client: TestClient) -> None:
    response = client.post("/session/missing/message", json={"message": "Hello"})
    assert response.status_code == 404
    response = client.post("/session/missing/end")
    assert response.status_code == 404


def test_requires_non_empty_message(client: TestClient) -> None:
    session_id = start(client)["session_id"]
    response = client.post(f"/session/{session_id}/message", json={"message": "   "})
    assert response.status_code == 422
