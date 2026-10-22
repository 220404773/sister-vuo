from __future__ import annotations

import json

from fastapi.testclient import TestClient

from app.config import Settings
from tests.conftest import FakeLLM


def start(
    client: TestClient,
    scenario_id: str = "abdominal_pain",
    difficulty: str | None = None,
) -> dict:
    payload: dict[str, str] = {"scenario_id": scenario_id}
    if difficulty is not None:
        payload["difficulty"] = difficulty
    response = client.post("/session/start", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def session_headers(started: dict) -> dict[str, str]:
    return {"X-Session-Token": started["session_token"]}


def test_ui_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Sister Vuo" in response.text
    assert "Start session" in response.text
    assert "Patient monitor" in response.text
    assert "Past sessions" in response.text
    assert "supervisor hint" in response.text


def test_health_and_scenarios(client: TestClient, fake_llm: FakeLLM) -> None:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["scenarios"] == 3
    assert health.json()["llm_configured"] is True
    assert health.json()["whatsapp_configured"] is True
    assert fake_llm.probe_calls == 0

    probed = client.get("/health", params={"probe": "true"})
    assert probed.status_code == 200
    assert probed.json()["status"] == "ok"
    assert probed.json()["llm"]["ok"] is True
    assert fake_llm.probe_calls == 1

    fake_llm.probe_result = {"ok": False, "error": "HTTP 401", "error_type": "LLMAuthenticationError"}
    degraded = client.get("/health", params={"probe": "true"})
    assert degraded.status_code == 200
    assert degraded.json()["status"] == "degraded"
    assert degraded.json()["llm"]["ok"] is False

    deep = client.get("/health/llm")
    assert deep.status_code == 200
    assert deep.json()["status"] == "degraded"

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
    headers = session_headers(started)
    assert started["session_token"]
    assert started["scenario"]["id"] == "abdominal_pain"
    assert started["student_turns"] == 0
    assert started["turns_remaining"] == 12
    assert "Sister" in started["opening_message"]
    assert started["difficulty"] == "Intermediate"
    assert started["hints_enabled"] is True
    assert started["monitor"]["status"] in {"stable", "deteriorating"}
    assert {item["id"] for item in started["actions"]} >= {"vitals", "call_ambulance"}

    message = client.post(
        f"/session/{session_id}/message",
        json={"message": "When did the pain start?"},
        headers=headers,
    )
    assert message.status_code == 200, message.text
    payload = message.json()
    assert payload["reply"] == fake_llm.default_reply
    assert payload["student_turns"] == 1
    assert payload["turns_remaining"] == 11
    assert payload["status"] == "active"
    assert payload["evaluation"] is None
    assert payload["monitor"]["clinical_minutes"] > 0
    assert "onset" in fake_llm.revealed_calls[-1]

    ended = client.post(f"/session/{session_id}/end", headers=headers)
    assert ended.status_code == 200, ended.text
    evaluation = ended.json()["evaluation"]
    assert 0 <= evaluation["overall_score"] <= 5
    assert set(evaluation["domain_scores"]) == {
        "red_flag_recognition",
        "history_completeness",
        "prioritisation",
        "safety",
        "communication_empathy",
        "time_to_escalation",
    }
    assert evaluation["debrief"]["timeline"]
    assert evaluation["critical_errors"]
    assert fake_llm.evaluation_calls == 1

    repeated = client.post(f"/session/{session_id}/end", headers=headers)
    assert repeated.status_code == 200
    assert repeated.json()["evaluation"] == evaluation
    assert fake_llm.evaluation_calls == 1

    snapshot = client.get(f"/session/{session_id}", headers=headers)
    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "completed"
    assert "diagnosis" not in snapshot.text

    log_path = settings.log_dir / f"{session_id}.jsonl"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")
    assert "session_started" in log_text
    assert "student_message" in log_text
    assert "evaluation" in log_text
    assert started["session_token"] not in log_text


def test_auto_evaluation_after_twelve_turns(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    started = start(client)
    session_id = started["session_id"]
    headers = session_headers(started)
    last = None
    for turn in range(1, 13):
        response = client.post(
            f"/session/{session_id}/message",
            json={"message": f"Question number {turn}"},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        last = response.json()

    assert last is not None
    assert last["student_turns"] == 12
    assert last["turns_remaining"] == 0
    assert last["status"] == "completed"
    assert last["auto_evaluated"] is True
    assert 0 <= last["evaluation"]["overall_score"] <= 5
    assert last["evaluation"]["debrief"]["timeline"]
    assert fake_llm.evaluation_calls == 1

    overflow = client.post(
        f"/session/{session_id}/message",
        json={"message": "One more question"},
        headers=headers,
    )
    assert overflow.status_code == 409


def test_evaluation_failure_at_turn_limit_can_be_retried(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    fake_llm.evaluation_failures = 1
    started = start(client)
    session_id = started["session_id"]
    headers = session_headers(started)
    last = None
    for turn in range(1, 13):
        response = client.post(
            f"/session/{session_id}/message",
            json={"message": f"Question number {turn}"},
            headers=headers,
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
        headers=headers,
    )
    assert overflow.status_code == 409

    retried = client.post(f"/session/{session_id}/end", headers=headers)
    assert retried.status_code == 200, retried.text
    assert 0 <= retried.json()["evaluation"]["overall_score"] <= 5
    assert fake_llm.evaluation_calls == 2


def test_session_can_be_ended_without_messages(
    client: TestClient,
    fake_llm: FakeLLM,
) -> None:
    started = start(client)
    session_id = started["session_id"]
    response = client.post(f"/session/{session_id}/end", headers=session_headers(started))
    assert response.status_code == 200
    evaluation = response.json()["evaluation"]
    assert 0 <= evaluation["overall_score"] <= 5
    assert evaluation["debrief"]["model_answer"]
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
    started = start(client)
    session_id = started["session_id"]
    response = client.post(
        f"/session/{session_id}/message",
        json={"message": "   "},
        headers=session_headers(started),
    )
    assert response.status_code == 422


def test_clinical_actions_update_the_monitor(client: TestClient) -> None:
    started = start(client, "chest_pain")
    session_id = started["session_id"]
    headers = session_headers(started)

    vitals = client.post(
        f"/session/{session_id}/action",
        json={"action_id": "vitals"},
        headers=headers,
    )
    assert vitals.status_code == 200, vitals.text
    result = vitals.json()
    assert "HR" in result["result"]
    assert result["monitor"]["clinical_minutes"] >= 3
    assert "vitals" in result["actions_taken"]

    ecg = client.post(
        f"/session/{session_id}/action", json={"action_id": "ecg"}, headers=headers
    )
    assert ecg.status_code == 200, ecg.text
    assert "ST-segment" in ecg.json()["result"]

    repeated = client.post(
        f"/session/{session_id}/action", json={"action_id": "ecg"}, headers=headers
    )
    assert repeated.status_code == 409

    unknown = client.post(
        f"/session/{session_id}/action", json={"action_id": "xray"}, headers=headers
    )
    assert unknown.status_code == 404

    ambulance = client.post(
        f"/session/{session_id}/action",
        json={"action_id": "call_ambulance"},
        headers=headers,
    )
    assert ambulance.status_code == 200, ambulance.text
    assert ambulance.json()["monitor"]["escalated"] is True
    assert ambulance.json()["monitor"]["escalation_minutes"] is not None

    monitor = client.get(f"/session/{session_id}/monitor", headers=headers)
    assert monitor.status_code == 200
    assert monitor.json()["escalated"] is True

    ended = client.post(f"/session/{session_id}/end", headers=headers)
    assert ended.status_code == 200, ended.text
    evaluation = ended.json()["evaluation"]
    assert evaluation["debrief"]["actions_taken"] == ["vitals", "ecg", "call_ambulance"]
    assert evaluation["debrief"]["time_to_escalation_minutes"] is not None
    assert evaluation["debrief"]["escalation_target_minutes"] > 0


def test_beginner_receives_automatic_hint_at_turn_three(client: TestClient) -> None:
    started = start(client, "abdominal_pain", "Beginner")
    session_id = started["session_id"]
    headers = session_headers(started)
    hints = []
    for turn in range(1, 4):
        response = client.post(
            f"/session/{session_id}/message",
            json={"message": f"History question {turn}"},
            headers=headers,
        )
        assert response.status_code == 200, response.text
        hints.append(response.json()["hint"])
    assert hints[0] is None and hints[1] is None
    assert hints[2]
    snapshot = client.get(f"/session/{session_id}", headers=headers)
    assert snapshot.json()["hints_used"] == 0


def test_difficulty_controls_hints(client: TestClient) -> None:
    beginner = start(client, "abdominal_pain", "Beginner")
    assert beginner["hints_enabled"] is True
    hint = client.post(
        f"/session/{beginner['session_id']}/hint",
        headers=session_headers(beginner),
    )
    assert hint.status_code == 200, hint.text
    assert hint.json()["hints_used"] == 1
    assert hint.json()["hint_penalty"] > 0

    advanced = start(client, "abdominal_pain", "Advanced")
    assert advanced["hints_enabled"] is False
    blocked = client.post(
        f"/session/{advanced['session_id']}/hint",
        headers=session_headers(advanced),
    )
    assert blocked.status_code == 409

    snapshot = client.get(
        f"/session/{advanced['session_id']}",
        headers=session_headers(advanced),
    )
    assert snapshot.status_code == 200
    assert snapshot.json()["difficulty"] == "Advanced"
    assert snapshot.json()["hints_used"] == 0
