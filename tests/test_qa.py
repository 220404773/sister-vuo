"""QA tests: one test (or more) per acceptance criterion, plus API contract checks.

Requirements pack: FR-01..FR-05, AC-01..AC-06. The pipeline runs this file with
the rest of the suite; a failing test here is a defect, not a flake.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import FakeLLM, FakeWhatsApp
from .test_whatsapp import make_payload, post_payload

VALID_URL = "https://knowledgehub.health.gov.za/elibrary/adult-primary-care-guide-apc-2026"


def make_app(settings: Settings, **changes):
    return create_app(
        settings_obj=replace(settings, **changes),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )


def make_client(settings: Settings, host: str = "testclient", **changes) -> TestClient:
    return TestClient(make_app(settings, **changes), client=(host, 50000))


def start(client: TestClient, scenario_id: str = "abdominal_pain") -> dict:
    response = client.post("/session/start", json={"scenario_id": scenario_id})
    assert response.status_code == 200, response.text
    return response.json()


def session_headers(started: dict) -> dict[str, str]:
    return {"X-Session-Token": started["session_token"]}


def write_guidelines(tmp_path: Path, steps: list[dict[str, str]]) -> Path:
    data = {
        "documents": {
            "apc": {
                "title": "Adult Primary Care Guide (APC)",
                "edition_year": "2026",
                "url": VALID_URL,
            }
        },
        "scenarios": {
            "abdominal_pain": {
                "primary_document": "apc",
                "section_text": "Abdominal pain",
                "steps": steps,
            }
        },
    }
    path = tmp_path / "guidelines.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def safe_steps() -> list[dict[str, str]]:
    return [
        {"id": "step-1", "text": "Ask about the pattern of the pain."},
        {"id": "step-2", "text": "Take vital signs and record them."},
        {"id": "step-3", "text": "Check for pregnancy if it is safe to do so."},
        {"id": "step-4", "text": "Explain the plan to the patient."},
        {"id": "step-5", "text": "Arrange urgent review."},
    ]


# --- FR-01 / AC-01: every session route needs the per-session token ----------


def test_ac01_all_session_routes_reject_missing_and_wrong_tokens(
    settings: Settings, tmp_path: Path
) -> None:
    with make_client(
        settings, guidelines_path=write_guidelines(tmp_path, safe_steps())
    ) as client:
        started = start(client)
        session_id = started["session_id"]
        headers = session_headers(started)

        calls = [
            ("POST", f"/session/{session_id}/message", {"message": "How long?"}),
            ("POST", f"/session/{session_id}/action", {"action_id": "vitals"}),
            ("POST", f"/session/{session_id}/hint", {"question": "pain"}),
            ("GET", f"/session/{session_id}/monitor", None),
            ("POST", f"/session/{session_id}/end", None),
            ("GET", f"/session/{session_id}", None),
        ]
        for method, path, body in calls:
            unknown = client.request(method, path.replace(session_id, "unknown-id"), json=body)
            missing = client.request(method, path, json=body)
            wrong = client.request(
                method, path, json=body, headers={"X-Session-Token": "wrong-token"}
            )
            assert unknown.status_code == 404, (method, path, unknown.text)
            assert missing.status_code == 404, (method, path, missing.text)
            assert wrong.status_code == 404, (method, path, wrong.text)
            assert missing.json() == unknown.json() == wrong.json()
            assert missing.json()["detail"] == "Session not found."

        # The right token still works on every route.
        assert client.get(f"/session/{session_id}/monitor", headers=headers).status_code == 200
        assert client.get(f"/session/{session_id}", headers=headers).status_code == 200
        message = client.post(
            f"/session/{session_id}/message",
            json={"message": "When did the pain start?"},
            headers=headers,
        )
        assert message.status_code == 200, message.text
        action = client.post(
            f"/session/{session_id}/action", json={"action_id": "vitals"}, headers=headers
        )
        assert action.status_code == 200, action.text
        hint = client.post(
            f"/session/{session_id}/hint", json={"question": "pain"}, headers=headers
        )
        assert hint.status_code == 200, hint.text
        ended = client.post(f"/session/{session_id}/end", headers=headers)
        assert ended.status_code == 200, ended.text


def test_ac01_a_token_only_opens_its_own_session(
    settings: Settings, tmp_path: Path
) -> None:
    with make_client(
        settings, guidelines_path=write_guidelines(tmp_path, safe_steps())
    ) as client:
        first = start(client)
        second = start(client)
        crossed = client.get(
            f"/session/{second['session_id']}", headers=session_headers(first)
        )
        assert crossed.status_code == 404
        assert crossed.json()["detail"] == "Session not found."
        assert (
            client.get(
                f"/session/{second['session_id']}", headers=session_headers(second)
            ).status_code
            == 200
        )


def test_ac01_whatsapp_sessions_do_not_require_a_token(
    client: TestClient, settings: Settings, fake_whatsapp: FakeWhatsApp
) -> None:
    phone = "27825555555"
    post_payload(client, settings, make_payload(phone, "START", "wamid.qa.start"))
    before = len(fake_whatsapp.sent)
    reply = post_payload(
        client, settings, make_payload(phone, "How long have you been unwell?", "wamid.qa.turn")
    )
    assert reply.status_code == 200
    # The follow-up was processed with only the phone binding, no session token.
    assert len(fake_whatsapp.sent) == before + 1
    assert "slow down" not in fake_whatsapp.sent[-1][1].lower()


# --- FR-01 / AC-02: the token is returned and never logged -------------------


def test_ac02_start_returns_a_random_token_that_is_not_logged(
    client: TestClient, settings: Settings
) -> None:
    first = start(client)
    second = start(client)

    token = first["session_token"]
    assert isinstance(token, str)
    assert len(token) >= 32
    assert token != second["session_token"]

    log_text = (settings.log_dir / f"{first['session_id']}.jsonl").read_text(encoding="utf-8")
    assert token not in log_text
    assert "session_token" not in log_text

    client.post(
        f"/session/{first['session_id']}/message",
        json={"message": "How long has it hurt?"},
        headers=session_headers(first),
    )
    after_turn = (settings.log_dir / f"{first['session_id']}.jsonl").read_text(encoding="utf-8")
    assert token not in after_turn


# --- FR-02 / AC-03: fail closed in production --------------------------------


def test_ac03_production_requires_an_api_key(settings: Settings) -> None:
    with pytest.raises(RuntimeError) as excinfo:
        make_app(settings, app_env="production", app_api_key="")
    assert "APP_API_KEY" in str(excinfo.value)

    with make_client(settings, app_env="production", app_api_key="prod-secret") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/scenarios").status_code == 401
        assert (
            client.get("/scenarios", headers={"X-API-Key": "prod-secret"}).status_code
            == 200
        )


# --- FR-03 / AC-04: rate limits ---------------------------------------------


def test_ac04_the_extra_request_is_429_and_other_clients_are_unaffected(
    settings: Settings,
) -> None:
    app = make_app(settings, rate_limit_start_per_minute=2)
    with TestClient(app, client=("198.51.100.10", 1)) as limited, TestClient(
        app, client=("198.51.100.11", 2)
    ) as other:
        assert limited.post("/session/start", json={"scenario_id": "chest_pain"}).status_code == 200
        assert limited.post("/session/start", json={"scenario_id": "chest_pain"}).status_code == 200
        rejected = limited.post("/session/start", json={"scenario_id": "chest_pain"})
        assert rejected.status_code == 429
        assert rejected.headers["Retry-After"].isdigit()
        assert int(rejected.headers["Retry-After"]) >= 1
        assert "detail" in rejected.json()

        allowed = other.post("/session/start", json={"scenario_id": "chest_pain"})
        assert allowed.status_code == 200, allowed.text


def test_ac04_turn_limit_is_shared_by_message_action_and_hint(
    settings: Settings, tmp_path: Path
) -> None:
    with make_client(
        settings,
        rate_limit_turn_per_minute=2,
        guidelines_path=write_guidelines(tmp_path, safe_steps()),
    ) as client:
        started = start(client)
        headers = session_headers(started)
        assert (
            client.post(
                f"/session/{started['session_id']}/message",
                json={"message": "When did it start?"},
                headers=headers,
            ).status_code
            == 200
        )
        assert (
            client.post(
                f"/session/{started['session_id']}/action",
                json={"action_id": "vitals"},
                headers=headers,
            ).status_code
            == 200
        )
        blocked = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "pain"},
            headers=headers,
        )
        assert blocked.status_code == 429
        assert blocked.headers["Retry-After"].isdigit()


def test_ac04_probe_limit_covers_both_probe_endpoints(settings: Settings) -> None:
    with make_client(settings, rate_limit_probe_per_minute=2) as client:
        assert client.get("/health", params={"probe": "true"}).status_code == 200
        assert client.get("/health/llm").status_code == 200
        limited = client.get("/health/llm")
        assert limited.status_code == 429
        assert limited.headers["Retry-After"].isdigit()


def test_ac04_whatsapp_limit_is_per_sender(
    client: TestClient, settings: Settings, fake_whatsapp: FakeWhatsApp
) -> None:
    limited_phone = "27827777777"
    other_phone = "27828888888"
    client.app.state.rate_limiter.whatsapp.limit = 1

    post_payload(client, settings, make_payload(limited_phone, "START", "wamid.qa.rl.start"))
    before = len(fake_whatsapp.sent)
    post_payload(client, settings, make_payload(limited_phone, "Hello", "wamid.qa.rl.second"))
    assert len(fake_whatsapp.sent) == before + 1
    assert "slow down" in fake_whatsapp.sent[-1][1].lower()

    sent_to_other_before = len(fake_whatsapp.sent)
    post_payload(client, settings, make_payload(other_phone, "START", "wamid.qa.rl.other"))
    assert len(fake_whatsapp.sent) == sent_to_other_before + 2
    assert all(recipient == other_phone for recipient, _ in fake_whatsapp.sent[sent_to_other_before:])


# --- FR-04 / AC-05: transcript retention ------------------------------------


def _write_transcript(path: Path, age_days: float) -> None:
    path.write_text("{}\n", encoding="utf-8")
    stamp = time.time() - age_days * 24 * 60 * 60
    os.utime(path, (stamp, stamp))


def test_ac05_startup_and_session_start_delete_only_expired_transcripts(
    settings: Settings,
) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    expired = settings.log_dir / "expired.jsonl"
    fresh = settings.log_dir / "fresh.jsonl"
    _write_transcript(expired, age_days=31)
    _write_transcript(fresh, age_days=1)

    app = make_app(settings, transcript_retention_days=30)
    assert not expired.exists()
    assert fresh.exists()

    late_expired = settings.log_dir / "late-expired.jsonl"
    late_fresh = settings.log_dir / "late-fresh.jsonl"
    _write_transcript(late_expired, age_days=31)
    _write_transcript(late_fresh, age_days=1)
    with TestClient(app) as client:
        start(client)
    assert not late_expired.exists()
    assert late_fresh.exists()


def test_ac05_zero_retention_keeps_transcripts(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    ancient = settings.log_dir / "ancient.jsonl"
    _write_transcript(ancient, age_days=400)
    make_app(settings, transcript_retention_days=0)
    assert ancient.exists()


# --- FR-05 / AC-06: hints never reveal the diagnosis ------------------------


def test_ac06_hint_skips_forbidden_steps_case_insensitively(
    settings: Settings, tmp_path: Path
) -> None:
    steps = [
        {"id": "step-1", "text": "Consider an ECTOPIC pregnancy and arrange urgent review."},
        {"id": "step-2", "text": "Take vital signs and record them."},
        {"id": "step-3", "text": "Ask about the pattern of the pain."},
        {"id": "step-4", "text": "Explain the plan to the patient."},
        {"id": "step-5", "text": "Check for pregnancy if it is safe to do so."},
    ]
    forbidden = ["ectopic pregnancy", "ectopic"]
    with make_client(
        settings, guidelines_path=write_guidelines(tmp_path, steps)
    ) as client:
        started = start(client)
        headers = session_headers(started)

        only_forbidden = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "ECTOPIC"},
            headers=headers,
        )
        assert only_forbidden.status_code == 200, only_forbidden.text
        hint = only_forbidden.json()["hint"]
        assert hint == "Check the guideline: Abdominal pain."
        assert not any(term in hint.lower() for term in forbidden)

        mixed = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "ectopic vital signs"},
            headers=headers,
        )
        assert mixed.status_code == 200, mixed.text
        assert mixed.json()["hint"] == "Take vital signs and record them."
        assert mixed.json()["step_ids"] == ["step-2"]
        assert not any(term in mixed.json()["hint"].lower() for term in forbidden)


# --- API contract -----------------------------------------------------------


def test_contract_start_and_monitor_shapes(client: TestClient) -> None:
    started = start(client)
    assert {
        "session_id",
        "session_token",
        "scenario",
        "opening_message",
        "student_turns",
        "max_student_turns",
        "turns_remaining",
        "status",
        "difficulty",
        "hints_enabled",
        "monitor",
        "actions",
        "guideline",
    } <= set(started)
    assert set(started["scenario"]) >= {
        "id",
        "title",
        "setting",
        "difficulty",
        "description",
        "patient_label",
        "learning_objectives",
    }
    monitor = started["monitor"]
    assert set(monitor) >= {
        "vitals",
        "clinical_minutes",
        "monitor_minutes",
        "status",
        "alerts",
        "escalated",
        "escalation_minutes",
        "escalation_target_minutes",
    }
    assert set(monitor["vitals"]) == {
        "heart_rate",
        "systolic_bp",
        "diastolic_bp",
        "spo2",
        "respiratory_rate",
        "pain_score",
        "temperature",
    }
    assert monitor["status"] in {"stable", "deteriorating", "critical"}
    assert started["status"] == "active"
    assert started["student_turns"] == 0
    assert started["turns_remaining"] == started["max_student_turns"]
    for action in started["actions"]:
        assert {"id", "label", "description", "minutes", "category", "available"} <= set(action)


def test_contract_error_shapes_and_validation(client: TestClient) -> None:
    unknown_scenario = client.post("/session/start", json={"scenario_id": "not-a-scenario"})
    assert unknown_scenario.status_code == 404
    assert set(unknown_scenario.json()) == {"detail"}

    missing_session = client.get("/session/not-a-session")
    assert missing_session.status_code == 404
    assert missing_session.json() == {"detail": "Session not found."}

    started = start(client)
    headers = session_headers(started)
    session_id = started["session_id"]

    assert (
        client.post("/session/start", json={"scenario_id": "x" * 81}).status_code == 422
    )
    assert (
        client.post(
            "/session/start", json={"scenario_id": "chest_pain", "student_name": "x" * 81}
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/session/{session_id}/message", json={"message": ""}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/session/{session_id}/message", json={"message": "x" * 2001}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/session/{session_id}/action", json={"action_id": "x" * 41}, headers=headers
        ).status_code
        == 422
    )
    assert (
        client.post(
            f"/session/{session_id}/hint", json={"question": "x" * 2001}, headers=headers
        ).status_code
        == 422
    )


def test_contract_completed_session_conflicts(client: TestClient) -> None:
    started = start(client)
    headers = session_headers(started)
    session_id = started["session_id"]

    assert client.post(f"/session/{session_id}/end", headers=headers).status_code == 200
    for method, path, body in [
        ("POST", f"/session/{session_id}/message", {"message": "One more?"}),
        ("POST", f"/session/{session_id}/action", {"action_id": "vitals"}),
        ("POST", f"/session/{session_id}/hint", None),
    ]:
        response = client.request(method, path, json=body, headers=headers)
        assert response.status_code == 409, (path, response.text)
        assert set(response.json()) == {"detail"}

    snapshot = client.get(f"/session/{session_id}", headers=headers)
    assert snapshot.status_code == 200
    assert snapshot.json()["status"] == "completed"


def test_contract_api_key_protects_body_but_leaves_open_pages(settings: Settings) -> None:
    with make_client(settings, app_api_key="secret-key") as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").status_code == 200

        for method, path, body in [
            ("GET", "/scenarios", None),
            ("GET", "/health/llm", None),
            ("GET", "/health?probe=true", None),
            ("POST", "/session/start", {"scenario_id": "chest_pain"}),
        ]:
            response = client.request(method, path, json=body)
            assert response.status_code == 401, (path, response.text)
            assert set(response.json()) == {"detail"}

        started = client.post(
            "/session/start",
            json={"scenario_id": "chest_pain"},
            headers={"X-API-Key": "secret-key"},
        ).json()
        assert (
            client.get(
                f"/session/{started['session_id']}",
                headers={
                    "X-API-Key": "secret-key",
                    "X-Session-Token": started["session_token"],
                },
            ).status_code
            == 200
        )
