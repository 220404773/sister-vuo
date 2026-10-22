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
from app.ratelimit import SlidingWindowLimiter

from .conftest import FakeLLM, FakeWhatsApp
from .test_whatsapp import make_payload, post_payload

VALID_URL = "https://knowledgehub.health.gov.za/elibrary/adult-primary-care-guide-apc-2026"


def make_client(settings: Settings, **changes) -> TestClient:
    app = create_app(
        settings_obj=replace(settings, **changes),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    return TestClient(app)


def start(client: TestClient) -> dict:
    response = client.post("/session/start", json={"scenario_id": "abdominal_pain"})
    assert response.status_code == 200, response.text
    return response.json()


def token_headers(started: dict) -> dict[str, str]:
    return {"X-Session-Token": started["session_token"]}


def test_session_routes_reject_a_missing_or_wrong_token(client: TestClient) -> None:
    started = start(client)
    session_id = started["session_id"]

    missing = client.post(
        f"/session/{session_id}/message", json={"message": "How long?"}
    )
    assert missing.status_code == 404
    assert missing.json()["detail"] == "Session not found."

    wrong = client.get(
        f"/session/{session_id}", headers={"X-Session-Token": "not-the-token"}
    )
    assert wrong.status_code == 404
    assert wrong.json()["detail"] == "Session not found."

    right = client.get(f"/session/{session_id}", headers=token_headers(started))
    assert right.status_code == 200
    assert right.json()["session_id"] == session_id


def test_production_refuses_to_start_without_an_api_key(settings: Settings) -> None:
    with pytest.raises(RuntimeError):
        create_app(
            settings_obj=replace(settings, app_env="production", app_api_key=""),
            llm_service=FakeLLM(),
            whatsapp_client=FakeWhatsApp(),
        )

    with make_client(settings, app_env="production", app_api_key="secret-key") as client:
        assert client.get("/health").status_code == 200


def test_start_rate_limit_returns_429_with_retry_after(settings: Settings) -> None:
    with make_client(settings, rate_limit_start_per_minute=2) as client:
        for _ in range(2):
            assert (
                client.post(
                    "/session/start", json={"scenario_id": "abdominal_pain"}
                ).status_code
                == 200
            )
        limited = client.post(
            "/session/start", json={"scenario_id": "abdominal_pain"}
        )
        assert limited.status_code == 429
        assert limited.headers["Retry-After"]


def test_turn_rate_limit_is_shared_by_message_action_and_hint(
    settings: Settings,
) -> None:
    with make_client(settings, rate_limit_turn_per_minute=1) as client:
        started = start(client)
        headers = token_headers(started)
        first = client.post(
            f"/session/{started['session_id']}/message",
            json={"message": "How long has it hurt?"},
            headers=headers,
        )
        assert first.status_code == 200
        second = client.post(
            f"/session/{started['session_id']}/action",
            json={"action_id": "vitals"},
            headers=headers,
        )
        assert second.status_code == 429
        assert second.headers["Retry-After"]


def test_limiter_keeps_clients_independent_and_memory_bounded() -> None:
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60, max_keys=2)
    assert limiter.check("one", now=0) is None
    assert limiter.check("one", now=1) is None
    assert limiter.check("one", now=2) is not None
    assert limiter.check("two", now=2) is None
    assert limiter.check("three", now=3) is None
    assert len(limiter._events) == 2


def test_transcript_retention_deletes_only_old_files(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    old = settings.log_dir / "old.jsonl"
    new = settings.log_dir / "new.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    new.write_text("{}\n", encoding="utf-8")
    old_time = time.time() - 31 * 24 * 60 * 60
    os.utime(old, (old_time, old_time))

    app = create_app(
        settings_obj=settings,
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    assert not old.exists()
    assert new.exists()

    later_old = settings.log_dir / "later-old.jsonl"
    later_old.write_text("{}\n", encoding="utf-8")
    os.utime(later_old, (old_time, old_time))
    with TestClient(app) as client:
        start(client)
    assert not later_old.exists()


def test_retention_zero_keeps_transcripts(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    old = settings.log_dir / "old.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    old_time = time.time() - 400 * 24 * 60 * 60
    os.utime(old, (old_time, old_time))
    create_app(
        settings_obj=replace(settings, transcript_retention_days=0),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    assert old.exists()


def test_whatsapp_rate_limit_sends_one_slow_down_reply(settings: Settings) -> None:
    with make_client(settings, rate_limit_whatsapp_per_minute=1) as client:
        phone = "27826666666"
        post_payload(client, settings, make_payload(phone, "START", "wamid.rl.start"))
        sent_before = len(client.app.state.whatsapp.sent)
        post_payload(
            client, settings, make_payload(phone, "How long?", "wamid.rl.question")
        )
        sent = client.app.state.whatsapp.sent
        assert len(sent) == sent_before + 1
        assert "slow down" in sent[-1][1].lower()
        session = client.app.state.manager.get_active_by_phone(phone)
        assert session is not None
        assert session.student_turns == 0


def write_guidelines(tmp_path: Path) -> Path:
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
                "steps": [
                    {
                        "id": "step-1",
                        "text": "Consider an ectopic pregnancy and arrange urgent review.",
                    },
                    {"id": "step-2", "text": "Take vital signs and record them."},
                    {"id": "step-3", "text": "Ask about the pattern of the pain."},
                    {"id": "step-4", "text": "Check for pregnancy if it is safe to do so."},
                    {"id": "step-5", "text": "Explain the plan to the patient."},
                ],
            }
        },
    }
    path = tmp_path / "guidelines.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_hint_skips_steps_that_reveal_the_diagnosis(
    settings: Settings, tmp_path: Path
) -> None:
    app = create_app(
        settings_obj=replace(settings, guidelines_path=write_guidelines(tmp_path)),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    with TestClient(app) as client:
        started = start(client)
        headers = token_headers(started)

        only_forbidden = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "ectopic"},
            headers=headers,
        )
        assert only_forbidden.status_code == 200
        assert "ectopic" not in only_forbidden.json()["hint"].lower()
        assert only_forbidden.json()["hint"] == "Check the guideline: Abdominal pain."

        mixed = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "ectopic vital signs"},
            headers=headers,
        )
        assert mixed.status_code == 200
        assert "ectopic" not in mixed.json()["hint"].lower()
        assert mixed.json()["hint"] == "Take vital signs and record them."
