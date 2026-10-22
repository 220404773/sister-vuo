"""QA round 2: acceptance-criterion and API-contract tests for the round-2 fixes.

These complement tests/test_qa.py (AC-01..AC-06, contract) and
tests/test_round2_fixes.py (the Medium security fixes). Every test here maps to a
line of the requirements pack; a failure is a defect, not a flake.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.llm import LLMUpstreamError, sanitize_evaluator_output
from app.main import create_app
from app.ratelimit import SlidingWindowLimiter

from .conftest import FakeLLM, FakeWhatsApp

VALID_URL = "https://knowledgehub.health.gov.za/elibrary/adult-primary-care-guide-apc-2026"
ROOT = PROJECT_ROOT


def make_client(settings: Settings, llm: FakeLLM | None = None, **changes) -> TestClient:
    app = create_app(
        settings_obj=replace(settings, **changes),
        llm_service=llm or FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    return TestClient(app)


def start(client: TestClient, scenario_id: str = "abdominal_pain") -> dict:
    response = client.post("/session/start", json={"scenario_id": scenario_id})
    assert response.status_code == 200, response.text
    return response.json()


def token_headers(started: dict) -> dict[str, str]:
    return {"X-Session-Token": started["session_token"]}


def guidelines_file(tmp_path: Path, steps: list[dict[str, str]]) -> Path:
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


def steps_with_a_forbidden_one() -> list[dict[str, str]]:
    return [
        {"id": "step-1", "text": "Consider an ectopic pregnancy and arrange urgent review."},
        {"id": "step-2", "text": "Take vital signs and record them."},
        {"id": "step-3", "text": "Ask about the pattern of the pain."},
        {"id": "step-4", "text": "Check for pregnancy if it is safe to do so."},
        {"id": "step-5", "text": "Explain the plan to the patient."},
    ]


# --- FR-01 / AC-02: the session token stays out of logs and snapshots ---------


def test_ac02_token_never_appears_in_logs_or_snapshots(
    client: TestClient, settings: Settings
) -> None:
    started = start(client)
    other = start(client)
    session_id = started["session_id"]
    headers = token_headers(started)
    token = started["session_token"]

    client.post(
        f"/session/{session_id}/message",
        json={"message": "When did the pain start?"},
        headers=headers,
    )
    client.post(f"/session/{session_id}/action", json={"action_id": "vitals"}, headers=headers)
    client.post(f"/session/{session_id}/hint", json={"question": "pain"}, headers=headers)
    snapshot = client.get(f"/session/{session_id}", headers=headers)
    client.post(f"/session/{session_id}/end", headers=headers)
    ended_snapshot = client.get(f"/session/{session_id}", headers=headers)

    assert token not in snapshot.text
    assert token not in ended_snapshot.text
    assert "session_token" not in snapshot.text
    assert "session_token" not in ended_snapshot.text

    own_log = (settings.log_dir / f"{session_id}.jsonl").read_text(encoding="utf-8")
    other_log = (settings.log_dir / f"{other['session_id']}.jsonl").read_text(encoding="utf-8")
    assert token not in own_log
    assert other["session_token"] not in own_log
    assert token not in other_log
    assert "session_token" not in own_log


# --- FR-01 / AC-01: token still required when the shared API key is set --------


def test_ac01_session_token_is_required_on_top_of_the_api_key(settings: Settings) -> None:
    with make_client(settings, app_api_key="secret-key") as client:
        key = {"X-API-Key": "secret-key"}
        started = client.post(
            "/session/start", json={"scenario_id": "abdominal_pain"}, headers=key
        ).json()
        session_id = started["session_id"]

        no_token = client.get(f"/session/{session_id}", headers=key)
        wrong_token = client.get(
            f"/session/{session_id}",
            headers={**key, "X-Session-Token": "wrong-token"},
        )
        assert no_token.status_code == 404
        assert wrong_token.status_code == 404
        assert no_token.json() == wrong_token.json() == {"detail": "Session not found."}

        ok = client.get(
            f"/session/{session_id}",
            headers={**key, "X-Session-Token": started["session_token"]},
        )
        assert ok.status_code == 200


# --- FR-02 / AC-03: production fails closed, documented configuration ---------


def test_fr02_render_and_docs_fail_closed(settings: Settings) -> None:
    render = (ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "key: APP_ENV" in render
    assert "value: production" in render
    assert "--proxy-headers" in render

    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for name in (
        "APP_ENV",
        "RATE_LIMIT_START_PER_MINUTE",
        "RATE_LIMIT_TURN_PER_MINUTE",
        "RATE_LIMIT_PROBE_PER_MINUTE",
        "RATE_LIMIT_WHATSAPP_PER_MINUTE",
        "TRANSCRIPT_RETENTION_DAYS",
    ):
        assert name in env_example

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "APP_ENV=production" in readme
    assert "TRANSCRIPT_RETENTION_DAYS" in readme
    assert "--proxy-headers" in readme


# --- FR-03 / AC-04: the sliding window actually expires ----------------------


def test_ac04_limiter_window_expires_and_is_per_key() -> None:
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60)
    assert limiter.check("client-a", now=0) is None
    assert limiter.check("client-a", now=10) is None
    blocked = limiter.check("client-a", now=20)
    assert blocked is not None and blocked >= 1
    assert limiter.check("client-b", now=20) is None

    # The window slides: event 0 expires by now=61, event 10 only by now=71.
    assert limiter.check("client-a", now=61) is None
    assert limiter.check("client-a", now=62) is not None
    assert limiter.check("client-a", now=71) is None


def test_ac04_probe_limit_is_per_client(settings: Settings) -> None:
    app = create_app(
        settings_obj=replace(settings, rate_limit_probe_per_minute=1),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    with TestClient(app, client=("198.51.100.21", 1)) as limited, TestClient(
        app, client=("198.51.100.22", 2)
    ) as other:
        assert limited.get("/health", params={"probe": "true"}).status_code == 200
        assert limited.get("/health", params={"probe": "true"}).status_code == 429
        assert other.get("/health", params={"probe": "true"}).status_code == 200


# --- FR-04 / AC-05: retention also runs when the app shuts down -------------


def test_ac05_shutdown_sweep_deletes_expired_transcripts(settings: Settings) -> None:
    app = create_app(
        settings_obj=replace(settings, transcript_retention_days=30),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        old = settings.log_dir / "shutdown-old.jsonl"
        new = settings.log_dir / "shutdown-new.jsonl"
        old.write_text("{}\n", encoding="utf-8")
        new.write_text("{}\n", encoding="utf-8")
        stamp = time.time() - 31 * 24 * 60 * 60
        os.utime(old, (stamp, stamp))
        assert old.exists()

    assert not old.exists()
    assert new.exists()


# --- FR-05 / AC-06: forbidden steps are hidden from hints but score ---------


def test_ac06_forbidden_steps_are_still_sent_to_the_evaluator(
    settings: Settings, tmp_path: Path
) -> None:
    llm = FakeLLM()
    with make_client(
        settings,
        llm=llm,
        guidelines_path=guidelines_file(tmp_path, steps_with_a_forbidden_one()),
    ) as client:
        started = start(client)
        headers = token_headers(started)

        hint = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "ectopic"},
            headers=headers,
        )
        assert hint.status_code == 200, hint.text
        assert "ectopic" not in hint.json()["hint"].lower()

        ended = client.post(f"/session/{started['session_id']}/end", headers=headers)
        assert ended.status_code == 200, ended.text

    assert llm.evaluation_contexts, "the evaluator was not called"
    context = llm.evaluation_contexts[0]
    assert context is not None
    scoring_steps = {step["id"]: step["text"] for step in context["guideline_steps"]}
    assert "step-1" in scoring_steps
    assert "ectopic" in scoring_steps["step-1"].lower()


def test_ac06_round2_sanitiser_removes_a_leaked_diagnosis() -> None:
    scenario = json.loads(
        (ROOT / "scenarios" / "abdominal_pain.json").read_text(encoding="utf-8")
    )
    from app.models import DomainScores, Evaluation

    evaluation = Evaluation(
        overall_score=4.0,
        domain_scores=DomainScores(
            red_flag_recognition=4,
            history_completeness=4,
            prioritisation=4,
            safety=4,
            communication_empathy=4,
            time_to_escalation=4,
        ),
        strengths=["Good rapport."],
        missed_red_flags=[],
        unsafe_actions=[],
        recommended_learning=["Review ectopic pregnancy management."],
        feedback_to_student="Good start. The diagnosis is ectopic pregnancy.",
    )
    filtered = sanitize_evaluator_output(
        evaluation, scenario, [{"role": "student", "content": "When did it start?"}]
    )
    assert "ectopic" not in filtered.feedback_to_student.lower()
    assert all("ectopic" not in item.lower() for item in filtered.recommended_learning)
    assert "Good start" in filtered.feedback_to_student


# --- API contract: validation boundaries, unicode and LLM failure shape ------


def test_contract_boundary_values_and_unicode_are_accepted(client: TestClient) -> None:
    started = client.post(
        "/session/start",
        json={"scenario_id": "abdominal_pain", "student_name": "n" * 80},
    )
    assert started.status_code == 200, started.text
    headers = {"X-Session-Token": started.json()["session_token"]}
    session_id = started.json()["session_id"]

    message = "Pain 😀 where? " + "x" * 1970
    assert len(message) <= 2000
    response = client.post(
        f"/session/{session_id}/message", json={"message": message}, headers=headers
    )
    assert response.status_code == 200, response.text
    snapshot = client.get(f"/session/{session_id}", headers=headers)
    assert message in [turn["content"] for turn in snapshot.json()["transcript"]]

    too_long = client.post(
        f"/session/{session_id}/message", json={"message": "x" * 2001}, headers=headers
    )
    assert too_long.status_code == 422


def test_contract_llm_failure_returns_502_with_short_detail(settings: Settings) -> None:
    class FailingLLM(FakeLLM):
        async def patient_reply(self, scenario, transcript, revealed_topics=()):
            raise LLMUpstreamError("The model provider is unavailable.")

    with make_client(settings, llm=FailingLLM()) as client:
        started = start(client)
        response = client.post(
            f"/session/{started['session_id']}/message",
            json={"message": "When did it start?"},
            headers=token_headers(started),
        )
        assert response.status_code == 502
        assert set(response.json()) == {"detail"}
        assert response.json()["detail"]


def test_round2_security_headers_are_on_error_responses(client: TestClient) -> None:
    response = client.get("/session/not-a-session")
    assert response.status_code == 404
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
