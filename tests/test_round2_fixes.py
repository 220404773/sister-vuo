from __future__ import annotations

import asyncio
import os
import time
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.llm import sanitize_evaluator_output
from app.main import _retention_sweeper, create_app
from app.models import DomainScores, Evaluation
from app.ratelimit import SlidingWindowLimiter
from app.sessions import ScenarioRepository, TranscriptLogger

from .conftest import FakeLLM, FakeWhatsApp


def make_client(settings: Settings, **changes) -> TestClient:
    app = create_app(
        settings_obj=replace(settings, **changes),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    return TestClient(app)


def leaky_evaluation(feedback: str, recommendation: str) -> Evaluation:
    return Evaluation(
        overall_score=4.0,
        domain_scores=DomainScores(
            red_flag_recognition=4,
            history_completeness=4,
            prioritisation=4,
            safety=4,
            communication_empathy=4,
            time_to_escalation=4,
        ),
        strengths=["Built rapport quickly."],
        missed_red_flags=[],
        unsafe_actions=[],
        recommended_learning=[recommendation, "Ask about the pain pattern."],
        feedback_to_student=feedback,
    )


def test_evaluator_output_cannot_leak_the_diagnosis() -> None:
    scenario = ScenarioRepository(PROJECT_ROOT / "scenarios").get("abdominal_pain")
    transcript = [
        {"role": "patient", "content": "It hurts."},
        {"role": "student", "content": "When did the pain start?"},
    ]
    evaluation = leaky_evaluation(
        "Good rapport. The diagnosis is ectopic pregnancy. Keep asking open questions.",
        "Review ectopic pregnancy management.",
    )

    filtered = sanitize_evaluator_output(evaluation, scenario, transcript)

    assert "ectopic" not in filtered.feedback_to_student.lower()
    assert "Good rapport" in filtered.feedback_to_student
    assert "open questions" in filtered.feedback_to_student
    assert filtered.recommended_learning == ["Ask about the pain pattern."]


def test_evaluator_may_use_a_term_the_student_already_named() -> None:
    scenario = ScenarioRepository(PROJECT_ROOT / "scenarios").get("abdominal_pain")
    transcript = [
        {"role": "student", "content": "Could this be an ectopic pregnancy?"},
    ]
    evaluation = leaky_evaluation(
        "You correctly considered ectopic pregnancy. Next, arrange urgent review.",
        "Revisit ectopic pregnancy red flags.",
    )

    filtered = sanitize_evaluator_output(evaluation, scenario, transcript)

    assert "ectopic pregnancy" in filtered.feedback_to_student.lower()
    assert filtered.recommended_learning


def test_limiter_does_not_evict_an_over_limit_key() -> None:
    limiter = SlidingWindowLimiter(limit=1, window_seconds=60, max_keys=2)
    assert limiter.check("a", now=0) is None
    assert limiter.check("b", now=0) is None
    assert limiter.check("a", now=1) is not None
    assert limiter.check("c", now=2) is None
    assert "a" in limiter._events
    assert limiter.check("a", now=3) is not None


def test_scenarios_endpoint_is_rate_limited(settings: Settings) -> None:
    with make_client(settings, rate_limit_start_per_minute=1) as client:
        assert client.get("/scenarios").status_code == 200
        limited = client.get("/scenarios")
        assert limited.status_code == 429
        assert limited.headers["Retry-After"]


def test_responses_carry_security_headers(client: TestClient) -> None:
    for path in ("/", "/health"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
        assert response.headers["Strict-Transport-Security"]


def test_retention_sweeper_deletes_expired_transcripts(
    settings: Settings, tmp_path: Path
) -> None:
    log_dir = tmp_path / "sweep-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    old = log_dir / "old.jsonl"
    new = log_dir / "new.jsonl"
    old.write_text("{}\n", encoding="utf-8")
    new.write_text("{}\n", encoding="utf-8")
    old_time = time.time() - 31 * 24 * 60 * 60
    os.utime(old, (old_time, old_time))

    async def run_once() -> None:
        task = asyncio.create_task(
            _retention_sweeper(TranscriptLogger(log_dir), 30, interval_seconds=0.01)
        )
        try:
            await asyncio.sleep(0.05)
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    asyncio.run(run_once())

    assert not old.exists()
    assert new.exists()
