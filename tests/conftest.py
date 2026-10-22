from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import PROJECT_ROOT, Settings
from app.llm import LLMUpstreamError
from app.main import create_app
from app.models import DomainScores, Evaluation


class FakeLLM:
    def __init__(self) -> None:
        self.patient_calls: list[list[dict[str, str]]] = []
        self.revealed_calls: list[list[str]] = []
        self.evaluation_calls = 0
        self.evaluation_contexts: list[dict | None] = []
        self.evaluation_failures = 0
        self.protocol_steps = None
        self.probe_calls = 0
        self.probe_result: dict = {
            "ok": True,
            "model": "test-model",
            "latency_ms": 1,
            "reply": "OK",
        }
        self.default_reply = "The pain started this morning and it is getting worse."

    async def probe(self) -> dict:
        self.probe_calls += 1
        return dict(self.probe_result)

    async def patient_reply(self, scenario, transcript, revealed_topics=()):
        self.patient_calls.append(transcript)
        self.revealed_calls.append(list(revealed_topics))
        return self.default_reply

    async def evaluate(self, scenario, transcript, context=None):
        self.evaluation_calls += 1
        self.evaluation_contexts.append(context)
        if self.evaluation_failures > 0:
            self.evaluation_failures -= 1
            raise LLMUpstreamError("simulated evaluator failure")
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
            strengths=["Used open questions and acknowledged the patient's worry."],
            missed_red_flags=["Did not ask about shoulder-tip pain."],
            unsafe_actions=[],
            recommended_learning=["Ask directly about pregnancy and syncope."],
            feedback_to_student=(
                "You built rapport and asked a clear opening question. "
                "Next time, explore the red flags earlier and explain your escalation plan."
            ),
            protocol_steps=self.protocol_steps,
        )

    async def close(self) -> None:
        return None


class FakeWhatsApp:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def send_text(self, to: str, body: str) -> None:
        self.sent.append((to, body))

    async def close(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        scenario_dir=PROJECT_ROOT / "scenarios",
        prompt_dir=PROJECT_ROOT / "prompts",
        static_dir=PROJECT_ROOT / "app" / "static",
        guidelines_path=tmp_path / "guidelines-missing.json",
        log_dir=tmp_path / "logs",
        whatsapp_verify_token="verify-token",
        whatsapp_access_token="access-token",
        whatsapp_phone_number_id="phone-id",
        whatsapp_app_secret="app-secret",
        # Pinned so a developer's real .env (APP_API_KEY, limits) cannot change test results.
        app_env="development",
        app_api_key="",
        session_ttl_minutes=120,
        max_active_sessions=200,
        rate_limit_start_per_minute=10,
        rate_limit_turn_per_minute=60,
        rate_limit_probe_per_minute=6,
        rate_limit_whatsapp_per_minute=30,
        transcript_retention_days=30,
        llm_json_mode=False,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_whatsapp() -> FakeWhatsApp:
    return FakeWhatsApp()


@pytest.fixture
def client(settings: Settings, fake_llm: FakeLLM, fake_whatsapp: FakeWhatsApp):
    app = create_app(
        settings_obj=settings,
        llm_service=fake_llm,
        whatsapp_client=fake_whatsapp,
    )
    with TestClient(app) as test_client:
        yield test_client
