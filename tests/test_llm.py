from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.llm import (
    LLMAuthenticationError,
    LLMConnectionError,
    LLMEvaluationError,
    LLMService,
    LLMTimeoutError,
    LLMUpstreamError,
    build_patient_prompt,
    limit_sentences,
    parse_evaluation,
    sanitize_patient_reply,
)
from app.sessions import ScenarioRepository

ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = ScenarioRepository(ROOT / "scenarios")
PATIENT_TEMPLATE = (ROOT / "prompts" / "patient.txt").read_text(encoding="utf-8")


def test_patient_prompt_does_not_leak_hidden_diagnosis() -> None:
    scenario = SCENARIOS.get("abdominal_pain")
    prompt = build_patient_prompt(PATIENT_TEMPLATE, scenario)
    assert "Sister" in prompt or "Naledi" in prompt
    assert "ectopic" not in prompt.lower()
    assert scenario["ground_truth"]["diagnosis"].lower() not in prompt.lower()


def test_patient_prompt_only_reveals_unlocked_topics() -> None:
    scenario = SCENARIOS.get("abdominal_pain")
    prompt = build_patient_prompt(PATIENT_TEMPLATE, scenario, ["onset"])
    assert "Sudden onset about six hours ago" in prompt
    assert "shoulder tip" not in prompt.lower()


def test_patient_prompt_uses_caregiver_role_for_child() -> None:
    scenario = SCENARIOS.get("paediatric_dehydration")
    prompt = build_patient_prompt(PATIENT_TEMPLATE, scenario)
    assert "Speak as the child's caregiver" in prompt
    assert "Tshepo" in prompt


def test_patient_reply_is_limited_to_three_sentences() -> None:
    reply = limit_sentences("First sentence. Second sentence. Third sentence. Fourth sentence.")
    assert reply == "First sentence. Second sentence. Third sentence."


def test_patient_reply_blocks_forbidden_diagnosis_terms() -> None:
    scenario = SCENARIOS.get("abdominal_pain")
    reply = sanitize_patient_reply("I think I have an ectopic pregnancy.", scenario)
    assert "ectopic" not in reply.lower()
    assert "I don't know" in reply


def test_patient_reply_strips_role_prefix() -> None:
    scenario = SCENARIOS.get("abdominal_pain")
    reply = sanitize_patient_reply("Patient: It started this morning.", scenario)
    assert reply == "It started this morning."


def test_parse_evaluation_with_fenced_json_and_recomputed_overall() -> None:
    payload = {
        "overall_score": 5.0,
        "domain_scores": {
            "red_flag_recognition": 4,
            "history_completeness": 4,
            "prioritisation": 3,
            "safety": 5,
            "communication_empathy": 5,
            "time_to_escalation": 2,
        },
        "strengths": ["Clear introduction."],
        "missed_red_flags": [],
        "unsafe_actions": [],
        "critical_errors": [],
        "recommended_learning": ["Escalate earlier."],
        "feedback_to_student": "You were kind and organised.",
    }
    raw = "Here is the result:\n```json\n" + json.dumps(payload) + "\n```"
    evaluation = parse_evaluation(raw)
    assert evaluation.overall_score == 3.8
    assert evaluation.domain_scores.red_flag_recognition == 4
    assert evaluation.feedback_to_student == "You were kind and organised."


def test_parse_evaluation_clamps_scores_and_normalises_lists() -> None:
    payload = {
        "domain_scores": {
            "red_flag_recognition": "6/5",
            "history_completeness": -2,
            "prioritisation": "4",
            "safety": 4.6,
            "communication_empathy": 4,
            "time_to_escalation": "3",
        },
        "strengths": "Used open questions.",
        "missed_red_flags": None,
        "unsafe_actions": [],
        "critical_errors": ["Did not call for help."],
        "recommended_learning": ["Review red flags."],
        "feedback_to_student": "Good start.",
    }
    evaluation = parse_evaluation(json.dumps(payload))
    assert evaluation.domain_scores.red_flag_recognition == 5
    assert evaluation.domain_scores.history_completeness == 0
    assert evaluation.domain_scores.safety == 5
    assert evaluation.overall_score == 3.5
    assert evaluation.strengths == ["Used open questions."]
    assert evaluation.missed_red_flags == []
    assert evaluation.critical_errors == ["Did not call for help."]


def test_parse_evaluation_rejects_missing_domain() -> None:
    payload = {
        "domain_scores": {
            "red_flag_recognition": 4,
            "history_completeness": 3,
            "prioritisation": 4,
            "communication_empathy": 4,
            "time_to_escalation": 4,
        },
        "feedback_to_student": "Good.",
    }
    with pytest.raises(LLMEvaluationError):
        parse_evaluation(json.dumps(payload))


EVALUATION_PAYLOAD = {
    "overall_score": 4.0,
    "domain_scores": {
        "red_flag_recognition": 4,
        "history_completeness": 4,
        "prioritisation": 4,
        "safety": 4,
        "communication_empathy": 4,
        "time_to_escalation": 4,
    },
    "strengths": ["Clear questions."],
    "missed_red_flags": [],
    "unsafe_actions": [],
    "critical_errors": [],
    "recommended_learning": ["Escalate sooner."],
    "feedback_to_student": "A solid consultation.",
}


def make_settings(json_mode: bool = False) -> Settings:
    return Settings(
        llm_base_url="http://llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        llm_json_mode=json_mode,
        scenario_dir=ROOT / "scenarios",
        prompt_dir=ROOT / "prompts",
        log_dir=ROOT / "logs" / "test",
    )


def test_llm_service_sends_correct_temperatures_and_parses() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        system_text = payload["messages"][0]["content"]
        if "senior nursing educator" in system_text:
            content = "```json\n" + json.dumps(EVALUATION_PAYLOAD) + "\n```"
        else:
            content = "Patient: It started this morning. It is getting worse. I feel light-headed."
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    async def run() -> tuple[str, float]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = LLMService(make_settings(), client=client)
        scenario = SCENARIOS.get("abdominal_pain")
        transcript = [
            {"role": "patient", "content": scenario["opening_line"]},
            {"role": "student", "content": "When did the pain start?"},
        ]
        reply = await service.patient_reply(scenario, transcript)
        evaluation = await service.evaluate(scenario, transcript)
        await service.close()
        return reply, evaluation.overall_score

    reply, overall = asyncio.run(run())
    assert reply == "It started this morning. It is getting worse. I feel light-headed."
    assert overall == 4.0
    assert str(requests[0]["temperature"]) == "0.7"
    assert str(requests[1]["temperature"]) == "0.2"
    assert requests[0]["model"] == "test-model"
    assert "response_format" not in requests[1]


def test_llm_service_uses_json_mode_when_enabled() -> None:
    requests: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        if "senior nursing educator" in payload["messages"][0]["content"]:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": json.dumps(EVALUATION_PAYLOAD)}}]},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "I feel unwell."}}]},
        )

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = LLMService(make_settings(json_mode=True), client=client)
        scenario = SCENARIOS.get("chest_pain")
        await service.evaluate(
            scenario,
            [
                {"role": "patient", "content": scenario["opening_line"]},
                {"role": "student", "content": "How long has this been going on?"},
            ],
        )
        await service.close()

    asyncio.run(run())
    assert requests[0]["response_format"] == {"type": "json_object"}


def test_llm_service_retries_invalid_evaluator_json() -> None:
    evaluator_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal evaluator_attempts
        payload = json.loads(request.content)
        if "senior nursing educator" in payload["messages"][0]["content"]:
            evaluator_attempts += 1
            content = "not json at all" if evaluator_attempts == 1 else json.dumps(EVALUATION_PAYLOAD)
            return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "I feel unwell."}}]})

    async def run() -> float:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = LLMService(make_settings(), client=client)
        scenario = SCENARIOS.get("paediatric_dehydration")
        evaluation = await service.evaluate(
            scenario,
            [{"role": "patient", "content": scenario["opening_line"]}],
        )
        await service.close()
        return evaluation.overall_score

    assert asyncio.run(run()) == 4.0
    assert evaluator_attempts == 2


def run_patient_reply(handler) -> tuple[str | None, Exception | None]:
    async def run() -> tuple[str | None, Exception | None]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = LLMService(make_settings(), client=client)
        scenario = SCENARIOS.get("chest_pain")
        try:
            reply = await service.patient_reply(
                scenario,
                [
                    {"role": "patient", "content": scenario["opening_line"]},
                    {"role": "student", "content": "How long has the pain been there?"},
                ],
            )
            return reply, None
        except Exception as exc:  # noqa: BLE001 - surfaced to the assertion
            return None, exc
        finally:
            await service.close()

    return asyncio.run(run())


def test_llm_auth_error_names_the_api_key_and_logs_provider_body(caplog) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"message": "No auth credentials found", "code": 401}},
        )

    caplog.set_level(logging.ERROR, logger="app.llm")
    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMAuthenticationError)
    assert "rejected the API key" in str(error)
    assert "No auth credentials found" in str(error)
    assert error.status_code == 401
    # The operator log must contain the real status and provider body, not just a generic text.
    assert "status=401" in caplog.text
    assert "No auth credentials found" in caplog.text
    assert "test-key" not in caplog.text


def test_llm_model_not_found_points_at_base_url_and_model() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "No endpoints found"}})

    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMUpstreamError)
    assert error.status_code == 404
    assert "LLM_BASE_URL" in str(error)
    assert "LLM_MODEL" in str(error)
    assert "No endpoints found" in str(error)


def test_llm_retries_once_after_rate_limit() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "0"})
        return httpx.Response(200, json={"choices": [{"message": {"content": "It started today."}}]})

    reply, error = run_patient_reply(handler)
    assert error is None
    assert reply == "It started today."
    assert attempts == 2


def test_llm_rate_limit_after_retry_is_clear() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "0"})

    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMUpstreamError)
    assert "rate-limiting" in str(error)
    assert error.status_code == 429


def test_llm_timeout_after_retry_explains_timeout() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("simulated timeout", request=request)

    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMTimeoutError)
    assert "did not respond" in str(error)
    assert attempts == 2


def test_llm_connection_failure_explains_connectivity() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("getaddrinfo failed", request=request)

    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMConnectionError)
    assert "Could not connect" in str(error)
    assert "LLM_BASE_URL" in str(error)


def test_llm_retries_once_after_server_error() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="upstream overloaded")
        return httpx.Response(200, json={"choices": [{"message": {"content": "I feel sick."}}]})

    reply, error = run_patient_reply(handler)
    assert error is None
    assert reply == "I feel sick."
    assert attempts == 2


def test_llm_json_mode_error_mentions_json_mode() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})

    _, error = run_patient_reply(handler)
    assert isinstance(error, LLMUpstreamError)
    assert "LLM_JSON_MODE" in str(error)


def test_probe_returns_structured_success_and_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"choices": [{"message": {"content": "OK"}}]})
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    async def run() -> tuple[dict, dict]:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        service = LLMService(make_settings(), client=client)
        try:
            return await service.probe(), await service.probe()
        finally:
            await service.close()

    ok, failed = asyncio.run(run())
    assert ok["ok"] is True
    assert ok["reply"] == "OK"
    assert failed["ok"] is False
    assert failed["error_type"] == "LLMAuthenticationError"
