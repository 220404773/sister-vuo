from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.guidelines import GuidelineStep, load_guidelines
from app.main import create_app
from app.models import ProtocolStepResult
from app.scoring import normalize_protocol_steps, protocol_counts
from app.whatsapp import handle_whatsapp_message
from tests.conftest import FakeLLM, FakeWhatsApp

VALID_URL = "https://knowledgehub.health.gov.za/elibrary/adult-primary-care-guide-apc-2026"


def full_guidelines(*, steps: int = 5, scenario_id: str = "chest_pain") -> dict:
    step_defs = [
        ("step-1", "Call for emergency help and stay with the patient."),
        ("step-2", "Give oxygen if the oxygen saturation is below 94 percent."),
        ("step-3", "Record a 12 lead ECG."),
        ("step-4", "Give aspirin 300 mg chewed."),
        ("step-5", "Check vital signs and pain score."),
        ("step-6", "Reassess the patient after treatment."),
        ("step-7", "Document the time of symptom onset."),
        ("step-8", "Explain the next steps to the patient."),
    ]
    return {
        "documents": {
            "apc": {
                "title": "Adult Primary Care Guide (APC)",
                "edition_year": "2026",
                "publisher": "National Department of Health",
                "url": VALID_URL,
            },
            "pathway": {
                "title": "Chest pain pathway",
                "edition_year": "2025",
                "url": None,
            },
        },
        "scenarios": {
            scenario_id: {
                "primary_document": "apc",
                "section_text": "Acute chest pain",
                "steps": [
                    {"id": step_id, "text": text}
                    for step_id, text in step_defs[:steps]
                ],
                "further_reading": ["apc", "pathway"],
            }
        },
    }


def links_only_guidelines() -> dict:
    data = full_guidelines()
    data["scenarios"]["chest_pain"]["steps"] = []
    return data


def write_guidelines(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "guidelines.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@contextmanager
def guideline_client(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
    data: dict,
) -> Iterator[TestClient]:
    path = write_guidelines(tmp_path, data)
    app = create_app(
        settings_obj=replace(settings, guidelines_path=path),
        llm_service=fake_llm,
        whatsapp_client=fake_whatsapp,
    )
    with TestClient(app) as client:
        yield client


def start_chest(client: TestClient, difficulty: str = "Intermediate") -> dict:
    response = client.post(
        "/session/start",
        json={"scenario_id": "chest_pain", "difficulty": difficulty},
    )
    assert response.status_code == 200, response.text
    return response.json()


def token_headers(started: dict) -> dict[str, str]:
    return {"X-Session-Token": started["session_token"]}


def test_full_mode_start_hint_and_history_isolation(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    with guideline_client(settings, tmp_path, fake_llm, fake_whatsapp, full_guidelines()) as client:
        health = client.get("/health").json()
        assert health["guidelines"] == "full"

        started = start_chest(client)
        guideline = started["guideline"]
        assert guideline["name"] == "Adult Primary Care Guide (APC)"
        assert guideline["edition_year"] == "2026"
        assert guideline["section_text"] == "Acute chest pain"
        assert guideline["mode"] == "full"
        assert guideline["url"] == VALID_URL
        assert [item["title"] for item in guideline["further_reading"]] == [
            "Adult Primary Care Guide (APC)",
            "Chest pain pathway",
        ]

        hint = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "Which observations should I record?"},
            headers=token_headers(started),
        )
        assert hint.status_code == 200, hint.text
        payload = hint.json()
        assert payload["step_ids"] == ["step-3"]
        assert payload["hint"] == "Record a 12 lead ECG."
        assert payload["hints_remaining"] == 2
        assert payload["question"] == "Which observations should I record?"

        snapshot = client.get(
            f"/session/{started['session_id']}", headers=token_headers(started)
        ).json()
        assert "Record a 12 lead ECG." not in json.dumps(snapshot["transcript"])
        assert snapshot["guideline_hints_used"] == 1

        message = client.post(
            f"/session/{started['session_id']}/message",
            json={"message": "The pain started this morning."},
            headers=token_headers(started),
        )
        assert message.status_code == 200, message.text
        assert "Record a 12 lead ECG." not in json.dumps(fake_llm.patient_calls[-1])

        log_path = settings.log_dir / f"{started['session_id']}.jsonl"
        log_text = log_path.read_text(encoding="utf-8")
        assert "Which observations should I record?" in log_text
        assert "step-3" in log_text
        assert "Record a 12 lead ECG." not in log_text


def test_hint_limit_is_three_per_session(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    with guideline_client(settings, tmp_path, fake_llm, fake_whatsapp, full_guidelines()) as client:
        started = start_chest(client)
        url = f"/session/{started['session_id']}/hint"
        headers = token_headers(started)

        remaining = []
        for _ in range(3):
            response = client.post(
                url, json={"question": "vital signs"}, headers=headers
            )
            assert response.status_code == 200, response.text
            remaining.append(response.json()["hints_remaining"])
        assert remaining == [2, 1, 0]

        exhausted = client.post(
            url, json={"question": "vital signs"}, headers=headers
        )
        assert exhausted.status_code == 409

        snapshot = client.get(
            f"/session/{started['session_id']}", headers=token_headers(started)
        ).json()
        assert snapshot["guideline_hints_used"] == 3
        assert snapshot["hints_used"] == 3


def test_links_only_mode_has_no_hints(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    with guideline_client(
        settings, tmp_path, fake_llm, fake_whatsapp, links_only_guidelines()
    ) as client:
        assert client.get("/health").json()["guidelines"] == "links_only"
        response = client.post("/session/start", json={"scenario_id": "chest_pain"})
        assert response.status_code == 200, response.text
        started = response.json()
        assert started["guideline"]["mode"] == "links_only"
        assert started["guideline"]["further_reading"]

        response = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "vital signs"},
            headers=token_headers(started),
        )
        assert response.status_code == 404


def test_missing_file_disables_guidelines_but_keeps_app_working(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    app = create_app(
        settings_obj=replace(settings, guidelines_path=tmp_path / "absent.json"),
        llm_service=fake_llm,
        whatsapp_client=fake_whatsapp,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["guidelines"] == "off"
        started = start_chest(client)
        assert started["guideline"] is None
        fallback = client.post(
            f"/session/{started['session_id']}/hint",
            json={"question": "vital signs"},
            headers=token_headers(started),
        )
        assert fallback.status_code == 200
        assert fallback.json()["hint"]


def test_invalid_file_disables_guidelines(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    app = create_app(
        settings_obj=replace(settings, guidelines_path=path),
        llm_service=fake_llm,
        whatsapp_client=fake_whatsapp,
    )
    with TestClient(app) as client:
        assert client.get("/health").json()["guidelines"] == "off"
        assert client.get("/scenarios").status_code == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["documents"]["apc"].update({"url": "http://knowledgehub.health.gov.za/x"}),
        lambda data: data["documents"]["apc"].update({"url": "https://example.com/x"}),
        lambda data: data["scenarios"]["chest_pain"].update(
            {"steps": [{"id": f"s{i}", "text": "Step."} for i in range(4)]}
        ),
        lambda data: data["scenarios"]["chest_pain"].update(
            {"steps": [{"id": f"s{i}", "text": "Step."} for i in range(9)]}
        ),
        lambda data: data["scenarios"]["chest_pain"].update(
            {
                "steps": [
                    {"id": "same", "text": "First."},
                    {"id": "same", "text": "Second."},
                    {"id": "s3", "text": "Third."},
                    {"id": "s4", "text": "Fourth."},
                    {"id": "s5", "text": "Fifth."},
                ]
            }
        ),
        lambda data: data["scenarios"]["chest_pain"].update({"page_number": 12}),
    ],
)
def test_invalid_urls_step_counts_and_page_fields_are_rejected(
    tmp_path: Path, mutate
) -> None:
    data = full_guidelines()
    mutate(data)
    assert load_guidelines(write_guidelines(tmp_path, data)).enabled is False


def test_protocol_normalisation_drops_unknown_and_duplicate_ids() -> None:
    configured = [
        GuidelineStep(id="step-1", text="First."),
        GuidelineStep(id="step-2", text="Second."),
        GuidelineStep(id="step-3", text="Third."),
    ]
    raw = [
        ProtocolStepResult(id="step-1", status="partial", evidence_turn=2),
        ProtocolStepResult(id="unknown", status="met", evidence_turn=1),
        ProtocolStepResult(id="step-1", status="met", evidence_turn=3),
        ProtocolStepResult(id="step-2", status="met", evidence_turn=4),
    ]
    result = normalize_protocol_steps(raw, configured)
    assert [(step.id, step.status, step.evidence_turn) for step in result] == [
        ("step-1", "partial", 2),
        ("step-2", "met", 4),
        ("step-3", "missed", None),
    ]
    assert protocol_counts(result) == {"met": 1, "partial": 1, "missed": 1}


def test_full_mode_end_normalises_protocol_and_notes_counts(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    fake_llm.protocol_steps = [
        {"id": "step-1", "status": "met", "evidence_turn": 1},
        {"id": "unknown", "status": "met", "evidence_turn": 1},
        {"id": "step-1", "status": "missed", "evidence_turn": 2},
    ]
    with guideline_client(settings, tmp_path, fake_llm, fake_whatsapp, full_guidelines()) as client:
        started = start_chest(client)
        ended = client.post(
            f"/session/{started['session_id']}/end", headers=token_headers(started)
        )
        assert ended.status_code == 200, ended.text
        evaluation = ended.json()["evaluation"]

        steps = evaluation["protocol_steps"]
        assert len(steps) == 5
        assert steps[0] == {"id": "step-1", "status": "met", "evidence_turn": 1}
        assert [step["status"] for step in steps[1:]] == ["missed", "missed", "missed", "missed"]
        assert "Protocol review: 1 met, 0 partial, 4 missed." in evaluation["feedback_to_student"]

        context = fake_llm.evaluation_contexts[-1]
        assert [step["id"] for step in context["guideline_steps"]] == [
            "step-1",
            "step-2",
            "step-3",
            "step-4",
            "step-5",
        ]
        assert context["guideline_steps"][3]["text"] == "Give aspirin 300 mg chewed."


def test_dose_guard_removes_unsupported_doses_only(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    with guideline_client(settings, tmp_path, fake_llm, fake_whatsapp, full_guidelines()) as client:
        original_evaluate = fake_llm.evaluate

        async def unsafe_evaluate(scenario, transcript, context=None):
            evaluation = await original_evaluate(scenario, transcript, context)
            evaluation.strengths = [
                "Explained the plan clearly.",
                "Advised 500 mg of aspirin.",
            ]
            evaluation.recommended_learning = [
                "Give 500 mg aspirin.",
                "Give aspirin 300 mg chewed.",
            ]
            evaluation.feedback_to_student = (
                "You were calm. Give 500 mg aspirin now. Give aspirin 300 mg chewed."
            )
            return evaluation

        fake_llm.evaluate = unsafe_evaluate
        started = start_chest(client)
        ended = client.post(
            f"/session/{started['session_id']}/end", headers=token_headers(started)
        )
        assert ended.status_code == 200, ended.text
        evaluation = ended.json()["evaluation"]

        assert "500 mg" not in json.dumps(evaluation)
        assert "300 mg" in evaluation["feedback_to_student"]
        assert evaluation["strengths"] == ["Explained the plan clearly."]
        assert evaluation["debrief"]["what_went_well"] == ["Explained the plan clearly."]
        assert any("300 mg" in item for item in evaluation["recommended_learning"])


def test_whatsapp_hint_question_and_guideline_debrief(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    data = full_guidelines(scenario_id="abdominal_pain")
    with guideline_client(settings, tmp_path, fake_llm, fake_whatsapp, data) as client:
        manager = client.app.state.manager

        async def run() -> None:
            phone = "27825555555"
            await handle_whatsapp_message(manager, fake_whatsapp, phone, "START 1")
            assert "Guideline: Adult Primary Care Guide (APC), 2026" in fake_whatsapp.sent[0][1]

            await handle_whatsapp_message(
                manager, fake_whatsapp, phone, "HINT Which observations should I record?"
            )
            assert "Record a 12 lead ECG." in fake_whatsapp.sent[-1][1]

            await handle_whatsapp_message(manager, fake_whatsapp, phone, "END")
            debrief = fake_whatsapp.sent[-1][1]
            assert "Guideline: Adult Primary Care Guide (APC), 2026" in debrief
            assert "Acute chest pain" in debrief
            assert "Protocol steps:" in debrief

        asyncio.run(run())


def test_static_ui_uses_safe_guideline_links() -> None:
    root = Path(__file__).resolve().parents[1]
    html = (root / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert "knowledgehub.health.gov.za" in html
    assert 'link.rel = "noopener noreferrer"' in html
    assert "renderGuidelineCard" in html
