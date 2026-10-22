from __future__ import annotations

from types import SimpleNamespace

from app.config import Settings
from app.scoring import assess
from app.sessions import ScenarioRepository


def _session(*messages: str) -> SimpleNamespace:
    return SimpleNamespace(
        transcript=[
            SimpleNamespace(role="student", content=message) for message in messages
        ],
        actions=[],
        revealed_topics=set(),
        difficulty="Intermediate",
        escalation_minutes=None,
    )


def test_onset_question_does_not_count_as_sudden_severe_pain(settings: Settings) -> None:
    scenario = ScenarioRepository(settings.scenario_dir).get("abdominal_pain")
    assessment = assess(scenario, _session("When did the pain start?"))

    assert "Sudden severe unilateral lower abdominal pain" not in (
        assessment.red_flags_recognised
    )


def test_drinking_question_does_not_count_as_unable_to_keep_fluids_down(
    settings: Settings,
) -> None:
    scenario = ScenarioRepository(settings.scenario_dir).get("paediatric_dehydration")
    assessment = assess(scenario, _session("Is he drinking?"))

    assert "Unable to keep fluids down" not in assessment.red_flags_recognised
