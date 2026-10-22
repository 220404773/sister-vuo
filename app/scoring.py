from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .clinical import (
    ESCALATION_ACTIONS,
    action_options,
    escalation_target_minutes,
)
from .models import Debrief, DomainScores, Evaluation, ProtocolStepResult, TimelineEvent

HINT_PENALTY_PER_HINT = 0.3
MAX_HINT_PENALTY = 1.2

_DOSE_PATTERN = re.compile(
    r"(?<![\w.])(\d+(?:\.\d+)?)\s*"
    r"(mg|mcg|micrograms?|milligrams?|grams?|g|ml|units?|iu)\b",
    re.IGNORECASE,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = " ".join(str(value).split()).strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def _student_texts(session: Any) -> list[str]:
    return [
        str(turn.content).lower()
        for turn in getattr(session, "transcript", [])
        if getattr(turn, "role", "") == "student"
    ]


def _time_score(minutes: float | None, target: float) -> int:
    if minutes is None:
        return 0
    ratio = minutes / max(target, 1.0)
    if ratio <= 1.0:
        return 5
    if ratio <= 1.5:
        return 4
    if ratio <= 2.5:
        return 3
    if ratio <= 4.0:
        return 2
    if ratio <= 6.0:
        return 1
    return 0


def _red_flags(scenario: dict[str, Any], session: Any) -> tuple[list[str], list[str], int]:
    rubric = scenario.get("rubric", {})
    checks = rubric.get("red_flags", []) if isinstance(rubric, dict) else []
    if not isinstance(checks, list) or not checks:
        fallback = scenario.get("ground_truth", {}).get("red_flags", [])
        checks = [{"flag": str(item), "triggers": [str(item).lower()]} for item in fallback]

    texts = _student_texts(session)
    actions = {action.action_id for action in getattr(session, "actions", [])}
    recognised: list[str] = []
    missed: list[str] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        flag = str(check.get("flag", "")).strip()
        if not flag:
            continue
        triggers = [str(item).lower() for item in check.get("triggers", []) if str(item).strip()]
        action_ids = {str(item) for item in check.get("actions", []) if str(item).strip()}
        found = any(trigger in text for trigger in triggers for text in texts)
        found = found or bool(action_ids & actions)
        if found:
            recognised.append(flag)
        else:
            missed.append(flag)
    total = len(recognised) + len(missed)
    score = round(5 * len(recognised) / total) if total else 3
    return recognised, missed, score


def _history(scenario: dict[str, Any], session: Any) -> tuple[list[str], list[str], int]:
    case_file = scenario.get("case_file", {})
    rules = case_file.get("disclosure", [])
    if not isinstance(rules, list) or not rules:
        return [], [], 3

    revealed = set(getattr(session, "revealed_topics", set()))
    total_weight = 0.0
    earned_weight = 0.0
    unlocked: list[str] = []
    missed: list[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        topic = str(rule.get("topic", "")).strip()
        if not topic:
            continue
        try:
            weight = float(rule.get("weight", 1))
        except (TypeError, ValueError):
            weight = 1.0
        if not bool(rule.get("required", True)):
            continue
        total_weight += max(0.0, weight)
        label = str(rule.get("label", topic.replace("_", " ").title()))
        if topic in revealed:
            earned_weight += max(0.0, weight)
            unlocked.append(label)
        else:
            missed.append(label)
    score = round(5 * earned_weight / total_weight) if total_weight else 3
    return unlocked, missed, score


@dataclass(slots=True)
class Assessment:
    red_flags_recognised: list[str] = field(default_factory=list)
    missed_red_flags: list[str] = field(default_factory=list)
    history_unlocked: list[str] = field(default_factory=list)
    history_missed: list[str] = field(default_factory=list)
    unsafe_actions: list[str] = field(default_factory=list)
    critical_errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    time_to_escalation_minutes: float | None = None
    escalation_target_minutes: float = 10.0
    actions_taken: list[str] = field(default_factory=list)


def assess(scenario: dict[str, Any], session: Any) -> Assessment:
    assessment = Assessment()
    assessment.red_flags_recognised, assessment.missed_red_flags, red_score = _red_flags(
        scenario, session
    )
    (
        assessment.history_unlocked,
        assessment.history_missed,
        history_score,
    ) = _history(scenario, session)

    options = {option.id: option for option in action_options(scenario)}
    actions = list(getattr(session, "actions", []))
    taken = {action.action_id for action in actions}
    assessment.actions_taken = list(dict.fromkeys(action.action_id for action in actions))

    penalties = 0
    for action in actions:
        if action.unsafe:
            reason = action.warning or f"{action.label} was not indicated."
            assessment.unsafe_actions.append(f"{action.label}: {reason}")
            assessment.critical_errors.append(f"{action.label} was unsafe or not indicated.")
            penalties += 3

    case_file = scenario.get("case_file", {})
    required = [
        str(item)
        for item in case_file.get("required_actions", [])
        if str(item) not in ESCALATION_ACTIONS and str(item) in options
    ]
    for action_id in required:
        option = options[action_id]
        if not option.available:
            continue
        if action_id not in taken:
            assessment.notes.append(f"Not completed: {option.label}.")
            penalties += 1

    if "vitals" in options and "vitals" not in taken:
        assessment.notes.append("Vital signs were never recorded.")
        penalties += 1

    critical_actions = [
        str(item) for item in case_file.get("critical_actions", ["call_ambulance"])
    ]
    escalation_minutes = getattr(session, "escalation_minutes", None)
    target = escalation_target_minutes(scenario, getattr(session, "difficulty", "Intermediate"))
    assessment.time_to_escalation_minutes = escalation_minutes
    assessment.escalation_target_minutes = target

    if escalation_minutes is None:
        assessment.critical_errors.append(
            "Did not escalate to emergency services or a senior clinician."
        )
        penalties += 3
    elif escalation_minutes > target * 2:
        assessment.notes.append(
            f"Escalation was delayed ({escalation_minutes:.0f} min; target {target:.0f} min)."
        )
        penalties += 1

    critical_taken = taken & set(critical_actions)
    if critical_actions and not critical_taken and escalation_minutes is not None:
        assessment.notes.append("The expected emergency pathway was not activated.")
        penalties += 2

    safety_score = max(0, min(5, round(5 - penalties)))
    time_score = _time_score(escalation_minutes, target)

    assessment.scores = {
        "red_flag_recognition": red_score,
        "history_completeness": history_score,
        "safety": safety_score,
        "time_to_escalation": time_score,
    }
    return assessment


def _normalise_unit(unit: str) -> str:
    unit = unit.lower()
    if unit.startswith("microgram"):
        return "mcg"
    if unit.startswith("milligram"):
        return "mg"
    if unit.startswith("gram"):
        return "g"
    if unit.startswith("unit"):
        return "unit"
    return unit


def _doses(text: str) -> set[tuple[float, str]]:
    return {
        (float(number), _normalise_unit(unit))
        for number, unit in _DOSE_PATTERN.findall(text)
    }


def _step_value(step: Any, name: str) -> str:
    if isinstance(step, dict):
        return str(step.get(name, ""))
    return str(getattr(step, name, ""))


def _supported_doses(configured_steps: Any) -> set[tuple[float, str]]:
    supported: set[tuple[float, str]] = set()
    for step in configured_steps or ():
        supported |= _doses(_step_value(step, "text"))
    return supported


def _sanitise_sentence(sentence: str, supported: set[tuple[float, str]]) -> str:
    if all(dose in supported for dose in _doses(sentence)):
        return sentence
    return ""


def _sanitise_text(text: str, supported: set[tuple[float, str]]) -> str:
    sentences = [item.strip() for item in _SENTENCE_SPLIT.split(text.strip())]
    kept = [sentence for sentence in sentences if sentence and _sanitise_sentence(sentence, supported)]
    return " ".join(kept).strip()


def _sanitise_list(values: list[str], supported: set[tuple[float, str]]) -> list[str]:
    result: list[str] = []
    for value in values:
        cleaned = _sanitise_text(str(value), supported)
        if cleaned:
            result.append(cleaned)
    return result


def apply_dose_guard(evaluation: Evaluation, configured_steps: Any) -> Evaluation:
    """Remove feedback sentences that name a dose not supplied in the protocol."""
    supported = _supported_doses(configured_steps)
    evaluation.strengths = _sanitise_list(evaluation.strengths, supported)
    evaluation.recommended_learning = _sanitise_list(
        evaluation.recommended_learning, supported
    )
    evaluation.feedback_to_student = _sanitise_text(
        evaluation.feedback_to_student, supported
    )
    if not evaluation.feedback_to_student:
        evaluation.feedback_to_student = (
            "Review the debrief and discuss the safe next steps with your supervisor."
        )
    evaluation.debrief.what_went_well = list(evaluation.strengths)
    return evaluation


def normalize_protocol_steps(
    raw_steps: list[Any] | None, configured_steps: Any
) -> list[ProtocolStepResult]:
    configured = list(configured_steps or ())
    configured_ids = {_step_value(step, "id") for step in configured}
    supplied: dict[str, ProtocolStepResult] = {}
    for item in raw_steps or ():
        if isinstance(item, ProtocolStepResult):
            step_id = item.id
            status = item.status
            evidence_turn = item.evidence_turn
        elif isinstance(item, dict):
            step_id = str(item.get("id", ""))
            status = str(item.get("status", ""))
            evidence_turn = item.get("evidence_turn")
        else:
            continue
        if step_id in supplied:
            continue
        if step_id not in configured_ids:
            continue
        if status not in {"met", "partial", "missed"}:
            continue
        if (
            not isinstance(evidence_turn, int)
            or isinstance(evidence_turn, bool)
            or evidence_turn < 1
        ):
            evidence_turn = None
        supplied[step_id] = ProtocolStepResult(
            id=step_id,
            status=status,  # type: ignore[arg-type]
            evidence_turn=evidence_turn,
        )

    return [
        supplied.get(
            _step_value(step, "id"),
            ProtocolStepResult(
                id=_step_value(step, "id"),
                status="missed",
                evidence_turn=None,
            ),
        )
        for step in configured
    ]


def protocol_counts(steps: list[ProtocolStepResult]) -> dict[str, int]:
    counts = {"met": 0, "partial": 0, "missed": 0}
    for step in steps:
        counts[step.status] += 1
    return counts


def merge_evaluation(
    scenario: dict[str, Any],
    session: Any,
    llm_evaluation: Evaluation,
    assessment: Assessment,
) -> Evaluation:
    llm_scores = llm_evaluation.domain_scores
    final_scores = {
        "red_flag_recognition": assessment.scores.get("red_flag_recognition", 0),
        "history_completeness": assessment.scores.get("history_completeness", 0),
        "prioritisation": llm_scores.prioritisation,
        "safety": assessment.scores.get("safety", 0),
        "communication_empathy": llm_scores.communication_empathy,
        "time_to_escalation": assessment.scores.get("time_to_escalation", 0),
    }
    hint_penalty = min(
        MAX_HINT_PENALTY,
        round(getattr(session, "hints_used", 0) * HINT_PENALTY_PER_HINT, 2),
    )
    overall = round(max(0.0, min(5.0, sum(final_scores.values()) / 6 - hint_penalty)), 1)

    guideline = getattr(session, "guideline", None)
    guideline_full = guideline is not None and guideline.mode == "full"
    protocol_steps = (
        normalize_protocol_steps(llm_evaluation.protocol_steps, guideline.steps)
        if guideline_full
        else None
    )
    feedback = llm_evaluation.feedback_to_student.strip()
    if protocol_steps is not None:
        counts = protocol_counts(protocol_steps)
        feedback += (
            f" Protocol review: {counts['met']} met, {counts['partial']} partial, "
            f"{counts['missed']} missed."
        )
    hints_used = getattr(session, "hints_used", 0)
    if hints_used:
        noun = "hint" if hints_used == 1 else "hints"
        feedback += f" You used {hints_used} supervisor {noun}."

    missed = _dedupe(assessment.missed_red_flags + llm_evaluation.missed_red_flags)
    unsafe = _dedupe(assessment.unsafe_actions + llm_evaluation.unsafe_actions)
    critical = _dedupe(assessment.critical_errors + llm_evaluation.critical_errors)
    what_was_missed = _dedupe(
        assessment.missed_red_flags
        + assessment.notes
        + [
            f"History not explored: {item}."
            for item in assessment.history_missed
        ]
    )

    debrief_data = scenario.get("debrief", {})
    if not isinstance(debrief_data, dict):
        debrief_data = {}
    timeline = [
        event if isinstance(event, TimelineEvent) else TimelineEvent(**event)
        for event in getattr(session, "timeline", [])
    ]
    debrief = Debrief(
        what_went_well=_dedupe(llm_evaluation.strengths)
        or ["You completed the simulation and can review the decisions below."],
        what_was_missed=what_was_missed,
        critical_errors=critical,
        ideal_approach=[
            str(item) for item in debrief_data.get("ideal_approach", []) if str(item).strip()
        ],
        model_answer=str(debrief_data.get("model_answer", "")).strip(),
        reflection_questions=[
            str(item)
            for item in debrief_data.get("reflection_questions", [])
            if str(item).strip()
        ],
        guideline_references=[
            str(item) for item in debrief_data.get("references", []) if str(item).strip()
        ],
        timeline=timeline,
        time_to_escalation_minutes=assessment.time_to_escalation_minutes,
        escalation_target_minutes=assessment.escalation_target_minutes,
        actions_taken=assessment.actions_taken,
    )

    evaluation = Evaluation(
        overall_score=overall,
        domain_scores=DomainScores(**final_scores),
        strengths=llm_evaluation.strengths,
        missed_red_flags=missed,
        unsafe_actions=unsafe,
        critical_errors=critical,
        recommended_learning=llm_evaluation.recommended_learning,
        feedback_to_student=feedback,
        hint_penalty=hint_penalty,
        debrief=debrief,
        protocol_steps=protocol_steps,
    )
    if guideline_full:
        evaluation = apply_dose_guard(evaluation, guideline.steps)
    return evaluation
