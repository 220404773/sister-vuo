from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .models import ActionOption, Difficulty, MonitorView, VitalsView

DIFFICULTY_PROFILES: dict[str, dict[str, Any]] = {
    "Beginner": {"time_scale": 0.6, "target_multiplier": 1.5, "hints": "auto"},
    "Intermediate": {"time_scale": 1.0, "target_multiplier": 1.0, "hints": "button"},
    "Advanced": {"time_scale": 1.4, "target_multiplier": 0.75, "hints": "none"},
}

ESCALATION_ACTIONS = {"call_ambulance", "escalate_doctor"}

CORE_ACTIONS: dict[str, dict[str, Any]] = {
    "vitals": {
        "label": "Take vitals",
        "description": "Record heart rate, blood pressure, oxygen saturation, respiratory rate, temperature and pain.",
        "minutes": 3.0,
        "category": "assessment",
        "repeatable": True,
    },
    "ecg": {
        "label": "12-lead ECG",
        "description": "Attach the monitor and record a twelve-lead ECG.",
        "minutes": 5.0,
        "category": "investigation",
        "repeatable": False,
    },
    "oxygen": {
        "label": "Give oxygen",
        "description": "Start oxygen by nasal prongs or mask according to the saturation.",
        "minutes": 2.0,
        "category": "treatment",
        "repeatable": False,
    },
    "aspirin": {
        "label": "Aspirin 300 mg",
        "description": "Give aspirin 300 mg chewed, if indicated and not contraindicated.",
        "minutes": 1.0,
        "category": "treatment",
        "repeatable": False,
    },
    "iv_access": {
        "label": "IV access",
        "description": "Site an intravenous cannula, take bloods and start fluids if indicated.",
        "minutes": 4.0,
        "category": "treatment",
        "repeatable": False,
    },
    "call_ambulance": {
        "label": "Call ambulance",
        "description": "Activate the emergency service for urgent transfer.",
        "minutes": 1.0,
        "category": "escalation",
        "repeatable": False,
    },
    "escalate_doctor": {
        "label": "Escalate to doctor",
        "description": "Hand over to the doctor or senior clinician using SBAR.",
        "minutes": 1.0,
        "category": "escalation",
        "repeatable": False,
    },
}

DEFAULT_VITALS = {
    "heart_rate": 80,
    "systolic_bp": 120,
    "diastolic_bp": 78,
    "spo2": 97,
    "respiratory_rate": 16,
    "pain_score": 0,
    "temperature": 36.8,
}

VITAL_LIMITS = {
    "heart_rate": (25, 230),
    "systolic_bp": (45, 240),
    "diastolic_bp": (25, 140),
    "spo2": (50, 100),
    "respiratory_rate": (3, 70),
    "pain_score": (0, 10),
    "temperature": (32.0, 43.0),
}


class ClinicalActionError(RuntimeError):
    pass


class UnknownAction(ClinicalActionError, LookupError):
    pass


class ActionUnavailable(ClinicalActionError):
    pass


class ActionAlreadyDone(ClinicalActionError):
    pass


@dataclass(slots=True)
class ActionRecord:
    action_id: str
    label: str
    clinical_minutes: float
    result: str
    warning: str | None = None
    unsafe: bool = False
    effects: dict[str, float] = field(default_factory=dict)
    deterioration_factor: float = 1.0
    repeatable: bool = False


def normalize_difficulty(value: str | None, fallback: str = "Intermediate") -> Difficulty:
    candidate = (value or fallback).strip().lower()
    mapping = {
        "beginner": "Beginner",
        "intermediate": "Intermediate",
        "advanced": "Advanced",
    }
    return mapping.get(candidate, "Intermediate")  # type: ignore[return-value]


def difficulty_profile(difficulty: str) -> dict[str, Any]:
    return DIFFICULTY_PROFILES.get(difficulty, DIFFICULTY_PROFILES["Intermediate"])


def hints_enabled(difficulty: str) -> bool:
    return difficulty_profile(difficulty)["hints"] != "none"


def auto_hints_enabled(difficulty: str) -> bool:
    return difficulty_profile(difficulty)["hints"] == "auto"


def _scenario_actions(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    value = scenario.get("clinical_actions", {})
    return value if isinstance(value, dict) else {}


def action_options(scenario: dict[str, Any]) -> list[ActionOption]:
    overrides = _scenario_actions(scenario)
    options: list[ActionOption] = []
    for action_id, default in CORE_ACTIONS.items():
        override = overrides.get(action_id, {})
        spec = {**default, **override}
        options.append(
            ActionOption(
                id=action_id,
                label=str(spec["label"]),
                description=str(spec["description"]),
                minutes=float(spec["minutes"]),
                category=str(spec["category"]),
                available=bool(spec.get("available", True)),
                unavailable_reason=spec.get("unavailable_reason"),
                repeatable=bool(spec.get("repeatable", False)),
            )
        )
    for action_id, override in overrides.items():
        if action_id in CORE_ACTIONS:
            continue
        options.append(
            ActionOption(
                id=action_id,
                label=str(override.get("label", action_id.replace("_", " ").title())),
                description=str(override.get("description", "Perform this clinical action.")),
                minutes=float(override.get("minutes", 2.0)),
                category=str(override.get("category", "investigation")),
                available=bool(override.get("available", True)),
                unavailable_reason=override.get("unavailable_reason"),
                repeatable=bool(override.get("repeatable", False)),
            )
        )
    return options


def action_option(scenario: dict[str, Any], action_id: str) -> ActionOption:
    for option in action_options(scenario):
        if option.id == action_id:
            return option
    raise UnknownAction(f"Unknown clinical action '{action_id}'.")


def baseline_vitals(scenario: dict[str, Any]) -> dict[str, float]:
    case_file = scenario.get("case_file", {})
    raw = case_file.get("baseline_vitals", {})
    values = dict(DEFAULT_VITALS)
    if isinstance(raw, dict):
        for key in DEFAULT_VITALS:
            if key in raw:
                try:
                    values[key] = float(raw[key])
                except (TypeError, ValueError):
                    continue
    return values


def escalation_target_minutes(scenario: dict[str, Any], difficulty: str) -> float:
    case_file = scenario.get("case_file", {})
    raw = case_file.get("escalation_target_minutes", 10)
    try:
        target = float(raw)
    except (TypeError, ValueError):
        target = 10.0
    return max(1.0, target * float(difficulty_profile(difficulty)["target_multiplier"]))


def _clamp(metric: str, value: float) -> float:
    low, high = VITAL_LIMITS[metric]
    return max(low, min(high, value))


def compute_monitor(
    scenario: dict[str, Any],
    clinical_minutes: float,
    wall_minutes: float,
    actions: list[ActionRecord],
    difficulty: str,
    escalation_minutes: float | None,
    *,
    target_minutes: float | None = None,
) -> MonitorView:
    profile = difficulty_profile(difficulty)
    effective_minutes = max(clinical_minutes, wall_minutes * float(profile["time_scale"]))
    baseline = baseline_vitals(scenario)
    case_file = scenario.get("case_file", {})
    trajectory = case_file.get("vitals_deterioration", {})
    if not isinstance(trajectory, dict):
        trajectory = {}

    per_ten = trajectory.get("per_10_minutes", {})
    if not isinstance(per_ten, dict):
        per_ten = {}
    accelerate_after = float(trajectory.get("accelerate_after_minutes", 10000) or 10000)
    acceleration = float(trajectory.get("acceleration_multiplier", 1.0) or 1.0)

    if effective_minutes <= accelerate_after:
        units = effective_minutes / 10.0
    else:
        units = (accelerate_after / 10.0) + (
            (effective_minutes - accelerate_after) / 10.0
        ) * acceleration

    deterioration_factor = 1.0
    for action in actions:
        deterioration_factor *= max(0.0, action.deterioration_factor)

    values: dict[str, float] = {}
    for metric, base in baseline.items():
        try:
            rate = float(per_ten.get(metric, 0))
        except (TypeError, ValueError):
            rate = 0.0
        values[metric] = base + rate * units * deterioration_factor

    for action in actions:
        for metric, delta in action.effects.items():
            if metric in values:
                values[metric] += delta

    for metric, value in values.items():
        values[metric] = _clamp(metric, value)

    target = (
        target_minutes
        if target_minutes is not None
        else escalation_target_minutes(scenario, difficulty)
    )
    alerts: list[str] = []
    if values["spo2"] < 90:
        alerts.append("Severe hypoxia: give oxygen and escalate now")
    elif values["spo2"] < 94:
        alerts.append("Hypoxia: oxygen saturation below 94%")
    if values["systolic_bp"] < 90:
        alerts.append("Hypotension: systolic BP below 90 mmHg")
    if values["heart_rate"] >= 130:
        alerts.append("Marked tachycardia")
    if values["heart_rate"] <= 50:
        alerts.append("Bradycardia")
    if values["respiratory_rate"] >= 30:
        alerts.append("Marked tachypnoea")
    if values["temperature"] >= 38.0:
        alerts.append("Fever")
    if escalation_minutes is None and clinical_minutes >= target:
        alerts.append("Escalation is overdue: call for help now")

    critical = (
        values["spo2"] < 90
        or values["systolic_bp"] < 90
        or values["heart_rate"] >= 140
        or values["heart_rate"] <= 45
        or values["respiratory_rate"] >= 35
    )
    if critical:
        status = "critical"
    elif effective_minutes >= 5 or alerts:
        status = "deteriorating"
    else:
        status = "stable"

    vitals = VitalsView(
        heart_rate=round(values["heart_rate"]),
        systolic_bp=round(values["systolic_bp"]),
        diastolic_bp=round(values["diastolic_bp"]),
        spo2=round(values["spo2"]),
        respiratory_rate=round(values["respiratory_rate"]),
        pain_score=round(values["pain_score"]),
        temperature=round(values["temperature"], 1),
    )
    return MonitorView(
        vitals=vitals,
        clinical_minutes=round(clinical_minutes, 1),
        monitor_minutes=round(effective_minutes, 1),
        status=status,  # type: ignore[arg-type]
        alerts=alerts,
        escalated=escalation_minutes is not None,
        escalation_minutes=escalation_minutes,
        escalation_target_minutes=round(target, 1),
    )


def perform_action(
    scenario: dict[str, Any],
    option: ActionOption,
    monitor_before: MonitorView,
) -> ActionRecord:
    overrides = _scenario_actions(scenario)
    override = overrides.get(option.id, {})
    if not isinstance(override, dict):
        override = {}

    effects: dict[str, float] = {}
    raw_effects = override.get("effects", {})
    if isinstance(raw_effects, dict):
        for metric, value in raw_effects.items():
            try:
                effects[str(metric)] = float(value)
            except (TypeError, ValueError):
                continue

    if option.id == "oxygen" and not effects:
        sat = monitor_before.vitals.spo2
        if sat < 94:
            effects["spo2"] = 3.0

    deterioration_factor = 1.0
    try:
        deterioration_factor = float(override.get("deterioration_factor", 1.0))
    except (TypeError, ValueError):
        deterioration_factor = 1.0
    deterioration_factor = max(0.0, min(1.5, deterioration_factor))

    result = str(override.get("result", "")).strip()
    if not result and option.id != "vitals":
        result = f"{option.label} completed."

    warning = override.get("warning")
    warning = str(warning).strip() if warning else None
    unsafe = bool(override.get("unsafe", False))
    if unsafe and not warning:
        warning = "This action is not indicated for this patient."

    return ActionRecord(
        action_id=option.id,
        label=option.label,
        clinical_minutes=0.0,
        result=result,
        warning=warning,
        unsafe=unsafe,
        effects=effects,
        deterioration_factor=deterioration_factor,
        repeatable=option.repeatable,
    )


def vitals_result(monitor: MonitorView) -> str:
    v = monitor.vitals
    return (
        "Observations recorded: "
        f"HR {v.heart_rate} bpm, BP {v.systolic_bp}/{v.diastolic_bp} mmHg, "
        f"SpO2 {v.spo2}%, RR {v.respiratory_rate}/min, pain {v.pain_score}/10, "
        f"temperature {v.temperature:.1f} C."
    )


def monitor_text(monitor: MonitorView) -> str:
    v = monitor.vitals
    lines = [
        (
            f"HR {v.heart_rate} bpm | BP {v.systolic_bp}/{v.diastolic_bp} mmHg | "
            f"SpO2 {v.spo2}% | RR {v.respiratory_rate}/min | pain {v.pain_score}/10"
        ),
        f"Clinical time {monitor.clinical_minutes:.0f} min | status {monitor.status}",
    ]
    if monitor.alerts:
        lines.append("Alerts: " + "; ".join(monitor.alerts))
    return "\n".join(lines)


def action_command_map() -> dict[str, str]:
    return {
        "VITALS": "vitals",
        "OBS": "vitals",
        "ECG": "ecg",
        "OXYGEN": "oxygen",
        "ASPIRIN": "aspirin",
        "IV": "iv_access",
        "IV ACCESS": "iv_access",
        "AMBULANCE": "call_ambulance",
        "CALL AMBULANCE": "call_ambulance",
        "DOCTOR": "escalate_doctor",
        "ESCALATE": "escalate_doctor",
        "PREGNANCY TEST": "pregnancy_test",
    }
