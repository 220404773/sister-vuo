from __future__ import annotations

import re
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


def patient_age_years(scenario: dict[str, Any]) -> float | None:
    patient = scenario.get("patient", {})
    try:
        age = float(patient.get("age"))
    except (TypeError, ValueError):
        return None
    unit = str(patient.get("age_unit", "years")).lower()
    if unit.startswith("month"):
        return age / 12
    if unit.startswith("week"):
        return age / 52
    if unit.startswith("day"):
        return age / 365
    return age


def age_group(scenario: dict[str, Any]) -> str:
    """'paediatric' or 'adult': the scenario's own setting, or from the patient's age."""
    explicit = str(scenario.get("age_group", "")).lower()
    if explicit in ("adult", "paediatric"):
        return explicit
    years = patient_age_years(scenario)
    return "paediatric" if years is not None and years < 13 else "adult"


def vital_band(scenario: dict[str, Any]) -> dict[str, float]:
    """Alarm limits by age (approximate, after APLS-style paediatric ranges)."""
    years = patient_age_years(scenario)
    if years is None or years >= 12:
        return {"hr_high": 130, "hr_low": 50, "rr_high": 30, "sbp_low": 90}
    if years < 1:
        return {"hr_high": 180, "hr_low": 100, "rr_high": 60, "sbp_low": 65}
    if years < 5:
        return {"hr_high": 160, "hr_low": 80, "rr_high": 40, "sbp_low": 70 + 2 * years}
    return {"hr_high": 140, "hr_low": 60, "rr_high": 30, "sbp_low": 70 + 2 * years}


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
    # Alarms state readings only, like a real monitor; they never tell the student what to do.
    # Limits follow the patient's age, so a normal infant heart rate does not alarm.
    band = vital_band(scenario)
    alerts: list[str] = []
    if values["spo2"] < 90:
        alerts.append("Severe hypoxia: oxygen saturation below 90%")
    elif values["spo2"] < 94:
        alerts.append("Hypoxia: oxygen saturation below 94%")
    if values["systolic_bp"] < band["sbp_low"]:
        alerts.append(f"Hypotension: systolic BP below {band['sbp_low']:.0f} mmHg")
    if values["heart_rate"] >= band["hr_high"]:
        alerts.append("Marked tachycardia")
    if values["heart_rate"] <= band["hr_low"]:
        alerts.append("Bradycardia")
    if values["respiratory_rate"] >= band["rr_high"]:
        alerts.append("Marked tachypnoea")
    if values["temperature"] >= 38.0:
        alerts.append("Fever")
    if (
        escalation_minutes is None
        and clinical_minutes >= target
        and normalize_difficulty(difficulty) == "Beginner"
    ):
        # A coaching prompt, so only the guided level shows it.
        alerts.append("Escalation is overdue: call for help now")

    critical = (
        values["spo2"] < 90
        or values["systolic_bp"] < band["sbp_low"]
        or values["heart_rate"] >= band["hr_high"] + 10
        or values["heart_rate"] <= band["hr_low"] - 5
        or values["respiratory_rate"] >= band["rr_high"] + 5
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


# Words that identify a typed action. A scenario can add its own under clinical_actions.<id>.keywords.
ACTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "vitals": ("observation", "vital", "obs", "blood pressure", "pulse", "heart rate",
               "oxygen saturation", "saturation", "oximet", "respiratory rate", "temperature",
               "capillary refill", "weigh", "measure", "assess"),
    "ecg": ("ecg", "electrocardiogram", "12-lead", "12 lead", "twelve-lead", "twelve lead",
            "rhythm strip", "cardiac monitor"),
    "oxygen": ("oxygen", "o2", "nasal prong", "non-rebreather", "rebreather", "face mask", "high flow",
               "high-flow"),
    "aspirin": ("aspirin", "acetylsalicylic", "disprin", "ecotrin"),
    "iv_access": ("cannula", "iv access", "intravenous access", "drip", "iv line", "intravenous line",
                  "venous access", "intraosseous", "bloods", "blood test", "take blood", "line in"),
    "call_ambulance": ("ambulance", "ems", "emergency service", "10177", "transfer", "refer to hospital",
                       "refer urgently", "urgent referral"),
    "escalate_doctor": ("doctor", "medical officer", "physician", "paediatrician", "pediatrician",
                        "registrar", "senior", "sbar", "escalate", "call for help", "team leader"),
    "pregnancy_test": ("pregnancy test", "hcg", "pregnant test", "urine pregnancy"),
}
_NEGATION = re.compile(r"\b(do not|don't|dont|avoid|withhold|without|no need|not give|never|hold off)\b")


def _keyword_hits(text: str, keywords: list[str]) -> int:
    hits = 0
    for keyword in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(keyword)
        if len(keyword) <= 3:
            pattern += r"(?![a-z0-9])"
        if re.search(pattern, text):
            hits += 1
    return hits


def match_action(scenario: dict[str, Any], text: str) -> str | None:
    """The action a typed sentence describes, or None. One action per sentence; a negated
    sentence ("do not give aspirin") is never treated as doing it."""
    lowered = " ".join(text.lower().split())
    if not lowered or _NEGATION.search(lowered):
        return None
    overrides = _scenario_actions(scenario)
    # Scenario-specific actions first, so "salbutamol by oxygen-driven nebuliser" is the
    # nebuliser, not plain oxygen, when the scores tie.
    ordered = [item for item in overrides if item not in CORE_ACTIONS] + list(CORE_ACTIONS)
    best, best_hits = None, 0
    for action_id in ordered:
        override = overrides.get(action_id, {})
        keywords = list(ACTION_KEYWORDS.get(action_id, ()))
        if isinstance(override, dict):
            keywords += [str(word).lower() for word in override.get("keywords", [])]
        hits = _keyword_hits(lowered, keywords)
        if hits > best_hits:
            best, best_hits = action_id, hits
    return best


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
