from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .clinical import (
    ESCALATION_ACTIONS,
    ActionAlreadyDone,
    ActionRecord,
    ActionUnavailable,
    action_option,
    auto_hints_enabled,
    compute_monitor,
    hints_enabled,
    normalize_difficulty,
    perform_action,
    vitals_result,
)
from .config import Settings
from .guidelines import (
    MAX_GUIDELINE_HINTS,
    GuidelineHintsUnavailable,
    GuidelineRepository,
    ScenarioGuideline,
)
from .models import (
    Evaluation,
    MonitorView,
    ScenarioSummary,
    SessionSnapshot,
    TimelineEvent,
    TranscriptTurn,
)
from .scoring import (
    HINT_PENALTY_PER_HINT,
    MAX_HINT_PENALTY,
    assess,
    merge_evaluation,
)

logger = logging.getLogger(__name__)

QUESTION_MINUTES = 1.0


class ScenarioError(RuntimeError):
    pass


class ScenarioNotFound(LookupError):
    pass


class SessionNotFound(LookupError):
    pass


class SessionCompleted(RuntimeError):
    pass


class TurnLimitReached(SessionCompleted):
    pass


class ActiveSessionExists(RuntimeError):
    pass


class SessionCapacityReached(RuntimeError):
    pass


class HintsDisabled(RuntimeError):
    pass


class NoHintsRemaining(RuntimeError):
    pass


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def mask_phone(phone: str | None) -> str | None:
    if not phone:
        return None
    digits = "".join(character for character in phone if character.isdigit())
    if len(digits) < 4:
        return "***"
    return f"***{digits[-4:]}"


def _require(mapping: dict[str, Any], key: str, file: Path) -> Any:
    if key not in mapping:
        raise ScenarioError(f"{file.name} is missing required key '{key}'")
    return mapping[key]


class ScenarioRepository:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise ScenarioError(f"Scenario directory not found: {self.directory}")

        scenarios: dict[str, dict[str, Any]] = {}
        for path in self.directory.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ScenarioError(f"Could not load scenario {path.name}: {exc}") from exc
            self._validate(data, path)
            scenario_id = str(data["id"])
            if scenario_id in scenarios:
                raise ScenarioError(f"Duplicate scenario id '{scenario_id}'")
            scenarios[scenario_id] = data

        if not scenarios:
            raise ScenarioError(f"No scenario JSON files found in {self.directory}")

        self._scenarios = scenarios
        self._ordered = sorted(
            scenarios.values(),
            key=lambda item: (int(item.get("order", 999)), str(item["title"])),
        )

    @staticmethod
    def _validate(data: Any, path: Path) -> None:
        if not isinstance(data, dict):
            raise ScenarioError(f"{path.name} must contain a JSON object")
        for key in (
            "id",
            "title",
            "setting",
            "difficulty",
            "description",
            "speaker",
            "patient",
            "presenting_complaint",
            "opening_line",
            "personality",
            "case_file",
            "ground_truth",
            "clinical_actions",
            "rubric",
            "hints",
            "debrief",
            "learning_objectives",
        ):
            _require(data, key, path)
        for key in (
            "speaker",
            "patient",
            "case_file",
            "ground_truth",
            "clinical_actions",
            "rubric",
            "debrief",
        ):
            if not isinstance(data[key], dict):
                raise ScenarioError(f"{path.name} key '{key}' must be an object")
        for key in ("title", "setting", "difficulty", "description", "opening_line"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ScenarioError(f"{path.name} key '{key}' must be a non-empty string")
        if not isinstance(data["learning_objectives"], list):
            raise ScenarioError(f"{path.name} key 'learning_objectives' must be a list")
        if not isinstance(data["hints"], list) or not all(
            isinstance(item, str) and item.strip() for item in data["hints"]
        ):
            raise ScenarioError(f"{path.name} key 'hints' must be a list of non-empty strings")

        for key in ("name", "role", "language", "style"):
            _require(data["speaker"], key, path)
        for key in ("name", "age", "sex"):
            _require(data["patient"], key, path)

        case_file = data["case_file"]
        for key in (
            "history",
            "risk_factors",
            "baseline_vitals",
            "vitals_deterioration",
            "escalation_target_minutes",
            "disclosure",
        ):
            _require(case_file, key, path)
        if not isinstance(case_file["history"], dict):
            raise ScenarioError(f"{path.name} case_file.history must be an object")
        if not isinstance(case_file["baseline_vitals"], dict):
            raise ScenarioError(f"{path.name} case_file.baseline_vitals must be an object")
        if not isinstance(case_file["vitals_deterioration"], dict):
            raise ScenarioError(f"{path.name} case_file.vitals_deterioration must be an object")
        if not isinstance(case_file["disclosure"], list) or not case_file["disclosure"]:
            raise ScenarioError(f"{path.name} case_file.disclosure must be a non-empty list")
        for rule in case_file["disclosure"]:
            if not isinstance(rule, dict) or not rule.get("topic"):
                raise ScenarioError(f"{path.name} has an invalid disclosure rule")
            if not isinstance(rule.get("triggers"), list) or not rule["triggers"]:
                raise ScenarioError(
                    f"{path.name} disclosure rule '{rule.get('topic')}' needs triggers"
                )

        ground_truth = data["ground_truth"]
        for key in ("diagnosis", "red_flags", "expected_actions", "forbidden_patient_terms"):
            _require(ground_truth, key, path)
        if not isinstance(ground_truth["forbidden_patient_terms"], list):
            raise ScenarioError(f"{path.name} forbidden_patient_terms must be a list")

        rubric = data["rubric"]
        if not isinstance(rubric.get("red_flags"), list) or not rubric["red_flags"]:
            raise ScenarioError(f"{path.name} rubric.red_flags must be a non-empty list")

        debrief = data["debrief"]
        for key in ("ideal_approach", "model_answer", "reflection_questions", "references"):
            _require(debrief, key, path)
        if not isinstance(debrief["ideal_approach"], list):
            raise ScenarioError(f"{path.name} debrief.ideal_approach must be a list")
        if not isinstance(debrief["reflection_questions"], list):
            raise ScenarioError(f"{path.name} debrief.reflection_questions must be a list")
        if not isinstance(debrief["references"], list):
            raise ScenarioError(f"{path.name} debrief.references must be a list")

    def get(self, scenario_id: str) -> dict[str, Any]:
        scenario = self._scenarios.get(scenario_id)
        if scenario is None:
            raise ScenarioNotFound(f"Unknown scenario '{scenario_id}'.")
        return scenario

    def summary(self, scenario_id: str) -> ScenarioSummary:
        return self._summary(self.get(scenario_id))

    def list_summaries(self) -> list[ScenarioSummary]:
        return [self._summary(scenario) for scenario in self._ordered]

    def resolve(self, value: str) -> str | None:
        candidate = value.strip()
        if not candidate:
            return None
        if candidate in self._scenarios:
            return candidate
        if candidate.isdigit():
            index = int(candidate) - 1
            if 0 <= index < len(self._ordered):
                return str(self._ordered[index]["id"])
        lowered = candidate.lower()
        for scenario in self._ordered:
            scenario_id = str(scenario["id"])
            if lowered in {scenario_id.lower(), str(scenario["title"]).lower()}:
                return scenario_id
            if lowered in str(scenario["title"]).lower():
                return scenario_id
        return None

    @staticmethod
    def _summary(scenario: dict[str, Any]) -> ScenarioSummary:
        patient = scenario["patient"]
        unit = patient.get("age_unit", "years")
        label = f"{patient.get('name')}, {patient.get('age')} {unit}"
        return ScenarioSummary(
            id=str(scenario["id"]),
            title=str(scenario["title"]),
            setting=str(scenario["setting"]),
            difficulty=str(scenario["difficulty"]),
            description=str(scenario["description"]),
            patient_label=label,
            learning_objectives=[str(item) for item in scenario["learning_objectives"]],
        )


class TranscriptLogger:
    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def log(self, session_id: str, event: str, **fields: Any) -> None:
        record = {"timestamp": utcnow(), "event": event, **fields}
        path = self.directory / f"{session_id}.jsonl"
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, default=str))
                handle.write("\n")
        except OSError:
            logger.exception("Could not write transcript log for session %s", session_id)

    def delete_expired(self, retention_days: int) -> int:
        """Delete transcript files older than the retention period.

        A retention of 0 (or less) means keep transcripts forever.
        """
        if retention_days <= 0:
            return 0
        cutoff = time.time() - retention_days * 24 * 60 * 60
        deleted = 0
        for path in self.directory.glob("*.jsonl"):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    deleted += 1
            except OSError:
                logger.exception("Could not delete expired transcript %s", path)
        if deleted:
            logger.info(
                "Deleted %d transcript file(s) older than %d day(s)",
                deleted,
                retention_days,
            )
        return deleted


@dataclass(slots=True)
class Session:
    id: str
    scenario: dict[str, Any]
    channel: str
    session_token: str | None = None
    phone: str | None = None
    student_name: str | None = None
    difficulty: str = "Intermediate"
    transcript: list[TranscriptTurn] = field(default_factory=list)
    student_turns: int = 0
    status: str = "active"
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
    created_at: str = field(default_factory=utcnow)
    last_activity: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    revealed_topics: set[str] = field(default_factory=set)
    clinical_minutes: float = 0.0
    started_monotonic: float = field(default_factory=time.monotonic)
    actions: list[ActionRecord] = field(default_factory=list)
    hints_used: int = 0
    hint_index: int = 0
    guideline: ScenarioGuideline | None = None
    guideline_hints_used: int = 0
    auto_hints_shown: set[int] = field(default_factory=set)
    escalation_minutes: float | None = None
    escalation_action: str | None = None
    timeline: list[TimelineEvent] = field(default_factory=list)


@dataclass(slots=True)
class TurnResult:
    session_id: str
    reply: str
    student_turns: int
    turns_remaining: int
    status: str
    monitor: MonitorView
    auto_evaluated: bool = False
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
    hint: str | None = None


@dataclass(slots=True)
class ActionResult:
    session_id: str
    action_id: str
    action_label: str
    result: str
    warning: str | None
    monitor: MonitorView
    status: str
    student_turns: int
    turns_remaining: int
    actions_taken: list[str]


@dataclass(slots=True)
class HintResult:
    session_id: str
    hint: str
    hints_used: int
    hint_penalty: float
    hints_remaining: int
    monitor: MonitorView
    question: str = ""
    step_ids: list[str] = field(default_factory=list)


def hint_penalty(hints_used: int) -> float:
    return min(MAX_HINT_PENALTY, round(hints_used * HINT_PENALTY_PER_HINT, 2))


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        scenarios: ScenarioRepository,
        llm: Any,
        transcripts: TranscriptLogger,
        guidelines: GuidelineRepository | None = None,
    ) -> None:
        self.settings = settings
        self.scenarios = scenarios
        self.llm = llm
        self.transcripts = transcripts
        self.guidelines = guidelines or GuidelineRepository.disabled()
        self.max_turns = settings.max_student_turns
        self._sessions: dict[str, Session] = {}
        self._phone_sessions: dict[str, str] = {}
        self._start_lock = asyncio.Lock()

    def active_count(self) -> int:
        return sum(1 for session in self._sessions.values() if session.status == "active")

    def get(self, session_id: str) -> Session:
        self._purge_expired()
        session = self._sessions.get(session_id)
        if session is None:
            raise SessionNotFound("Session not found.")
        session.last_activity = time.monotonic()
        return session

    def authorize(self, session_id: str, token: str | None) -> Session:
        """Look up a session and verify its per-session token in constant time."""
        self._purge_expired()
        session = self._sessions.get(session_id)
        expected = session.session_token if session is not None else None
        supplied = token or ""
        if (
            session is None
            or not expected
            or not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))
        ):
            raise SessionNotFound("Session not found.")
        session.last_activity = time.monotonic()
        return session

    def _purge_expired(self) -> None:
        """Drop sessions idle for longer than the TTL, so memory stays bounded."""
        cutoff = time.monotonic() - self.settings.session_ttl_minutes * 60
        expired = [
            session
            for session in self._sessions.values()
            if session.last_activity < cutoff and not session.lock.locked()
        ]
        for session in expired:
            self._sessions.pop(session.id, None)
            self._unbind_phone(session)
            self.transcripts.log(session.id, "session_expired")

    def get_active_by_phone(self, phone: str) -> Session | None:
        self._purge_expired()
        session_id = self._phone_sessions.get(phone)
        if session_id is None:
            return None
        session = self._sessions.get(session_id)
        if session is None:
            self._phone_sessions.pop(phone, None)
            return None
        if session.status != "active":
            self._phone_sessions.pop(phone, None)
            return None
        return session

    def _event(
        self,
        session: Session,
        kind: str,
        actor: str,
        title: str,
        detail: str = "",
    ) -> TimelineEvent:
        event = TimelineEvent(
            at=utcnow(),
            clinical_minutes=round(session.clinical_minutes, 1),
            kind=kind,  # type: ignore[arg-type]
            actor=actor,  # type: ignore[arg-type]
            title=title,
            detail=detail,
        )
        session.timeline.append(event)
        return event

    def _monitor(self, session: Session) -> MonitorView:
        wall_minutes = max(0.0, (time.monotonic() - session.started_monotonic) / 60.0)
        return compute_monitor(
            session.scenario,
            session.clinical_minutes,
            wall_minutes,
            session.actions,
            session.difficulty,
            session.escalation_minutes,
        )

    def _reveal(self, session: Session, text: str) -> list[str]:
        normalized = " ".join(text.lower().split())
        case_file = session.scenario.get("case_file", {})
        rules = case_file.get("disclosure", [])
        newly_revealed: list[str] = []
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            topic = str(rule.get("topic", "")).strip()
            if not topic or topic in session.revealed_topics:
                continue
            triggers = [str(item).lower() for item in rule.get("triggers", [])]
            if any(trigger and trigger in normalized for trigger in triggers):
                session.revealed_topics.add(topic)
                newly_revealed.append(topic)
        if newly_revealed:
            self.transcripts.log(
                session.id,
                "history_revealed",
                topics=newly_revealed,
            )
        return newly_revealed

    def _hints(self, session: Session) -> list[str]:
        hints = session.scenario.get("hints", [])
        return [str(item) for item in hints] if isinstance(hints, list) else []

    def _next_hint(self, session: Session) -> str | None:
        hints = self._hints(session)
        if session.hint_index >= len(hints):
            return None
        hint = hints[session.hint_index]
        session.hint_index += 1
        return hint

    def _auto_hint(self, session: Session) -> str | None:
        if not auto_hints_enabled(session.difficulty):
            return None
        if session.student_turns not in {3, 6, 9}:
            return None
        if session.student_turns in session.auto_hints_shown:
            return None
        hint = self._next_hint(session)
        if hint is None:
            return None
        session.auto_hints_shown.add(session.student_turns)
        self._event(session, "hint", "supervisor", "Beginner hint", hint)
        return hint

    async def start(
        self,
        scenario_id: str,
        channel: str = "web",
        phone: str | None = None,
        student_name: str | None = None,
        difficulty: str | None = None,
    ) -> Session:
        scenario = self.scenarios.get(scenario_id)
        selected = normalize_difficulty(difficulty, str(scenario["difficulty"]))
        async with self._start_lock:
            self._purge_expired()
            self.transcripts.delete_expired(self.settings.transcript_retention_days)
            if phone is not None:
                active = self.get_active_by_phone(phone)
                if active is not None:
                    raise ActiveSessionExists(
                        "This phone number already has an active session. Reply END to finish it first."
                    )
            if self.active_count() >= self.settings.max_active_sessions:
                raise SessionCapacityReached(
                    "The simulator is at capacity. Please try again later."
                )
            session = Session(
                id=uuid.uuid4().hex,
                session_token=secrets.token_urlsafe(32),
                scenario=scenario,
                channel=channel,
                phone=phone,
                student_name=student_name,
                difficulty=selected,
                guideline=self.guidelines.for_scenario(scenario_id),
                transcript=[
                    TranscriptTurn(
                        role="patient",
                        content=str(scenario["opening_line"]),
                        created_at=utcnow(),
                    )
                ],
            )
            self._event(
                session,
                "system",
                "system",
                "Session started",
                f"{scenario['title']} ({selected})",
            )
            self._event(session, "message", "patient", "Patient opening statement", str(scenario["opening_line"]))
            self._sessions[session.id] = session
            if phone is not None:
                self._phone_sessions[phone] = session.id

        self.transcripts.log(
            session.id,
            "session_started",
            scenario_id=scenario["id"],
            channel=channel,
            phone=mask_phone(phone),
            difficulty=selected,
            guideline_mode=session.guideline.mode if session.guideline else "off",
        )
        return session

    async def message(self, session_id: str, text: str) -> TurnResult:
        session = self.get(session_id)
        async with session.lock:
            if session.status != "active":
                raise SessionCompleted(
                    "This session has already ended. Start a new session to continue."
                )
            if session.student_turns >= self.max_turns:
                raise TurnLimitReached(
                    "The turn limit has been reached. End the session to generate feedback."
                )

            newly_revealed = self._reveal(session, text)
            history = [
                {"role": turn.role, "content": turn.content} for turn in session.transcript
            ]
            history.append({"role": "student", "content": text})

            reply = await self.llm.patient_reply(
                session.scenario,
                history,
                sorted(session.revealed_topics),
            )

            session.clinical_minutes += QUESTION_MINUTES
            session.transcript.append(
                TranscriptTurn(role="student", content=text, created_at=utcnow())
            )
            session.student_turns += 1
            self._event(session, "message", "student", "Student question", text)
            self.transcripts.log(
                session.id,
                "student_message",
                turn=session.student_turns,
                text=text,
                revealed=newly_revealed,
            )
            session.transcript.append(
                TranscriptTurn(role="patient", content=reply, created_at=utcnow())
            )
            self._event(session, "message", "patient", "Patient reply", reply)
            self.transcripts.log(
                session.id,
                "patient_message",
                turn=session.student_turns,
                text=reply,
            )

            monitor = self._monitor(session)
            hint = self._auto_hint(session)
            auto_attempted = session.student_turns >= self.max_turns
            evaluation: Evaluation | None = None
            evaluation_error: str | None = None

            if auto_attempted:
                try:
                    evaluation = await self._evaluate(session)
                except Exception:
                    logger.exception("Evaluation failed for session %s", session.id)
                    evaluation_error = (
                        "Evaluation could not be generated. Try ending the session again."
                    )
                    session.evaluation_error = evaluation_error
                    self.transcripts.log(session.id, "evaluation_failed")

            return TurnResult(
                session_id=session.id,
                reply=reply,
                student_turns=session.student_turns,
                turns_remaining=max(0, self.max_turns - session.student_turns),
                status=session.status,
                monitor=monitor,
                auto_evaluated=evaluation is not None,
                evaluation=evaluation,
                evaluation_error=evaluation_error,
                hint=hint,
            )

    async def action(self, session_id: str, action_id: str) -> ActionResult:
        session = self.get(session_id)
        async with session.lock:
            if session.status != "active":
                raise SessionCompleted(
                    "This session has already ended. Start a new session to continue."
                )

            option = action_option(session.scenario, action_id)
            if not option.available:
                raise ActionUnavailable(
                    option.unavailable_reason or "That action is not available here."
                )
            already_taken = any(item.action_id == action_id for item in session.actions)
            if already_taken and not option.repeatable:
                raise ActionAlreadyDone("That action has already been completed.")

            monitor_before = self._monitor(session)
            record = perform_action(session.scenario, option, monitor_before)
            session.clinical_minutes += option.minutes
            record.clinical_minutes = round(session.clinical_minutes, 1)

            if action_id in ESCALATION_ACTIONS and session.escalation_minutes is None:
                session.escalation_minutes = record.clinical_minutes
                session.escalation_action = action_id

            session.actions.append(record)
            monitor = self._monitor(session)

            if action_id == "vitals":
                extra = "" if record.result.startswith("Take vitals") else record.result
                record.result = vitals_result(monitor)
                if extra:
                    record.result = f"{record.result} {extra}"

            kind = "vitals" if action_id == "vitals" else "escalation" if action_id in ESCALATION_ACTIONS else "action"
            self._event(session, kind, "nurse", option.label, record.result)
            self.transcripts.log(
                session.id,
                "clinical_action",
                action_id=action_id,
                label=option.label,
                minutes=option.minutes,
                result=record.result,
                warning=record.warning,
                unsafe=record.unsafe,
                clinical_minutes=session.clinical_minutes,
            )
            if record.unsafe:
                self.transcripts.log(session.id, "unsafe_action", action_id=action_id)
            if action_id in ESCALATION_ACTIONS:
                self.transcripts.log(
                    session.id,
                    "escalation",
                    action_id=action_id,
                    clinical_minutes=session.clinical_minutes,
                )

            actions_taken = list(dict.fromkeys(item.action_id for item in session.actions))
            return ActionResult(
                session_id=session.id,
                action_id=action_id,
                action_label=option.label,
                result=record.result,
                warning=record.warning,
                monitor=monitor,
                status=session.status,
                student_turns=session.student_turns,
                turns_remaining=max(0, self.max_turns - session.student_turns),
                actions_taken=actions_taken,
            )

    async def hint(self, session_id: str, question: str = "") -> HintResult:
        session = self.get(session_id)
        async with session.lock:
            if session.status != "active":
                raise SessionCompleted("This session has already ended.")
            if session.guideline is not None and session.guideline.mode != "full":
                raise GuidelineHintsUnavailable(
                    "This scenario has no guideline steps for supervisor hints."
                )
            if session.guideline is None and self.guidelines.enabled:
                raise GuidelineHintsUnavailable(
                    "This scenario has no guideline steps for supervisor hints."
                )
            if not hints_enabled(session.difficulty):
                raise HintsDisabled(
                    "Supervisor hints are disabled in Advanced mode. Trust your assessment."
                )
            if session.guideline is not None:
                if session.guideline_hints_used >= MAX_GUIDELINE_HINTS:
                    raise NoHintsRemaining(
                        f"No supervisor hints remain (maximum {MAX_GUIDELINE_HINTS} per session)."
                    )
                forbidden = tuple(
                    str(term)
                    for term in session.scenario.get("ground_truth", {}).get(
                        "forbidden_patient_terms", []
                    )
                )
                hint = self.guidelines.hint(
                    str(session.scenario["id"]), question, forbidden
                )
                session.guideline_hints_used += 1
                session.hints_used += 1
                self._event(
                    session,
                    "hint",
                    "supervisor",
                    "Guideline hint",
                    "Steps: " + ", ".join(hint.step_ids) if hint.step_ids else "No stored step matched",
                )
                self.transcripts.log(
                    session.id,
                    "guideline_hint",
                    question=question,
                    step_ids=hint.step_ids,
                    hints_used=session.hints_used,
                )
                return HintResult(
                    session_id=session.id,
                    hint=hint.text,
                    question=question,
                    step_ids=hint.step_ids,
                    hints_used=session.hints_used,
                    hint_penalty=hint_penalty(session.hints_used),
                    hints_remaining=max(0, MAX_GUIDELINE_HINTS - session.guideline_hints_used),
                    monitor=self._monitor(session),
                )

            if self.guidelines.enabled:
                raise GuidelineHintsUnavailable(
                    "This scenario has no guideline steps for supervisor hints."
                )

            hint = self._next_hint(session)
            if hint is None:
                raise NoHintsRemaining("No supervisor hints remain for this scenario.")
            session.hints_used += 1
            self._event(session, "hint", "supervisor", "Supervisor hint", hint)
            self.transcripts.log(
                session.id,
                "hint",
                hint=hint,
                hints_used=session.hints_used,
            )
            hints = self._hints(session)
            return HintResult(
                session_id=session.id,
                hint=hint,
                hints_used=session.hints_used,
                hint_penalty=hint_penalty(session.hints_used),
                hints_remaining=max(0, len(hints) - session.hint_index),
                monitor=self._monitor(session),
            )

    async def monitor(self, session_id: str) -> MonitorView:
        session = self.get(session_id)
        async with session.lock:
            return self._monitor(session)

    async def end(self, session_id: str) -> Evaluation:
        session = self.get(session_id)
        async with session.lock:
            if session.evaluation is not None:
                if session.status != "completed":
                    session.status = "completed"
                    self._unbind_phone(session)
                return session.evaluation
            try:
                evaluation = await self._evaluate(session)
            except Exception:
                self.transcripts.log(session.id, "evaluation_failed", trigger="student_end")
                raise
            return evaluation

    async def snapshot(self, session_id: str) -> SessionSnapshot:
        session = self.get(session_id)
        async with session.lock:
            return SessionSnapshot(
                session_id=session.id,
                scenario=self.scenarios.summary(str(session.scenario["id"])),
                status=session.status,
                student_turns=session.student_turns,
                turns_remaining=max(0, self.max_turns - session.student_turns),
                transcript=list(session.transcript),
                monitor=self._monitor(session),
                timeline=list(session.timeline),
                difficulty=session.difficulty,  # type: ignore[arg-type]
                hints_used=session.hints_used,
                hint_penalty=hint_penalty(session.hints_used),
                guideline=session.guideline.view() if session.guideline else None,
                guideline_hints_used=session.guideline_hints_used,
                evaluation=session.evaluation,
                evaluation_error=session.evaluation_error,
            )

    async def _evaluate(self, session: Session) -> Evaluation:
        history = [{"role": turn.role, "content": turn.content} for turn in session.transcript]
        monitor = self._monitor(session)
        context = {
            "difficulty": session.difficulty,
            "clinical_minutes": round(session.clinical_minutes, 1),
            "monitor": monitor.model_dump(),
            "escalation_minutes": session.escalation_minutes,
            "actions": [
                {
                    "action_id": action.action_id,
                    "label": action.label,
                    "clinical_minutes": action.clinical_minutes,
                    "result": action.result,
                    "unsafe": action.unsafe,
                    "effects": action.effects,
                }
                for action in session.actions
            ],
            "timeline": [event.model_dump() for event in session.timeline],
            "hints_used": session.hints_used,
            "guideline_hints_used": session.guideline_hints_used,
        }
        if session.guideline is not None and session.guideline.mode == "full":
            context["guideline_steps"] = [
                {"id": step.id, "text": step.text} for step in session.guideline.steps
            ]
        llm_evaluation = await self.llm.evaluate(session.scenario, history, context)
        assessment = assess(session.scenario, session)
        evaluation = merge_evaluation(session.scenario, session, llm_evaluation, assessment)
        session.evaluation = evaluation
        session.evaluation_error = None
        session.status = "completed"
        self.transcripts.log(session.id, "evaluation", evaluation=evaluation.model_dump())
        self._unbind_phone(session)
        return evaluation

    def _unbind_phone(self, session: Session) -> None:
        if session.phone is not None and self._phone_sessions.get(session.phone) == session.id:
            self._phone_sessions.pop(session.phone, None)
