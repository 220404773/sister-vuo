from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .models import Evaluation, ScenarioSummary, SessionSnapshot, TranscriptTurn

logger = logging.getLogger(__name__)


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
            "history",
            "ground_truth",
            "learning_objectives",
        ):
            _require(data, key, path)
        for key in ("speaker", "patient", "history", "ground_truth"):
            if not isinstance(data[key], dict):
                raise ScenarioError(f"{path.name} key '{key}' must be an object")
        for key in ("title", "setting", "difficulty", "description", "opening_line"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ScenarioError(f"{path.name} key '{key}' must be a non-empty string")
        if not isinstance(data["learning_objectives"], list):
            raise ScenarioError(f"{path.name} key 'learning_objectives' must be a list")
        for key in ("name", "role", "language", "style"):
            _require(data["speaker"], key, path)
        for key in ("name", "age", "sex"):
            _require(data["patient"], key, path)
        for key in ("diagnosis", "red_flags", "expected_actions", "forbidden_patient_terms"):
            _require(data["ground_truth"], key, path)
        if not isinstance(data["ground_truth"]["forbidden_patient_terms"], list):
            raise ScenarioError(f"{path.name} forbidden_patient_terms must be a list")

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


@dataclass(slots=True)
class Session:
    id: str
    scenario: dict[str, Any]
    channel: str
    phone: str | None = None
    student_name: str | None = None
    transcript: list[TranscriptTurn] = field(default_factory=list)
    student_turns: int = 0
    status: str = "active"
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
    created_at: str = field(default_factory=utcnow)
    last_activity: float = field(default_factory=time.monotonic)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


@dataclass(slots=True)
class TurnResult:
    session_id: str
    reply: str
    student_turns: int
    turns_remaining: int
    status: str
    auto_evaluated: bool = False
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None


class SessionManager:
    def __init__(
        self,
        settings: Settings,
        scenarios: ScenarioRepository,
        llm: Any,
        transcripts: TranscriptLogger,
    ) -> None:
        self.settings = settings
        self.scenarios = scenarios
        self.llm = llm
        self.transcripts = transcripts
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

    async def start(
        self,
        scenario_id: str,
        channel: str = "web",
        phone: str | None = None,
        student_name: str | None = None,
    ) -> Session:
        scenario = self.scenarios.get(scenario_id)
        async with self._start_lock:
            self._purge_expired()
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
                scenario=scenario,
                channel=channel,
                phone=phone,
                student_name=student_name,
                transcript=[
                    TranscriptTurn(
                        role="patient",
                        content=str(scenario["opening_line"]),
                        created_at=utcnow(),
                    )
                ],
            )
            self._sessions[session.id] = session
            if phone is not None:
                self._phone_sessions[phone] = session.id

        self.transcripts.log(
            session.id,
            "session_started",
            scenario_id=scenario["id"],
            channel=channel,
            phone=mask_phone(phone),
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

            student_turn = TranscriptTurn(role="student", content=text, created_at=utcnow())
            history = [
                {"role": turn.role, "content": turn.content} for turn in session.transcript
            ]
            history.append({"role": "student", "content": text})

            reply = await self.llm.patient_reply(session.scenario, history)

            session.transcript.append(student_turn)
            session.student_turns += 1
            self.transcripts.log(
                session.id,
                "student_message",
                turn=session.student_turns,
                text=text,
            )
            patient_turn = TranscriptTurn(role="patient", content=reply, created_at=utcnow())
            session.transcript.append(patient_turn)
            self.transcripts.log(
                session.id,
                "patient_message",
                turn=session.student_turns,
                text=reply,
            )

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
                auto_evaluated=evaluation is not None,
                evaluation=evaluation,
                evaluation_error=evaluation_error,
            )

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
                evaluation=session.evaluation,
                evaluation_error=session.evaluation_error,
            )

    async def _evaluate(self, session: Session) -> Evaluation:
        history = [{"role": turn.role, "content": turn.content} for turn in session.transcript]
        evaluation = await self.llm.evaluate(session.scenario, history)
        session.evaluation = evaluation
        session.evaluation_error = None
        session.status = "completed"
        self.transcripts.log(session.id, "evaluation", evaluation=evaluation.model_dump())
        self._unbind_phone(session)
        return evaluation

    def _unbind_phone(self, session: Session) -> None:
        if session.phone is not None and self._phone_sessions.get(session.phone) == session.id:
            self._phone_sessions.pop(session.phone, None)
