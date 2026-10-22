from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

Score = Annotated[int, Field(ge=0, le=5)]
Difficulty = Literal["Beginner", "Intermediate", "Advanced"]


class StartRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=80)
    student_name: str | None = Field(default=None, max_length=80)
    difficulty: Difficulty | None = None

    @field_validator("scenario_id")
    @classmethod
    def clean_scenario_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("scenario_id must not be empty")
        return value


class MessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def clean_message(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be empty")
        return value


class ActionRequest(BaseModel):
    action_id: str = Field(min_length=1, max_length=40)

    @field_validator("action_id")
    @classmethod
    def clean_action_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("action_id must not be empty")
        return value


class ScenarioSummary(BaseModel):
    id: str
    title: str
    setting: str
    difficulty: str
    description: str
    patient_label: str
    learning_objectives: list[str]


class VitalsView(BaseModel):
    heart_rate: int
    systolic_bp: int
    diastolic_bp: int
    spo2: int
    respiratory_rate: int
    pain_score: int
    temperature: float


class MonitorView(BaseModel):
    vitals: VitalsView
    clinical_minutes: float = Field(ge=0)
    monitor_minutes: float = Field(ge=0)
    status: Literal["stable", "deteriorating", "critical"]
    alerts: list[str] = Field(default_factory=list)
    escalated: bool = False
    escalation_minutes: float | None = None
    escalation_target_minutes: float = Field(gt=0)


class ActionOption(BaseModel):
    id: str
    label: str
    description: str
    minutes: float = Field(ge=0)
    category: str
    available: bool = True
    unavailable_reason: str | None = None
    repeatable: bool = False


class TimelineEvent(BaseModel):
    at: str
    clinical_minutes: float = Field(ge=0)
    kind: Literal["message", "action", "vitals", "hint", "escalation", "system"]
    actor: Literal["student", "patient", "nurse", "system", "supervisor"]
    title: str
    detail: str = ""


GuidelineMode = Literal["links_only", "full"]


class GuidelineLink(BaseModel):
    title: str
    edition_year: str
    url: str | None = None


class GuidelineView(BaseModel):
    name: str
    edition_year: str
    section_text: str
    url: str | None = None
    mode: GuidelineMode
    further_reading: list[GuidelineLink] = Field(default_factory=list)


class ProtocolStepResult(BaseModel):
    id: str
    status: Literal["met", "partial", "missed"]
    evidence_turn: int | None = Field(default=None, ge=1)


class StartResponse(BaseModel):
    session_id: str
    scenario: ScenarioSummary
    opening_message: str
    student_turns: int
    max_student_turns: int
    turns_remaining: int
    status: Literal["active", "completed"]
    difficulty: Difficulty
    hints_enabled: bool
    monitor: MonitorView
    actions: list[ActionOption]
    guideline: GuidelineView | None = None


class DomainScores(BaseModel):
    red_flag_recognition: Score
    history_completeness: Score
    prioritisation: Score
    safety: Score
    communication_empathy: Score
    time_to_escalation: Score


class Debrief(BaseModel):
    what_went_well: list[str] = Field(default_factory=list)
    what_was_missed: list[str] = Field(default_factory=list)
    critical_errors: list[str] = Field(default_factory=list)
    ideal_approach: list[str] = Field(default_factory=list)
    model_answer: str = ""
    reflection_questions: list[str] = Field(default_factory=list)
    guideline_references: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    time_to_escalation_minutes: float | None = None
    escalation_target_minutes: float = 10.0
    actions_taken: list[str] = Field(default_factory=list)


class Evaluation(BaseModel):
    overall_score: float = Field(ge=0, le=5)
    domain_scores: DomainScores
    strengths: list[str] = Field(default_factory=list)
    missed_red_flags: list[str] = Field(default_factory=list)
    unsafe_actions: list[str] = Field(default_factory=list)
    critical_errors: list[str] = Field(default_factory=list)
    recommended_learning: list[str] = Field(default_factory=list)
    feedback_to_student: str
    hint_penalty: float = Field(default=0.0, ge=0, le=2)
    debrief: Debrief = Field(default_factory=Debrief)
    protocol_steps: list[ProtocolStepResult] | None = None


class MessageResponse(BaseModel):
    session_id: str
    reply: str
    student_turns: int
    turns_remaining: int
    status: Literal["active", "completed"]
    auto_evaluated: bool = False
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
    monitor: MonitorView
    hint: str | None = None


class ActionResponse(BaseModel):
    session_id: str
    action_id: str
    action_label: str
    result: str
    warning: str | None = None
    monitor: MonitorView
    status: Literal["active", "completed"]
    student_turns: int
    turns_remaining: int
    actions_taken: list[str] = Field(default_factory=list)


class HintRequest(BaseModel):
    question: str = Field(default="", max_length=2000)

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        return value.strip()


class HintResponse(BaseModel):
    session_id: str
    hint: str
    question: str = ""
    step_ids: list[str] = Field(default_factory=list)
    hints_used: int
    hint_penalty: float
    hints_remaining: int
    monitor: MonitorView


class EndResponse(BaseModel):
    session_id: str
    evaluation: Evaluation


class TranscriptTurn(BaseModel):
    role: Literal["student", "patient"]
    content: str
    created_at: str


class SessionSnapshot(BaseModel):
    session_id: str
    scenario: ScenarioSummary
    status: Literal["active", "completed"]
    student_turns: int
    turns_remaining: int
    transcript: list[TranscriptTurn]
    monitor: MonitorView
    timeline: list[TimelineEvent]
    difficulty: Difficulty
    hints_used: int
    hint_penalty: float
    guideline: GuidelineView | None = None
    guideline_hints_used: int = 0
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
