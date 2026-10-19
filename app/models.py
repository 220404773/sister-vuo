from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

Score = Annotated[int, Field(ge=0, le=5)]


class StartRequest(BaseModel):
    scenario_id: str = Field(min_length=1, max_length=80)
    student_name: str | None = Field(default=None, max_length=80)

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


class ScenarioSummary(BaseModel):
    id: str
    title: str
    setting: str
    difficulty: str
    description: str
    patient_label: str
    learning_objectives: list[str]


class StartResponse(BaseModel):
    session_id: str
    scenario: ScenarioSummary
    opening_message: str
    student_turns: int
    max_student_turns: int
    turns_remaining: int
    status: Literal["active", "completed"]


class DomainScores(BaseModel):
    history_taking: Score
    clinical_reasoning: Score
    communication_empathy: Score
    language_accessibility: Score
    safety_referral: Score


class Evaluation(BaseModel):
    overall_score: float = Field(ge=0, le=5)
    domain_scores: DomainScores
    strengths: list[str] = Field(default_factory=list)
    missed_red_flags: list[str] = Field(default_factory=list)
    unsafe_actions: list[str] = Field(default_factory=list)
    recommended_learning: list[str] = Field(default_factory=list)
    feedback_to_student: str


class MessageResponse(BaseModel):
    session_id: str
    reply: str
    student_turns: int
    turns_remaining: int
    status: Literal["active", "completed"]
    auto_evaluated: bool = False
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None


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
    evaluation: Evaluation | None = None
    evaluation_error: str | None = None
