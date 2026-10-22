from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import PROJECT_ROOT
from .models import GuidelineLink, GuidelineView

logger = logging.getLogger(__name__)

MAX_GUIDELINE_HINTS = 3
FULL_MODE_MIN_STEPS = 5
FULL_MODE_MAX_STEPS = 8
KNOWLEDGE_HUB_HOST = "knowledgehub.health.gov.za"

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "is",
    "it",
    "its",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "please",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


class GuidelineHintsUnavailable(RuntimeError):
    """Raised when a scenario has no usable guideline steps for hints."""


class GuidelineStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=80)
    text: str = Field(min_length=1, max_length=2000)

    @field_validator("id", "text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = " ".join(value.split()).strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class GuidelineDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    edition_year: str = Field(min_length=1, max_length=80)
    publisher: str | None = Field(default=None, max_length=300)
    url: str | None = None

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname != KNOWLEDGE_HUB_HOST
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(
                "document urls must be https on knowledgehub.health.gov.za or null"
            )
        return value


class ScenarioGuidelineFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_document: str = Field(min_length=1, max_length=120)
    section_text: str = Field(min_length=1, max_length=500)
    steps: list[GuidelineStep] = Field(default_factory=list)
    further_reading: list[str] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def unique_step_ids(cls, value: list[GuidelineStep]) -> list[GuidelineStep]:
        ids = [step.id for step in value]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        return value


class GuidelinesFile(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    note: str | None = Field(default=None, alias="_note")
    documents: dict[str, GuidelineDocument] = Field(default_factory=dict)
    scenarios: dict[str, ScenarioGuidelineFile] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScenarioGuideline:
    name: str
    edition_year: str
    section_text: str
    url: str | None
    steps: tuple[GuidelineStep, ...]
    further_reading: tuple[GuidelineLink, ...]

    @property
    def mode(self) -> Literal["links_only", "full"]:
        return "full" if self.steps else "links_only"

    def view(self) -> GuidelineView:
        return GuidelineView(
            name=self.name,
            edition_year=self.edition_year,
            section_text=self.section_text,
            url=self.url,
            mode=self.mode,
            further_reading=list(self.further_reading),
        )


@dataclass(frozen=True, slots=True)
class GuidelineHint:
    text: str
    step_ids: list[str]


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 2 and token not in _STOPWORDS
    }


def _mentions_forbidden(text: str, forbidden_terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(
        term.strip() and term.strip().lower() in lowered for term in forbidden_terms
    )


def select_steps(
    question: str,
    steps: tuple[GuidelineStep, ...],
    forbidden_terms: tuple[str, ...] = (),
) -> list[GuidelineStep]:
    """Select the stored step(s) with the strongest simple keyword overlap."""
    question_tokens = _tokens(question)
    if not question_tokens:
        return []

    scored: list[tuple[int, int, GuidelineStep]] = []
    for index, step in enumerate(steps):
        overlap = len(question_tokens & _tokens(step.text))
        if overlap and not _mentions_forbidden(step.text, forbidden_terms):
            scored.append((overlap, index, step))
    if not scored:
        return []

    best = max(score for score, _, _ in scored)
    return [step for score, _, step in scored if score == best][:2]


def _contains_page_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            if "page" in str(key).lower() or _contains_page_key(item):
                return True
    elif isinstance(value, list):
        return any(_contains_page_key(item) for item in value)
    return False


def _validation_message(exc: ValidationError) -> str:
    details = []
    for error in exc.errors(include_url=False, include_input=False):
        location = ".".join(str(part) for part in error.get("loc", ()))
        details.append(f"{location}: {error.get('msg', 'invalid value')}")
    return "; ".join(details) or "invalid schema"


class GuidelineRepository:
    def __init__(
        self,
        scenarios: dict[str, ScenarioGuideline] | None = None,
        source: Path | None = None,
    ) -> None:
        self._scenarios = scenarios or {}
        self.source = source

    @classmethod
    def disabled(cls) -> GuidelineRepository:
        return cls()

    @property
    def enabled(self) -> bool:
        return bool(self._scenarios)

    @property
    def scenarios(self) -> dict[str, ScenarioGuideline]:
        return dict(self._scenarios)

    @property
    def mode(self) -> str:
        if not self._scenarios:
            return "off"
        if any(item.mode == "full" for item in self._scenarios.values()):
            return "full"
        return "links_only"

    def for_scenario(self, scenario_id: str) -> ScenarioGuideline | None:
        return self._scenarios.get(scenario_id)

    def view_for(self, scenario_id: str) -> GuidelineView | None:
        guideline = self.for_scenario(scenario_id)
        return guideline.view() if guideline else None

    def hint(
        self,
        scenario_id: str,
        question: str,
        forbidden_terms: tuple[str, ...] = (),
    ) -> GuidelineHint:
        guideline = self.for_scenario(scenario_id)
        if guideline is None or guideline.mode != "full":
            raise GuidelineHintsUnavailable(
                "This scenario has no guideline steps for supervisor hints."
            )
        selected = select_steps(question, guideline.steps, forbidden_terms)
        if not selected:
            section = guideline.section_text.strip().rstrip(".")
            return GuidelineHint(text=f"Check the guideline: {section}.", step_ids=[])
        return GuidelineHint(
            text="\n".join(step.text for step in selected),
            step_ids=[step.id for step in selected],
        )


def load_guidelines(path: Path | str) -> GuidelineRepository:
    source = Path(path)
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    try:
        raw_text = source.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning(
            "Guidelines disabled: %s was not found. Set GUIDELINES_PATH or add the file.",
            source,
        )
        return GuidelineRepository.disabled()
    except OSError as exc:
        logger.warning("Guidelines disabled: could not read %s (%s).", source, exc)
        return GuidelineRepository.disabled()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Guidelines disabled: %s is not valid JSON (line %s, column %s).",
            source,
            exc.lineno,
            exc.colno,
        )
        return GuidelineRepository.disabled()

    if _contains_page_key(data):
        logger.warning(
            "Guidelines disabled: %s contains a page-number field; page numbers are not supported.",
            source,
        )
        return GuidelineRepository.disabled()

    try:
        parsed = GuidelinesFile.model_validate(data)
    except ValidationError as exc:
        logger.warning(
            "Guidelines disabled: %s is invalid (%s).", source, _validation_message(exc)
        )
        return GuidelineRepository.disabled()

    try:
        documents = parsed.documents
        scenarios: dict[str, ScenarioGuideline] = {}
        for scenario_id, entry in parsed.scenarios.items():
            count = len(entry.steps)
            if count != 0 and not (FULL_MODE_MIN_STEPS <= count <= FULL_MODE_MAX_STEPS):
                logger.warning(
                    "Guideline feature disabled for scenario '%s': %d steps supplied "
                    "(expected 0 or %d-%d).",
                    scenario_id,
                    count,
                    FULL_MODE_MIN_STEPS,
                    FULL_MODE_MAX_STEPS,
                )
                continue
            primary = documents.get(entry.primary_document)
            if primary is None:
                raise ValueError(
                    f"scenario '{scenario_id}' references unknown document "
                    f"'{entry.primary_document}'"
                )
            readings: list[GuidelineLink] = []
            for document_id in entry.further_reading:
                document = documents.get(document_id)
                if document is None:
                    raise ValueError(
                        f"scenario '{scenario_id}' references unknown document '{document_id}'"
                    )
                readings.append(
                    GuidelineLink(
                        title=document.title,
                        edition_year=document.edition_year,
                        url=document.url,
                    )
                )
            scenarios[scenario_id] = ScenarioGuideline(
                name=primary.title,
                edition_year=primary.edition_year,
                section_text=entry.section_text,
                url=primary.url,
                steps=tuple(entry.steps),
                further_reading=tuple(readings),
            )
    except ValueError as exc:
        logger.warning("Guidelines disabled: %s is invalid (%s).", source, exc)
        return GuidelineRepository.disabled()

    repository = GuidelineRepository(scenarios, source)
    if not repository.enabled:
        logger.warning("Guidelines disabled: %s has no usable scenario definitions.", source)
    return repository
