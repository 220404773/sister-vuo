"""The /v1 endpoints: Learning tab, written cases and integrated questions (the approved API spec)."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, ConfigDict, Field

from .written import (
    MAX_EVAL_ATTEMPTS,
    WrittenCaseError,
    WrittenCaseRepository,
    WrittenSessionStore,
    answer_question,
    clean_answers,
    clean_text,
    evaluate_answers,
    find_abbreviations,
    judge_integrated,
)

logger = logging.getLogger(__name__)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StartBody(Strict):
    pass


class AskBody(Strict):
    question: str = Field(min_length=1, max_length=500)
    topic: Literal["history", "examination", "investigations", "observations", "auto"] = "auto"


class AbbreviationBody(Strict):
    texts: list[str] = Field(max_length=50)


class SubmitBody(Strict):
    answers: dict[str, Any]


class IntegratedBody(Strict):
    choice: str | None = Field(default=None, max_length=2)
    value: bool | None = None
    justification: str | None = Field(default=None, max_length=2000)
    text: str | None = Field(default=None, max_length=6000)


def _body_hash(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def build_router(
    repository: WrittenCaseRepository,
    store: WrittenSessionStore,
    llm: Any,
    prompts: dict[str, str],
    require_api_key: Any,
    limit_start: Any,
    limit_turn: Any,
) -> APIRouter:
    router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])

    def session_for(session_id: str, token: str | None) -> Any:
        return store.authorize(session_id, token)

    # --- Learning tab ---------------------------------------------------------------------

    @router.get("/learning/conditions")
    async def list_conditions() -> dict[str, Any]:
        return {"items": repository.condition_list(), "next_cursor": None}

    @router.get("/learning/conditions/{condition_id}")
    async def get_condition(condition_id: str) -> dict[str, Any]:
        return repository.learning_view(repository.get_condition(condition_id))

    # --- Clinical Scenarios tab ---------------------------------------------------------

    @router.get("/cases")
    async def list_cases(request: Request) -> dict[str, Any]:
        chat = [
            {"kind": "chat", "id": item.id, "title": item.title}
            for item in request.app.state.manager.scenarios.list_summaries()
        ]
        return {"items": chat + repository.case_list(), "integrated_available": bool(repository.integrated)}

    @router.post("/written-cases/{case_id}/sessions", status_code=201, dependencies=[Depends(limit_start)])
    async def start_written(case_id: str, _: StartBody | None = None) -> dict[str, Any]:
        record = repository.get_case(case_id)
        session, token = store.create(case_id)
        return {
            "session_id": session.session_id,
            "session_token": token,
            "case": repository.display(record),
            "limits": {"max_ask": store.max_ask, "max_answer_chars": 20_000, "max_rows": 200},
        }

    @router.post("/written-cases/sessions/{session_id}/ask", dependencies=[Depends(limit_turn)])
    async def ask(
        session_id: str,
        body: AskBody,
        x_session_token: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None),
    ) -> dict[str, Any]:
        session = session_for(session_id, x_session_token)
        if idempotency_key and idempotency_key in session.ask_replies:
            return session.ask_replies[idempotency_key]
        if session.state == "submitted":
            raise WrittenCaseError(409, "submitted", "This case has been submitted.")
        if len(session.ask_log) >= store.max_ask:
            raise WrittenCaseError(429, "ask_limit", "You have asked the maximum number of questions for this case.")
        question = clean_text(body.question, 500)
        answer, source, topic = answer_question(repository.get_case(session.case_id), question, body.topic)
        entry = {"id": str(len(session.ask_log) + 1), "question": question, "topic": topic,
                 "source": source, "answer": answer}
        session.ask_log.append(entry)
        reply = {"answer": answer, "source": source, "log_entry": entry}
        if idempotency_key:
            session.ask_replies[idempotency_key[:100]] = reply
        return reply

    @router.get("/written-cases/sessions/{session_id}/log")
    async def ask_log(session_id: str, x_session_token: str | None = Header(default=None)) -> dict[str, Any]:
        session = session_for(session_id, x_session_token)
        return {"items": session.ask_log, "next_cursor": None}

    @router.post("/written-cases/sessions/{session_id}/abbreviations", dependencies=[Depends(limit_turn)])
    async def abbreviations(
        session_id: str, body: AbbreviationBody, x_session_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        session_for(session_id, x_session_token)
        flags = []
        for index, text in enumerate(body.texts):
            for flag in find_abbreviations(text[:2000]):
                flags.append({"text_index": index, **flag})
        return {"flags": flags}

    @router.post("/written-cases/sessions/{session_id}/submit", dependencies=[Depends(limit_turn)])
    async def submit(
        session_id: str, body: SubmitBody, x_session_token: str | None = Header(default=None)
    ) -> dict[str, Any]:
        session = session_for(session_id, x_session_token)
        answers = clean_answers(body.answers)
        digest = _body_hash(answers)
        if session.state == "submitted":
            # Idempotent: the same answers return the stored result; different ones conflict.
            if digest == session.answers_hash and session.result is not None:
                return session.result
            raise WrittenCaseError(409, "submitted", "This case has already been submitted.")
        record = repository.get_case(session.case_id)
        session.state = "submitted"  # the reveal gate: nothing below runs before this line
        session.answers, session.answers_hash = answers, digest
        session.eval_attempts = 1
        evaluation = await evaluate_answers(llm, prompts["written"], record, answers)
        session.result = {"status": "submitted", "reveal": repository.reveal(record), "evaluation": evaluation}
        return session.result

    @router.post("/written-cases/sessions/{session_id}/evaluate", dependencies=[Depends(limit_turn)])
    async def evaluate_again(session_id: str, x_session_token: str | None = Header(default=None)) -> dict[str, Any]:
        session = session_for(session_id, x_session_token)
        if session.state != "submitted" or session.result is None:
            raise WrittenCaseError(409, "not_submitted", "Submit your answers first.")
        if session.result["evaluation"]["status"] == "complete":
            return session.result["evaluation"]
        if session.eval_attempts >= MAX_EVAL_ATTEMPTS:
            raise WrittenCaseError(429, "eval_limit", "The evaluation could not be completed. Please start the case again later.")
        session.eval_attempts += 1
        evaluation = await evaluate_answers(llm, prompts["written"], repository.get_case(session.case_id), session.answers)
        session.result["evaluation"] = evaluation
        return evaluation

    @router.get("/written-cases/sessions/{session_id}/result")
    async def result(session_id: str, x_session_token: str | None = Header(default=None)) -> dict[str, Any]:
        session = session_for(session_id, x_session_token)
        if session.result is None:
            raise WrittenCaseError(409, "not_submitted", "Submit your answers first.")
        return session.result

    # --- integrated questions (no session; rate-limited) ---------------------------------

    @router.get("/integrated-questions")
    async def integrated(condition: str | None = None) -> dict[str, Any]:
        # The same condition list as the Learning tab: questions for one condition at a time.
        return {"items": repository.integrated_view(condition)}

    @router.post("/integrated-questions/{question_id}/answer", dependencies=[Depends(limit_turn)])
    async def integrated_answer(question_id: str, body: IntegratedBody) -> dict[str, Any]:
        item = repository.integrated_item(question_id)
        kind = item["kind"]
        if kind == "mcq":
            keys = "ABCDEFGH"[: len(item["options"])]
            if not body.choice or body.choice.upper() not in keys:
                raise WrittenCaseError(400, "invalid", "Choose one of the options.")
            correct = keys.index(body.choice.upper()) == item["answer"]
            answer_key = keys[item["answer"]]
            return {"correct": correct, "marked_by": "code", "justification_evaluation": None,
                    "feedback": item.get("reason", ""),
                    "explanation": {"answer": f"{answer_key}: {item['options'][item['answer']]}",
                                    "reason": item.get("reason", "")}}
        if kind == "true_false":
            if body.value is None:
                raise WrittenCaseError(400, "invalid", "Choose true or false.")
            justification = clean_text(body.justification, 2000)
            judged = (await judge_integrated(llm, prompts["integrated"], item, justification)
                      if justification else {"status": "complete", "feedback": "Add a one-line justification."})
            return {"correct": body.value == item["answer"], "marked_by": "code",
                    "justification_evaluation": judged, "feedback": judged["feedback"],
                    "explanation": {"answer": "True" if item["answer"] else "False",
                                    "reason": item.get("justification", "")}}
        text = clean_text(body.text, 6000)
        if not text:
            raise WrittenCaseError(400, "invalid", "Write your answer first.")
        judged = await judge_integrated(llm, prompts["integrated"], item, text)
        return {"correct": None, "marked_by": "evaluator", "justification_evaluation": judged,
                "feedback": judged["feedback"],
                "explanation": {"columns": item.get("columns", []), "rows": item.get("rows", [])}}

    return router

