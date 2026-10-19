from __future__ import annotations

import inspect
import logging
import secrets
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .config import Settings, get_settings
from .llm import LLMConfigurationError, LLMError, LLMService
from .models import (
    EndResponse,
    MessageRequest,
    MessageResponse,
    ScenarioSummary,
    SessionSnapshot,
    StartRequest,
    StartResponse,
)
from .sessions import (
    ActiveSessionExists,
    ScenarioNotFound,
    ScenarioRepository,
    SessionCapacityReached,
    SessionCompleted,
    SessionManager,
    SessionNotFound,
    TranscriptLogger,
    TurnLimitReached,
)
from .whatsapp import MessageDeduper, WhatsAppClient
from .whatsapp import router as whatsapp_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(
    settings_obj: Settings | None = None,
    llm_service: Any | None = None,
    whatsapp_client: Any | None = None,
) -> FastAPI:
    app_settings = settings_obj or get_settings()
    scenarios = ScenarioRepository(app_settings.scenario_dir)
    transcripts = TranscriptLogger(app_settings.log_dir)
    llm = llm_service or LLMService(app_settings)
    manager = SessionManager(app_settings, scenarios, llm, transcripts)
    whatsapp = whatsapp_client or WhatsAppClient(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if not app_settings.llm_configured:
            logger.warning(
                "LLM is not configured. Set LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL in .env."
            )
        if not app_settings.whatsapp_configured:
            logger.warning(
                "WhatsApp webhook is disabled until WHATSAPP_VERIFY_TOKEN, "
                "WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_APP_SECRET are set."
            )
        if not app_settings.app_api_key:
            logger.warning("APP_API_KEY is not set; the REST API and chat UI are unauthenticated.")
        yield
        for resource in (llm, whatsapp):
            close = getattr(resource, "close", None)
            if close is None:
                continue
            result = close()
            if inspect.isawaitable(result):
                await result

    app = FastAPI(
        title="Sister Vuo",
        description="AI nursing patient simulator",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.scenarios = scenarios
    app.state.manager = manager
    app.state.whatsapp = whatsapp
    app.state.whatsapp_dedupe = MessageDeduper()

    @app.exception_handler(ScenarioNotFound)
    async def handle_scenario_not_found(_: Request, exc: ScenarioNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(SessionNotFound)
    async def handle_session_not_found(_: Request, exc: SessionNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ActiveSessionExists)
    async def handle_active_session(_: Request, exc: ActiveSessionExists) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(TurnLimitReached)
    async def handle_turn_limit(_: Request, exc: TurnLimitReached) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SessionCompleted)
    async def handle_session_completed(_: Request, exc: SessionCompleted) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SessionCapacityReached)
    async def handle_capacity(_: Request, exc: SessionCapacityReached) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMConfigurationError)
    async def handle_llm_config(_: Request, exc: LLMConfigurationError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(LLMError)
    async def handle_llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.warning("LLM request failed: %s", exc)
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    async def require_api_key(request: Request) -> None:
        expected = request.app.state.settings.app_api_key
        if not expected:
            return
        supplied = request.headers.get("x-api-key", "")
        if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
            raise HTTPException(status_code=401, detail="A valid API key is required.")

    protected = [Depends(require_api_key)]

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(app_settings.static_dir / "index.html")

    @app.get("/scenarios", response_model=list[ScenarioSummary], dependencies=protected)
    async def list_scenarios(request: Request) -> list[ScenarioSummary]:
        return request.app.state.scenarios.list_summaries()

    @app.get("/health")
    async def health(request: Request) -> dict[str, Any]:
        settings: Settings = request.app.state.settings
        manager: SessionManager = request.app.state.manager
        return {
            "status": "ok",
            "llm_configured": settings.llm_configured,
            "whatsapp_configured": settings.whatsapp_configured,
            "scenarios": len(manager.scenarios.list_summaries()),
            "active_sessions": manager.active_count(),
        }

    @app.post("/session/start", response_model=StartResponse, dependencies=protected)
    async def start_session(payload: StartRequest, request: Request) -> StartResponse:
        settings: Settings = request.app.state.settings
        manager: SessionManager = request.app.state.manager
        session = await manager.start(
            payload.scenario_id,
            channel="web",
            student_name=payload.student_name,
        )
        summary = manager.scenarios.summary(str(session.scenario["id"]))
        return StartResponse(
            session_id=session.id,
            scenario=summary,
            opening_message=session.transcript[0].content,
            student_turns=session.student_turns,
            max_student_turns=settings.max_student_turns,
            turns_remaining=settings.max_student_turns - session.student_turns,
            status=session.status,
        )

    @app.post(
        "/session/{session_id}/message", response_model=MessageResponse, dependencies=protected
    )
    async def send_message(
        session_id: str,
        payload: MessageRequest,
        request: Request,
    ) -> MessageResponse:
        manager: SessionManager = request.app.state.manager
        outcome = await manager.message(session_id, payload.message)
        return MessageResponse(
            session_id=outcome.session_id,
            reply=outcome.reply,
            student_turns=outcome.student_turns,
            turns_remaining=outcome.turns_remaining,
            status=outcome.status,
            auto_evaluated=outcome.auto_evaluated,
            evaluation=outcome.evaluation,
            evaluation_error=outcome.evaluation_error,
        )

    @app.post("/session/{session_id}/end", response_model=EndResponse, dependencies=protected)
    async def end_session(session_id: str, request: Request) -> EndResponse:
        manager: SessionManager = request.app.state.manager
        evaluation = await manager.end(session_id)
        return EndResponse(session_id=session_id, evaluation=evaluation)

    @app.get("/session/{session_id}", response_model=SessionSnapshot, dependencies=protected)
    async def get_session(session_id: str, request: Request) -> SessionSnapshot:
        manager: SessionManager = request.app.state.manager
        return await manager.snapshot(session_id)

    app.include_router(whatsapp_router)
    return app


app = create_app()
