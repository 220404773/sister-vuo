from __future__ import annotations

import asyncio
import inspect
import logging
import secrets
from contextlib import asynccontextmanager, suppress
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from .clinical import (
    ActionAlreadyDone,
    ActionUnavailable,
    UnknownAction,
    action_options,
    hints_enabled,
)
from .config import Settings, get_settings
from .guidelines import GuidelineHintsUnavailable, load_guidelines
from .llm import LLMConfigurationError, LLMError, LLMService
from .models import (
    ActionRequest,
    ActionResponse,
    EndResponse,
    HintRequest,
    HintResponse,
    MessageRequest,
    MessageResponse,
    MonitorView,
    ScenarioSummary,
    SessionSnapshot,
    StartRequest,
    StartResponse,
)
from .ratelimit import RateLimiter, SlidingWindowLimiter
from .sessions import (
    ActiveSessionExists,
    HintsDisabled,
    NoHintsRemaining,
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

RETENTION_SWEEP_SECONDS = 60 * 60

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
        "form-action 'self'"
    ),
    "Strict-Transport-Security": "max-age=31536000",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
}


async def _retention_sweeper(
    transcripts: TranscriptLogger,
    retention_days: int,
    interval_seconds: float = RETENTION_SWEEP_SECONDS,
) -> None:
    """Delete expired transcripts even when the app runs for a long time."""
    while True:
        await asyncio.sleep(interval_seconds)
        transcripts.delete_expired(retention_days)


def create_app(
    settings_obj: Settings | None = None,
    llm_service: Any | None = None,
    whatsapp_client: Any | None = None,
) -> FastAPI:
    app_settings = settings_obj or get_settings()
    if app_settings.app_env == "production" and not app_settings.app_api_key:
        raise RuntimeError(
            "APP_ENV is production but APP_API_KEY is not set. Refusing to start with an "
            "open API. Set APP_API_KEY or use APP_ENV=development."
        )
    scenarios = ScenarioRepository(app_settings.scenario_dir)
    transcripts = TranscriptLogger(app_settings.log_dir)
    transcripts.delete_expired(app_settings.transcript_retention_days)
    llm = llm_service or LLMService(app_settings)
    guidelines = load_guidelines(app_settings.guidelines_path)
    manager = SessionManager(app_settings, scenarios, llm, transcripts, guidelines)
    whatsapp = whatsapp_client or WhatsAppClient(app_settings)
    rate_limiter = RateLimiter(app_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        transcripts.delete_expired(app_settings.transcript_retention_days)
        config_issues = app_settings.llm_config_issues()
        if config_issues:
            logger.error("LLM configuration is incomplete: %s", " ".join(config_issues))
        else:
            logger.info(
                "LLM configured: base_url=%s model=%s timeout=%ss",
                app_settings.llm_base_url,
                app_settings.llm_model,
                app_settings.llm_timeout_seconds,
            )
        if not app_settings.whatsapp_configured:
            logger.warning(
                "WhatsApp webhook is disabled until WHATSAPP_VERIFY_TOKEN, "
                "WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_APP_SECRET are set."
            )
        if not app_settings.app_api_key:
            logger.warning("APP_API_KEY is not set; the REST API and chat UI are unauthenticated.")
        sweeper = asyncio.create_task(
            _retention_sweeper(transcripts, app_settings.transcript_retention_days)
        )
        try:
            yield
        finally:
            sweeper.cancel()
            with suppress(asyncio.CancelledError):
                await sweeper
            transcripts.delete_expired(app_settings.transcript_retention_days)
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
    app.state.guidelines = guidelines
    app.state.manager = manager
    app.state.whatsapp = whatsapp
    app.state.whatsapp_dedupe = MessageDeduper()
    app.state.rate_limiter = rate_limiter

    @app.middleware("http")
    async def add_security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers[name] = value
        return response

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

    @app.exception_handler(UnknownAction)
    async def handle_unknown_action(_: Request, exc: UnknownAction) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ActionUnavailable)
    async def handle_action_unavailable(_: Request, exc: ActionUnavailable) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ActionAlreadyDone)
    async def handle_action_done(_: Request, exc: ActionAlreadyDone) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(HintsDisabled)
    async def handle_hints_disabled(_: Request, exc: HintsDisabled) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(NoHintsRemaining)
    async def handle_no_hints(_: Request, exc: NoHintsRemaining) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(GuidelineHintsUnavailable)
    async def handle_guideline_hints_unavailable(
        _: Request, exc: GuidelineHintsUnavailable
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

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

    async def require_session_token(session_id: str, request: Request) -> None:
        manager: SessionManager = request.app.state.manager
        manager.authorize(session_id, request.headers.get("x-session-token"))

    def client_host(request: Request) -> str:
        return request.client.host if request.client is not None else "unknown"

    def enforce_limit(limiter: SlidingWindowLimiter, key: str) -> None:
        retry_after = limiter.check(key)
        if retry_after is not None:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please slow down and try again shortly.",
                headers={"Retry-After": str(int(retry_after))},
            )

    async def limit_start(request: Request) -> None:
        enforce_limit(rate_limiter.start, client_host(request))

    async def limit_turn(request: Request) -> None:
        enforce_limit(rate_limiter.turn, client_host(request))

    async def limit_probe(request: Request) -> None:
        enforce_limit(rate_limiter.probe, client_host(request))

    protected_session = [Depends(require_api_key), Depends(require_session_token), Depends(limit_turn)]

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(app_settings.static_dir / "index.html")

    @app.get(
        "/scenarios",
        response_model=list[ScenarioSummary],
        dependencies=[Depends(require_api_key), Depends(limit_start)],
    )
    async def list_scenarios(request: Request) -> list[ScenarioSummary]:
        return request.app.state.scenarios.list_summaries()

    @app.get("/health")
    async def health(request: Request, probe: bool = False) -> dict[str, Any]:
        """Fast status by default; probe=true makes one tiny model call."""
        settings: Settings = request.app.state.settings
        manager: SessionManager = request.app.state.manager
        payload: dict[str, Any] = {
            "status": "ok",
            "llm_configured": settings.llm_configured,
            "llm_config_issues": settings.llm_config_issues(),
            "whatsapp_configured": settings.whatsapp_configured,
            "scenarios": len(manager.scenarios.list_summaries()),
            "active_sessions": manager.active_count(),
            "guidelines": manager.guidelines.mode,
        }
        if probe:
            await require_api_key(request)
            enforce_limit(rate_limiter.probe, client_host(request))
            result = await llm.probe()
            payload["llm_base_url"] = settings.llm_base_url
            payload["llm_model"] = settings.llm_model
            payload["llm"] = result
            if not result["ok"]:
                payload["status"] = "degraded"
        return payload

    @app.get("/health/llm", dependencies=[Depends(require_api_key), Depends(limit_probe)])
    async def health_llm(request: Request) -> dict[str, Any]:
        """Dedicated deep check: always makes one tiny model call."""
        settings: Settings = request.app.state.settings
        result = await llm.probe()
        return {
            "status": "ok" if result["ok"] else "degraded",
            "llm_configured": settings.llm_configured,
            "llm_base_url": settings.llm_base_url,
            "llm_model": settings.llm_model,
            "llm": result,
        }

    @app.post(
        "/session/start",
        response_model=StartResponse,
        dependencies=[Depends(require_api_key), Depends(limit_start)],
    )
    async def start_session(payload: StartRequest, request: Request) -> StartResponse:
        settings: Settings = request.app.state.settings
        manager: SessionManager = request.app.state.manager
        session = await manager.start(
            payload.scenario_id,
            channel="web",
            student_name=payload.student_name,
            difficulty=payload.difficulty,
        )
        summary = manager.scenarios.summary(str(session.scenario["id"]))
        return StartResponse(
            session_id=session.id,
            session_token=session.session_token or "",
            scenario=summary,
            opening_message=session.transcript[0].content,
            student_turns=session.student_turns,
            max_student_turns=settings.max_student_turns,
            turns_remaining=settings.max_student_turns - session.student_turns,
            status=session.status,
            difficulty=session.difficulty,  # type: ignore[arg-type]
            hints_enabled=hints_enabled(session.difficulty),
            monitor=await manager.monitor(session.id),
            actions=action_options(session.scenario),
            guideline=session.guideline.view() if session.guideline else None,
        )

    @app.post(
        "/session/{session_id}/message", response_model=MessageResponse, dependencies=protected_session
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
            monitor=outcome.monitor,
            hint=outcome.hint,
        )

    @app.post(
        "/session/{session_id}/action",
        response_model=ActionResponse,
        dependencies=protected_session,
    )
    async def clinical_action(
        session_id: str,
        payload: ActionRequest,
        request: Request,
    ) -> ActionResponse:
        manager: SessionManager = request.app.state.manager
        outcome = await manager.action(session_id, payload.action_id)
        return ActionResponse(
            session_id=outcome.session_id,
            action_id=outcome.action_id,
            action_label=outcome.action_label,
            result=outcome.result,
            warning=outcome.warning,
            monitor=outcome.monitor,
            status=outcome.status,  # type: ignore[arg-type]
            student_turns=outcome.student_turns,
            turns_remaining=outcome.turns_remaining,
            actions_taken=outcome.actions_taken,
        )

    @app.post(
        "/session/{session_id}/hint",
        response_model=HintResponse,
        dependencies=protected_session,
    )
    async def supervisor_hint(
        session_id: str, request: Request, payload: HintRequest | None = None
    ) -> HintResponse:
        manager: SessionManager = request.app.state.manager
        outcome = await manager.hint(session_id, payload.question if payload else "")
        return HintResponse(
            session_id=outcome.session_id,
            hint=outcome.hint,
            question=outcome.question,
            step_ids=outcome.step_ids,
            hints_used=outcome.hints_used,
            hint_penalty=outcome.hint_penalty,
            hints_remaining=outcome.hints_remaining,
            monitor=outcome.monitor,
        )

    @app.get(
        "/session/{session_id}/monitor",
        response_model=MonitorView,
        dependencies=protected_session,
    )
    async def get_monitor(session_id: str, request: Request) -> MonitorView:
        manager: SessionManager = request.app.state.manager
        return await manager.monitor(session_id)

    @app.post("/session/{session_id}/end", response_model=EndResponse, dependencies=protected_session)
    async def end_session(session_id: str, request: Request) -> EndResponse:
        manager: SessionManager = request.app.state.manager
        evaluation = await manager.end(session_id)
        return EndResponse(session_id=session_id, evaluation=evaluation)

    @app.get("/session/{session_id}", response_model=SessionSnapshot, dependencies=protected_session)
    async def get_session(session_id: str, request: Request) -> SessionSnapshot:
        manager: SessionManager = request.app.state.manager
        return await manager.snapshot(session_id)

    app.include_router(whatsapp_router)
    return app


app = create_app()
