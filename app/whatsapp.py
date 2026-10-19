from __future__ import annotations

import hashlib
import hmac
import json
import logging
import secrets
import time
from collections import OrderedDict
from typing import Any

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from .config import Settings
from .models import Evaluation
from .sessions import (
    ActiveSessionExists,
    ScenarioNotFound,
    SessionCapacityReached,
    SessionCompleted,
    SessionManager,
    TurnLimitReached,
    mask_phone,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["whatsapp"])

DOMAIN_LABELS = {
    "history_taking": "History taking",
    "clinical_reasoning": "Clinical reasoning",
    "communication_empathy": "Communication and empathy",
    "language_accessibility": "Language accessibility",
    "safety_referral": "Safety and referral",
}


class WhatsAppConfigurationError(RuntimeError):
    pass


class WhatsAppSendError(RuntimeError):
    pass


def split_text(text: str, limit: int = 3500) -> list[str]:
    remaining = text.strip()
    chunks: list[str] = []
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = remaining.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


class WhatsAppClient:
    def __init__(
        self,
        settings: Settings,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self._client = client

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.whatsapp_timeout_seconds)
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_text(self, to: str, body: str) -> None:
        if not self.settings.whatsapp_access_token or not self.settings.whatsapp_phone_number_id:
            raise WhatsAppConfigurationError("WhatsApp sending is not configured.")

        url = (
            f"https://graph.facebook.com/{self.settings.whatsapp_api_version}/"
            f"{self.settings.whatsapp_phone_number_id}/messages"
        )
        headers = {
            "Authorization": f"Bearer {self.settings.whatsapp_access_token}",
            "Content-Type": "application/json",
        }
        client = self._get_client()

        for chunk in split_text(body):
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {"preview_url": False, "body": chunk},
            }
            try:
                response = await client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.RequestError) as exc:
                logger.warning("WhatsApp send failed for %s: network error", mask_phone(to))
                raise WhatsAppSendError("Could not reach WhatsApp.") from exc
            if response.status_code >= 400:
                logger.warning(
                    "WhatsApp send failed for %s with HTTP %s",
                    mask_phone(to),
                    response.status_code,
                )
                raise WhatsAppSendError(
                    f"WhatsApp rejected the message (HTTP {response.status_code})."
                )


class MessageDeduper:
    def __init__(self, ttl_seconds: int = 600, max_items: int = 2000) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_items = max_items
        self._seen: OrderedDict[str, float] = OrderedDict()

    def seen(self, message_id: str) -> bool:
        now = time.time()
        # Oldest entries come first, so expired ones can be dropped from the front.
        while self._seen and now - next(iter(self._seen.values())) >= self.ttl_seconds:
            self._seen.popitem(last=False)
        if message_id in self._seen:
            return True
        self._seen[message_id] = now
        while len(self._seen) > self.max_items:
            self._seen.popitem(last=False)
        return False


def _format_menu(manager: SessionManager) -> str:
    summaries = manager.scenarios.list_summaries()
    lines = [
        "Sister Vuo - AI nursing patient simulator",
        "",
        "Choose a scenario:",
    ]
    for index, summary in enumerate(summaries, start=1):
        lines.append(f"{index}. {summary.title} ({summary.difficulty})")
    lines.extend(
        [
            "",
            f"Reply START for scenario 1, or START 1 to {len(summaries)} for a specific scenario.",
            "Reply HELP to see this menu again.",
        ]
    )
    return "\n".join(lines)


def _format_evaluation(evaluation: Evaluation) -> str:
    lines = [
        "Sister Vuo - session feedback",
        f"Overall score: {evaluation.overall_score:.1f}/5",
        "",
    ]
    for key, label in DOMAIN_LABELS.items():
        score = getattr(evaluation.domain_scores, key)
        lines.append(f"{label}: {score}/5")
    lines.append("")
    for title, items, empty in (
        ("Strengths", evaluation.strengths, "None recorded."),
        ("Missed red flags", evaluation.missed_red_flags, "None identified."),
        ("Unsafe actions", evaluation.unsafe_actions, "None identified."),
        ("Recommended learning", evaluation.recommended_learning, "None recorded."),
    ):
        lines.append(f"{title}:")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append(f"- {empty}")
        lines.append("")
    lines.extend(
        [
            "Feedback:",
            evaluation.feedback_to_student,
            "",
            "This is simulation feedback for education only, not clinical advice.",
        ]
    )
    return "\n".join(lines)


def _signature_is_valid(secret: str, body: bytes, header: str | None) -> bool:
    if not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


async def _safe_send(client: Any, to: str, body: str) -> None:
    try:
        await client.send_text(to, body)
    except Exception:
        logger.exception("Could not send WhatsApp message to %s", mask_phone(to))


async def _send_unsupported(client: Any, sender: str) -> None:
    await _safe_send(client, sender, "Please send text messages only.")


async def handle_whatsapp_message(
    manager: SessionManager,
    client: Any,
    sender: str,
    text: str,
) -> None:
    command = text.strip()
    upper = command.upper()
    active = manager.get_active_by_phone(sender)

    if upper in {"HELP", "MENU", "SCENARIOS"}:
        await client.send_text(sender, _format_menu(manager))
        return

    if upper == "START" or upper.startswith("START "):
        if active is not None:
            await client.send_text(
                sender,
                "You already have an active session. Reply END to finish it first, "
                "or continue with your questions.",
            )
            return
        summaries = manager.scenarios.list_summaries()
        argument = command[5:].strip()
        scenario_id = summaries[0].id if not argument else manager.scenarios.resolve(argument)
        if scenario_id is None:
            await client.send_text(
                sender,
                f"I do not know that scenario.\n\n{_format_menu(manager)}",
            )
            return
        try:
            session = await manager.start(scenario_id, channel="whatsapp", phone=sender)
        except ActiveSessionExists:
            await client.send_text(
                sender,
                "You already have an active session. Reply END to finish it first.",
            )
            return
        except ScenarioNotFound:
            await client.send_text(sender, _format_menu(manager))
            return
        except SessionCapacityReached:
            await client.send_text(
                sender, "The simulator is busy right now. Please try START again later."
            )
            return
        summary = manager.scenarios.summary(str(session.scenario["id"]))
        await client.send_text(
            sender,
            (
                f"Simulation started: {summary.title}.\n\n"
                "You are the student nurse. Take a history, identify red flags, and explain "
                "your safety plan. Reply END to finish now; the session ends automatically "
                "after 12 messages.\n\n"
                "This is a fictional simulation. Do not send real patient data."
            ),
        )
        await client.send_text(sender, session.transcript[0].content)
        return

    if upper == "END":
        if active is None:
            await client.send_text(sender, "There is no active session. Reply START to begin.")
            return
        try:
            evaluation = await manager.end(active.id)
        except Exception:
            logger.exception("WhatsApp evaluation failed for %s", mask_phone(sender))
            await client.send_text(
                sender,
                "I could not generate feedback right now. Reply END to try again.",
            )
            return
        await client.send_text(sender, _format_evaluation(evaluation))
        return

    if active is None:
        await client.send_text(sender, f"There is no active session.\n\n{_format_menu(manager)}")
        return

    if len(command) > 2000:
        await client.send_text(sender, "Please keep messages under 2000 characters.")
        return

    try:
        outcome = await manager.message(active.id, command)
    except TurnLimitReached:
        await client.send_text(
            sender,
            "This session has reached its 12-message limit. Reply END to receive feedback.",
        )
        return
    except SessionCompleted:
        await client.send_text(sender, "This session has ended. Reply START to begin again.")
        return

    await client.send_text(sender, outcome.reply)
    if outcome.evaluation is not None:
        await client.send_text(
            sender,
            "Session complete after 12 student messages.\n\n"
            f"{_format_evaluation(outcome.evaluation)}",
        )
    elif outcome.evaluation_error:
        await client.send_text(
            sender,
            "Session complete. I could not generate feedback right now. "
            "Reply END to try again.",
        )


async def _process_whatsapp_message(
    manager: SessionManager,
    client: Any,
    sender: str,
    text: str,
) -> None:
    try:
        await handle_whatsapp_message(manager, client, sender, text)
    except Exception:
        logger.exception("WhatsApp message processing failed for %s", mask_phone(sender))
        await _safe_send(
            client,
            sender,
            "Sorry, something went wrong while handling that message. "
            "Reply END to finish the session or START to begin again.",
        )


@router.get("/webhook/whatsapp", response_class=PlainTextResponse)
async def verify_whatsapp_webhook(
    request: Request,
    hub_mode: str | None = Query(default=None, alias="hub.mode"),
    hub_verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> PlainTextResponse:
    settings: Settings = request.app.state.settings
    if not settings.whatsapp_configured:
        raise HTTPException(status_code=503, detail="WhatsApp is not configured.")
    token_ok = secrets.compare_digest(
        (hub_verify_token or "").encode("utf-8"),
        settings.whatsapp_verify_token.encode("utf-8"),
    )
    if hub_mode != "subscribe" or not token_ok:
        raise HTTPException(status_code=403, detail="Webhook verification failed.")
    if hub_challenge is None:
        raise HTTPException(status_code=400, detail="Missing verification challenge.")
    return PlainTextResponse(hub_challenge)


@router.post("/webhook/whatsapp")
async def receive_whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    settings: Settings = request.app.state.settings
    if not settings.whatsapp_configured:
        raise HTTPException(status_code=503, detail="WhatsApp is not configured.")

    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")
    if not _signature_is_valid(settings.whatsapp_app_secret, raw_body, signature):
        raise HTTPException(status_code=403, detail="Invalid webhook signature.")

    try:
        payload = json.loads(raw_body or b"{}")
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON payload.") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid webhook payload.")

    manager: SessionManager = request.app.state.manager
    client: Any = request.app.state.whatsapp
    deduper: MessageDeduper = request.app.state.whatsapp_dedupe

    for entry in payload.get("entry", []):
        if not isinstance(entry, dict):
            continue
        for change in entry.get("changes", []):
            if not isinstance(change, dict):
                continue
            value = change.get("value", {})
            if not isinstance(value, dict):
                continue
            for message in value.get("messages", []):
                if not isinstance(message, dict):
                    continue
                sender = str(message.get("from", "")).strip()
                if not sender:
                    continue
                message_id = str(message.get("id", "")).strip()
                if not message_id:
                    message_id = (
                        f"{sender}:{message.get('timestamp', '')}:{message.get('type', '')}"
                    )
                if deduper.seen(message_id):
                    continue
                if message.get("type") != "text":
                    background_tasks.add_task(_send_unsupported, client, sender)
                    continue
                text_payload = message.get("text", {})
                text = (
                    str(text_payload.get("body", "")).strip()
                    if isinstance(text_payload, dict)
                    else ""
                )
                if not text:
                    continue
                background_tasks.add_task(
                    _process_whatsapp_message,
                    manager,
                    client,
                    sender,
                    text,
                )

    return {"status": "ok"}
