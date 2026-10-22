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

from .clinical import (
    ActionAlreadyDone,
    ActionUnavailable,
    UnknownAction,
    action_command_map,
    monitor_text,
)
from .config import Settings
from .guidelines import GuidelineHintsUnavailable
from .models import Evaluation, GuidelineView
from .sessions import (
    ActiveSessionExists,
    HintsDisabled,
    NoHintsRemaining,
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
    "red_flag_recognition": "Red flag recognition",
    "history_completeness": "History completeness",
    "prioritisation": "Prioritisation",
    "safety": "Safety",
    "communication_empathy": "Communication and empathy",
    "time_to_escalation": "Time to escalation",
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


def _guideline_line(guideline: GuidelineView | None) -> str:
    if guideline is None:
        return ""
    return f"\nGuideline: {guideline.name}, {guideline.edition_year}"


def _format_evaluation(
    evaluation: Evaluation, guideline: GuidelineView | None = None
) -> str:
    lines = [
        "Sister Vuo - debrief",
        f"Overall score: {evaluation.overall_score:.1f}/5",
        "",
    ]
    for key, label in DOMAIN_LABELS.items():
        score = getattr(evaluation.domain_scores, key)
        lines.append(f"{label}: {score}/5")
    if evaluation.hint_penalty:
        lines.append(f"Supervisor hint penalty: -{evaluation.hint_penalty:.1f}")
    debrief = evaluation.debrief
    if debrief.time_to_escalation_minutes is not None:
        lines.append(
            f"Time to escalation: {debrief.time_to_escalation_minutes:.0f} min "
            f"(target {debrief.escalation_target_minutes:.0f} min)"
        )
    else:
        lines.append("Time to escalation: never escalated")
    lines.append("")
    for title, items, empty, unsafe in (
        ("What went well", evaluation.strengths, "None recorded.", False),
        ("What was missed", debrief.what_was_missed, "None identified.", False),
        ("Unsafe actions", evaluation.unsafe_actions, "None identified.", True),
        ("Critical errors", evaluation.critical_errors, "None identified.", True),
        ("Recommended learning", evaluation.recommended_learning, "None recorded.", False),
    ):
        if unsafe and not items:
            continue
        lines.append(f"{title}:")
        if items:
            lines.extend(f"- {item}" for item in items)
        else:
            lines.append(f"- {empty}")
        lines.append("")
    if debrief.ideal_approach:
        lines.append("Ideal approach:")
        lines.extend(f"{index}. {item}" for index, item in enumerate(debrief.ideal_approach, 1))
        lines.append("")
    if debrief.model_answer:
        lines.extend(["Model answer:", debrief.model_answer, ""])
    if debrief.reflection_questions:
        lines.append("Reflection questions:")
        lines.extend(f"- {item}" for item in debrief.reflection_questions)
        lines.append("")
    if debrief.guideline_references:
        lines.append("Guidelines:")
        lines.extend(f"- {item}" for item in debrief.guideline_references)
        lines.append("")
    if guideline is not None:
        lines.append(f"Guideline: {guideline.name}, {guideline.edition_year}")
        lines.append(guideline.section_text)
        if guideline.url:
            lines.append(guideline.url)
        if guideline.further_reading:
            lines.append("Further reading:")
            for item in guideline.further_reading:
                suffix = f" - {item.url}" if item.url else ""
                lines.append(f"- {item.title}, {item.edition_year}{suffix}")
        if evaluation.protocol_steps:
            lines.append("Protocol steps:")
            for step in evaluation.protocol_steps:
                lines.append(f"- {step.id}: {step.status}")
        lines.append("")
    lines.extend(
        [
            "Feedback:",
            evaluation.feedback_to_student,
            "",
            "This is a fictional education simulation, not clinical advice.",
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
        guideline_line = _guideline_line(
            session.guideline.view() if session.guideline else None
        )
        await client.send_text(
            sender,
            (
                f"Simulation started: {summary.title}.{guideline_line}\n\n"
                "You are the student nurse. Take a history, identify red flags, and explain "
                "your safety plan. Reply END to finish now; the session ends automatically "
                "after 12 messages.\n\n"
                "Clinical actions: VITALS, ECG, OXYGEN, ASPIRIN, IV, AMBULANCE, DOCTOR, "
                "PREGNANCY TEST, HINT or HINT <question> or STATUS.\n\n"
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
        guideline = active.guideline.view() if active.guideline else None
        await client.send_text(sender, _format_evaluation(evaluation, guideline))
        return

    if active is None:
        await client.send_text(sender, f"There is no active session.\n\n{_format_menu(manager)}")
        return

    if upper == "STATUS":
        await client.send_text(sender, monitor_text(await manager.monitor(active.id)))
        return

    if upper == "HINT" or upper.startswith("HINT "):
        question = command[5:].strip() if upper.startswith("HINT ") else ""
        if len(question) > 2000:
            await client.send_text(sender, "Please keep your hint question under 2000 characters.")
            return
        try:
            hint = await manager.hint(active.id, question)
        except HintsDisabled as exc:
            await client.send_text(sender, str(exc))
            return
        except NoHintsRemaining as exc:
            await client.send_text(sender, str(exc))
            return
        except GuidelineHintsUnavailable as exc:
            await client.send_text(sender, str(exc))
            return
        except SessionCompleted:
            await client.send_text(sender, "This session has ended. Reply START to begin again.")
            return
        await client.send_text(
            sender,
            f"Supervisor hint ({hint.hint_penalty:.1f} point penalty, "
            f"{hint.hints_remaining} left):\n{hint.hint}",
        )
        return

    action_id = action_command_map().get(upper)
    if action_id is not None:
        try:
            result = await manager.action(active.id, action_id)
        except UnknownAction:
            await client.send_text(sender, "That clinical action is not available.")
            return
        except ActionUnavailable as exc:
            await client.send_text(sender, str(exc))
            return
        except ActionAlreadyDone as exc:
            await client.send_text(sender, str(exc))
            return
        except SessionCompleted:
            await client.send_text(sender, "This session has ended. Reply START to begin again.")
            return
        message = f"{result.action_label}: {result.result}"
        if result.warning:
            message += f"\n\nSafety note: {result.warning}"
        message += f"\n\n{monitor_text(result.monitor)}"
        await client.send_text(sender, message)
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
    if outcome.hint:
        await client.send_text(sender, f"Beginner hint:\n{outcome.hint}")
    if outcome.evaluation is not None:
        guideline = active.guideline.view() if active.guideline else None
        await client.send_text(
            sender,
            "Session complete after 12 student messages.\n\n"
            f"{_format_evaluation(outcome.evaluation, guideline)}",
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
    limiter = request.app.state.rate_limiter.whatsapp

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
                if limiter.check(sender) is not None:
                    background_tasks.add_task(
                        _safe_send,
                        client,
                        sender,
                        "Please slow down and try again in a minute.",
                    )
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
