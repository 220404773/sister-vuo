from __future__ import annotations

import asyncio
import hashlib
import hmac
import json

import httpx
from fastapi.testclient import TestClient

from app.config import Settings
from app.whatsapp import WhatsAppClient, split_text
from tests.conftest import FakeWhatsApp


def make_payload(phone: str, text: str, message_id: str, message_type: str = "text") -> dict:
    message = {
        "from": phone,
        "id": message_id,
        "timestamp": "1700000000",
        "type": message_type,
    }
    if message_type == "text":
        message["text"] = {"body": text}
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "business-account",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "27000000000",
                                "phone_number_id": "phone-id",
                            },
                            "messages": [message],
                        },
                    }
                ],
            }
        ],
    }


def post_payload(
    client: TestClient,
    settings: Settings,
    payload: dict,
    signature: str | None = None,
) -> object:
    body = json.dumps(payload).encode("utf-8")
    if signature is None:
        signature = "sha256=" + hmac.new(
            settings.whatsapp_app_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
    return client.post(
        "/webhook/whatsapp",
        content=body,
        headers={
            "Content-Type": "application/json",
            "X-Hub-Signature-256": signature,
        },
    )


def test_webhook_verification(client: TestClient) -> None:
    response = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "verify-token",
            "hub.challenge": "challenge-value",
        },
    )
    assert response.status_code == 200
    assert response.text == "challenge-value"

    rejected = client.get(
        "/webhook/whatsapp",
        params={
            "hub.mode": "subscribe",
            "hub.verify_token": "wrong-token",
            "hub.challenge": "challenge-value",
        },
    )
    assert rejected.status_code == 403


def test_invalid_signature_is_rejected(
    client: TestClient,
    settings: Settings,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    payload = make_payload("27820000000", "START", "wamid.bad-signature")
    response = post_payload(client, settings, payload, signature="sha256=deadbeef")
    assert response.status_code == 403
    assert fake_whatsapp.sent == []


def test_start_message_end_over_whatsapp(
    client: TestClient,
    settings: Settings,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    phone = "27820000000"
    started = post_payload(
        client,
        settings,
        make_payload(phone, "START 2", "wamid.start"),
    )
    assert started.status_code == 200
    assert len(fake_whatsapp.sent) == 2
    assert all(recipient == phone for recipient, _ in fake_whatsapp.sent)
    assert "Paediatric dehydration" in fake_whatsapp.sent[0][1]
    assert "Tshepo" in fake_whatsapp.sent[1][1]

    logs = list(settings.log_dir.glob("*.jsonl"))
    assert len(logs) == 1
    log_text = logs[0].read_text(encoding="utf-8")
    assert phone not in log_text
    assert "***0000" in log_text

    asked = post_payload(
        client,
        settings,
        make_payload(phone, "How long has he been unwell?", "wamid.ask"),
    )
    assert asked.status_code == 200
    assert len(fake_whatsapp.sent) == 3
    assert "pain" in fake_whatsapp.sent[2][1].lower() or "worse" in fake_whatsapp.sent[2][1].lower()

    ended = post_payload(client, settings, make_payload(phone, "END", "wamid.end"))
    assert ended.status_code == 200
    assert len(fake_whatsapp.sent) == 4
    assert "Overall score: 4.0/5" in fake_whatsapp.sent[3][1]
    assert "Missed red flags" in fake_whatsapp.sent[3][1]


def test_duplicate_webhook_delivery_is_ignored(
    client: TestClient,
    settings: Settings,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    payload = make_payload("27821111111", "START", "wamid.duplicate")
    first = post_payload(client, settings, payload)
    second = post_payload(client, settings, payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(fake_whatsapp.sent) == 2


def test_start_while_active_does_not_create_second_session(
    client: TestClient,
    settings: Settings,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    phone = "27822222222"
    post_payload(client, settings, make_payload(phone, "START", "wamid.first-start"))
    post_payload(client, settings, make_payload(phone, "START 3", "wamid.second-start"))
    assert len(fake_whatsapp.sent) == 3
    assert "already have an active session" in fake_whatsapp.sent[2][1]


def test_non_text_message_gets_guidance(
    client: TestClient,
    settings: Settings,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    response = post_payload(
        client,
        settings,
        make_payload("27823333333", "", "wamid.image", message_type="image"),
    )
    assert response.status_code == 200
    assert len(fake_whatsapp.sent) == 1
    assert "text messages only" in fake_whatsapp.sent[0][1]


def test_split_text_respects_limit() -> None:
    text = "word " * 400
    chunks = split_text(text, limit=100)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_whatsapp_client_sends_graph_payload(settings: Settings) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.out"}]})

    async def run() -> None:
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        whatsapp = WhatsAppClient(settings, client=client)
        await whatsapp.send_text("27820000000", "Hello patient")
        await whatsapp.close()

    asyncio.run(run())
    assert len(requests) == 1
    request = requests[0]
    expected_url = (
        f"https://graph.facebook.com/{settings.whatsapp_api_version}/"
        f"{settings.whatsapp_phone_number_id}/messages"
    )
    assert str(request.url) == expected_url
    assert request.headers["Authorization"] == "Bearer access-token"
    payload = json.loads(request.content)
    assert payload["to"] == "27820000000"
    assert payload["type"] == "text"
    assert payload["text"]["body"] == "Hello patient"
