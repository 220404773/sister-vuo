from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "static" / "index.html"


def read_ui() -> str:
    return UI.read_text(encoding="utf-8")


def test_ui_keeps_the_session_token_in_memory_only() -> None:
    html = read_ui()
    assert "sessionToken: null" in html
    assert "state.sessionToken = response.session_token" in html
    assert "state.sessionToken = null" in html
    for line in html.splitlines():
        if "localStorage" in line:
            assert "sessionToken" not in line
            assert "session_token" not in line


def test_ui_sends_the_token_on_session_requests_but_not_on_start() -> None:
    html = read_ui()
    assert 'headers["X-Session-Token"] = state.sessionToken' in html
    assert 'path.startsWith("/session/")' in html
    assert 'path !== "/session/start"' in html


def test_ui_renders_external_text_as_text_not_html() -> None:
    html = read_ui()
    assert "bubble.textContent = text;" in html
    assert "item.textContent = alert;" in html
    assert "els.errorText.textContent = message;" in html


def test_errors_and_hints_are_announced_to_assistive_technology() -> None:
    html = read_ui()
    assert 'id="errorBanner" class="error-banner" role="alert"' in html
    assert 'id="hintCard" class="hint-card" role="status"' in html


def test_privacy_pages_document_automatic_retention() -> None:
    privacy = (ROOT / "docs" / "privacy.html").read_text(encoding="utf-8")
    deletion = (ROOT / "docs" / "data-deletion.html").read_text(encoding="utf-8")
    assert "deleted automatically" in privacy
    assert "retention period" in privacy
    assert "Automatic deletion" in deletion
    assert "on request" in deletion


def test_ui_is_still_served_with_the_session_contract(
    client: TestClient, settings: Settings
) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert "Sister Vuo" in page.text

    started = client.post(
        "/session/start", json={"scenario_id": "abdominal_pain"}
    ).json()
    headers = {"X-Session-Token": started["session_token"]}
    assert client.get(f"/session/{started['session_id']}", headers=headers).status_code == 200
    assert client.get(f"/session/{started['session_id']}").status_code == 404
