from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "app" / "static" / "index.html"

VOID = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


class _Element:
    def __init__(self, tag: str, attrs: dict[str, str], parent: _Element | None) -> None:
        self.tag = tag
        self.attrs = attrs
        self.parent = parent
        self.children: list[_Element] = []
        self.content: list[str | _Element] = []

    @property
    def classes(self) -> set[str]:
        return set((self.attrs.get("class") or "").split())

    def text(self) -> str:
        pieces = [item if isinstance(item, str) else item.text() for item in self.content]
        return " ".join(" ".join(pieces).split())

    def descendants(self) -> list[_Element]:
        found: list[_Element] = []
        for child in self.children:
            found.append(child)
            found.extend(child.descendants())
        return found

    def contains(self, node: _Element) -> bool:
        return any(node is self or node is item for item in self.descendants())


class _Document(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Element("document", {}, None)
        self.stack = [self.root]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Element(tag, {k: v or "" for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        self.stack[-1].content.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        node = _Element(tag, {k: v or "" for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        self.stack[-1].content.append(node)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        self.stack[-1].content.append(data)


def read_ui() -> str:
    return UI.read_text(encoding="utf-8")


def parse_ui() -> _Document:
    document = _Document()
    document.feed(read_ui())
    document.close()
    return document


def find_by_id(document: _Document, element_id: str) -> list[_Element]:
    return [el for el in document.root.descendants() if el.attrs.get("id") == element_id]


def find_all(document: _Document, tag: str, class_name: str | None = None) -> list[_Element]:
    return [
        el
        for el in document.root.descendants()
        if el.tag == tag and (class_name is None or class_name in el.classes)
    ]


def test_header_holds_brand_start_bar_then_badge() -> None:
    document = parse_ui()
    topbar = find_all(document, "header", "topbar")[0]
    assert [child.tag for child in topbar.children] == ["div", "form", "div"]
    brand, start_bar, badge = topbar.children
    assert "brand" in brand.classes
    assert start_bar.attrs.get("id") == "startForm"
    assert "start-bar" in start_bar.classes
    assert badge.attrs.get("class") == "badge"
    order = [
        el.attrs["id"]
        for el in start_bar.descendants()
        if el.attrs.get("id") in {"difficultySelect", "studentName", "startButton"}
    ]
    assert order == ["difficultySelect", "studentName", "startButton"]


def test_moved_controls_appear_once_and_leave_the_setup_panel() -> None:
    html = read_ui()
    document = parse_ui()
    setup_panel = find_all(document, "aside", "setup-panel")[0]
    for element_id in ("difficultySelect", "studentName", "startButton"):
        assert html.count(f'id="{element_id}"') == 1
        node = find_by_id(document, element_id)[0]
        assert not setup_panel.contains(node)
    for element_id in ("scenarioSelect", "healthButton"):
        assert html.count(f'id="{element_id}"') == 1
    assert setup_panel.contains(find_by_id(document, "healthButton")[0])
    assert setup_panel.contains(find_by_id(document, "scenarioSelect")[0])


def test_scenario_stays_in_the_form_through_its_form_attribute() -> None:
    document = parse_ui()
    scenario = find_by_id(document, "scenarioSelect")[0]
    start_form = find_by_id(document, "startForm")[0]
    assert scenario.attrs.get("form") == "startForm"
    assert scenario.attrs.get("required") is not None
    for element_id in ("difficultySelect", "studentName", "startButton"):
        assert start_form.contains(find_by_id(document, element_id)[0])


def test_moved_controls_keep_their_labels_and_attributes() -> None:
    document = parse_ui()
    labels = {
        label.attrs["for"]: label
        for label in find_all(document, "label")
        if label.attrs.get("for")
    }
    assert "Learning mode" in labels["difficultySelect"].text()
    name_label = labels["studentName"].text()
    assert "Your name" in name_label
    assert "optional" in name_label
    difficulty = find_by_id(document, "difficultySelect")[0]
    assert [option.text() for option in difficulty.children if option.tag == "option"] == [
        "Beginner - guided with automatic hints",
        "Intermediate - hints on request",
        "Advanced - no hints, faster deterioration",
    ]
    name = find_by_id(document, "studentName")[0]
    assert name.attrs.get("maxlength") == "80"
    assert name.attrs.get("placeholder") == "Student name"


def test_start_bar_responds_at_920_and_560_with_tap_targets() -> None:
    html = read_ui()
    tablet = html.split("@media (max-width: 920px)", 1)[1].split("@media", 1)[0]
    phone = html.split("@media (max-width: 560px)", 1)[1].split("</style>", 1)[0]
    assert ".start-bar" in tablet and "flex-wrap: wrap" in tablet
    assert ".start-bar" in phone and "flex-direction: column" in phone
    assert "flex: 0 1 auto" in phone, "reset the 920px flex-basis inside the column layout"
    assert "min-height: 44px" in phone


def test_script_still_finds_every_element_by_id() -> None:
    html = read_ui()
    document = parse_ui()
    for element_id in set(re.findall(r'getElementById\("([^"]+)"\)', html)):
        assert find_by_id(document, element_id), element_id


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
