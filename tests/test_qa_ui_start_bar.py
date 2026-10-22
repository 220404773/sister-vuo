"""QA tests: move the session start controls into the page header.

Requirements pack FR-01..FR-06, acceptance criteria AC-01..AC-06, plus the
POST /session/start contract the moved controls feed. Tests only: no product
code is edited here, and nothing is skipped or weakened.
"""

from __future__ import annotations

import re
from dataclasses import replace

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import FakeLLM, FakeWhatsApp
from .test_frontend import find_all, find_by_id, parse_ui, read_ui

MOVED = ("difficultySelect", "studentName", "startButton")
DIFFICULTIES = ["Beginner", "Intermediate", "Advanced"]


def _media_block(css: str, query: str) -> str:
    """Return the body of the first @media (query) block, braces balanced."""
    start = css.index(f"@media ({query})")
    opening = css.index("{", start)
    depth = 0
    for index in range(opening, len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[opening + 1 : index]
    raise AssertionError(f"unterminated @media ({query}) block")


def _declarations(block: str, selector: str) -> str:
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", block):
        selectors = {part.strip() for part in match.group(1).split(",")}
        if selector in selectors:
            return match.group(2)
    raise AssertionError(f"no CSS rule for {selector}")


def _js_function(source: str, name: str) -> str:
    start = source.index(f"function {name}(")
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function {name}")


# --- AC-01: header child order and start bar control order -------------------


def test_ac01_topbar_holds_brand_then_start_bar_then_badge() -> None:
    document = parse_ui()
    topbar = find_all(document, "header", "topbar")[0]

    assert len(topbar.children) == 3
    brand, start_bar, badge = topbar.children
    assert brand.tag == "div" and "brand" in brand.classes
    assert "Sister Vuo" in brand.text()
    assert start_bar.tag == "form"
    assert start_bar.attrs.get("id") == "startForm"
    assert "start-bar" in start_bar.classes
    assert badge.tag == "div" and "badge" in badge.classes
    assert badge.text() == "Education simulation"

    controls = [
        node
        for node in start_bar.descendants()
        if node.attrs.get("id") in set(MOVED)
    ]
    assert [node.attrs["id"] for node in controls] == list(MOVED)


# --- AC-02: one of each id, health stays left, moved controls leave the panel -


def test_ac02_ids_are_unique_and_only_health_and_scenario_stay_left() -> None:
    html = read_ui()
    document = parse_ui()
    setup_panel = find_all(document, "aside", "setup-panel")[0]

    for element_id in (*MOVED, "scenarioSelect", "healthButton"):
        nodes = find_by_id(document, element_id)
        assert len(nodes) == 1, element_id
        assert html.count(f'id="{element_id}"') == 1, element_id

    for element_id in MOVED:
        assert not setup_panel.contains(find_by_id(document, element_id)[0])
    assert setup_panel.contains(find_by_id(document, "healthButton")[0])
    assert not setup_panel.contains(find_by_id(document, "scenarioSelect")[0])


# --- AC-03: the scenario joins the header form through form= ----------------


def test_ac03_scenario_validates_through_the_header_form() -> None:
    document = parse_ui()
    scenario = find_by_id(document, "scenarioSelect")[0]
    start_form = find_by_id(document, "startForm")[0]

    assert scenario.tag == "select"
    assert "required" in scenario.attrs
    assert "form" not in start_form.attrs
    assert start_form.contains(scenario)
    for element_id in MOVED:
        assert start_form.contains(find_by_id(document, element_id)[0])


# --- AC-04 / FR-02: labels, options, limits and placeholder are unchanged ----


def test_ac04_moved_controls_keep_labels_options_and_input_attributes() -> None:
    document = parse_ui()
    labels = {}
    for label in find_all(document, "label"):
        target = label.attrs.get("for")
        if target in MOVED:
            labels[target] = label

    assert set(labels) == {"difficultySelect", "studentName"}
    assert labels["difficultySelect"].text() == "Learning mode"
    assert "Your name" in labels["studentName"].text()
    assert "optional" in labels["studentName"].text()

    difficulty = find_by_id(document, "difficultySelect")[0]
    options = [node for node in difficulty.children if node.tag == "option"]
    assert [option.attrs.get("value") for option in options] == DIFFICULTIES
    assert [option.text() for option in options] == [
        "Beginner - guided with automatic hints",
        "Intermediate - hints on request",
        "Advanced - no hints, faster deterioration",
    ]

    name = find_by_id(document, "studentName")[0]
    assert name.attrs.get("maxlength") == "80"
    assert name.attrs.get("placeholder") == "Student name"
    assert find_by_id(document, "startButton")[0].text() == "Start session"


# --- AC-05: wrap at 920px, stack and 44px targets at 560px ------------------


def test_ac05_media_queries_wrap_then_stack_with_44px_targets() -> None:
    css = read_ui()
    tablet = _media_block(css, "max-width: 920px")
    phone = _media_block(css, "max-width: 560px")

    assert "flex-wrap: wrap" in _declarations(tablet, ".start-bar")
    assert "flex-direction: column" in _declarations(phone, ".start-bar")
    for selector in (
        ".start-bar #difficultySelect",
        ".start-bar #studentName",
        ".start-bar #startButton",
    ):
        assert "min-height: 44px" in _declarations(phone, selector)


# --- AC-06: the script's element lookups and API headers are unchanged ------


def test_ac06_every_script_element_lookup_still_exists() -> None:
    html = read_ui()
    document = parse_ui()
    ids_in_markup = {
        node.attrs["id"] for node in document.root.descendants() if "id" in node.attrs
    }
    referenced = set(re.findall(r'getElementById\("([^"]+)"\)', html))
    assert referenced, "the script should look elements up by id"
    assert referenced <= ids_in_markup, sorted(referenced - ids_in_markup)

    els_block = re.search(r"const els = \{(.*?)\n    \};", html, re.DOTALL)
    assert els_block, "the els lookup table is missing"
    defined = set(re.findall(r"^\s*(\w+):", els_block.group(1), re.MULTILINE))
    used = set(re.findall(r"\bels\.(\w+)", html))
    assert used <= defined, sorted(used - defined)


def test_ac06_fetch_still_sends_api_key_and_session_token() -> None:
    html = read_ui()
    assert len(re.findall(r"\bfetch\s*\(", html)) == 1
    assert 'headers["X-API-Key"] = key;' in html
    assert 'headers["X-Session-Token"] = state.sessionToken;' in html
    assert 'path.startsWith("/session/")' in html
    assert 'path !== "/session/start"' in html
    assert "fetch(path, { ...options, headers })" in html


# --- FR-05 / FR-06: preserved behaviour and existing styling ----------------


def test_fr05_start_stays_hidden_and_the_three_controls_follow_state() -> None:
    html = read_ui()
    update = _js_function(html, "updateControls")
    assert "els.startButton.hidden = hasSession;" in update
    assert "els.startButton.disabled = state.busy || !els.scenarioSelect.value;" in update
    for element_id in ("scenarioSelect", "difficultySelect", "studentName"):
        assert f"els.{element_id}.disabled = hasSession || state.busy;" in update
    assert 'els.studentName.value = "";' in _js_function(html, "newSession")


def test_fr06_start_bar_adds_no_new_colours_or_fonts_and_keeps_focus() -> None:
    css = read_ui()
    base = _declarations(css.split("@media", 1)[0], ".start-bar")
    for banned in ("color", "background", "font"):
        assert banned not in base
    assert "select:focus" in css
    assert "input:focus" in css


# --- POST /session/start contract fed by the moved controls -----------------


def test_contract_ui_options_are_accepted_by_the_start_endpoint(
    client: TestClient,
) -> None:
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert page.text.replace("\r\n", "\n") == read_ui()

    scenario_id = client.get("/scenarios").json()[0]["id"]
    long_name = "A" * 80
    for difficulty in DIFFICULTIES:
        response = client.post(
            "/session/start",
            json={
                "scenario_id": scenario_id,
                "student_name": long_name,
                "difficulty": difficulty,
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["difficulty"] == difficulty
        assert body["student_turns"] == 0
        assert {
            "session_id",
            "session_token",
            "scenario",
            "opening_message",
            "monitor",
            "actions",
            "guideline",
        } <= set(body)


def test_contract_start_rejects_invalid_input_with_the_spec_shapes(
    client: TestClient,
) -> None:
    for payload in (
        {"scenario_id": ""},
        {"scenario_id": "x" * 81},
        {"scenario_id": "abdominal_pain", "student_name": "x" * 81},
        {"scenario_id": "abdominal_pain", "difficulty": "Expert"},
    ):
        response = client.post("/session/start", json=payload)
        assert response.status_code == 422, (payload, response.text)
        assert "detail" in response.json()

    unknown = client.post("/session/start", json={"scenario_id": "not-a-scenario"})
    assert unknown.status_code == 404
    assert set(unknown.json()) == {"detail"}


def test_contract_start_requires_key_and_each_session_its_token(
    settings: Settings,
) -> None:
    app = create_app(
        settings_obj=replace(settings, app_api_key="qa-start-key"),
        llm_service=FakeLLM(),
        whatsapp_client=FakeWhatsApp(),
    )
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        missing_key = client.post("/session/start", json={"scenario_id": "chest_pain"})
        assert missing_key.status_code == 401

        started = client.post(
            "/session/start",
            json={"scenario_id": "chest_pain"},
            headers={"X-API-Key": "qa-start-key"},
        )
        assert started.status_code == 200, started.text
        data = started.json()
        path = f"/session/{data['session_id']}"

        assert client.get(path, headers={"X-API-Key": "qa-start-key"}).status_code == 404
        assert (
            client.get(path, headers={"X-Session-Token": data["session_token"]}).status_code
            == 401
        )
        allowed = client.get(
            path,
            headers={
                "X-API-Key": "qa-start-key",
                "X-Session-Token": data["session_token"],
            },
        )
        assert allowed.status_code == 200


# --- Round-2 regressions for the build round 1 review findings -------------


def test_round2_560px_resets_the_inherited_start_bar_flex_basis() -> None:
    css = read_ui()
    tablet = _declarations(_media_block(css, "max-width: 920px"), ".start-bar")
    phone = _declarations(_media_block(css, "max-width: 560px"), ".start-bar")

    assert "flex-basis: 100%" in tablet
    assert "flex-wrap: wrap" in tablet
    assert "flex-direction: column" in phone
    assert "flex: 0 1 auto" in phone or "flex-basis: auto" in phone, (
        "the 560px column layout must reset the 920px flex-basis"
    )


def test_round2_560px_controls_stack_full_width_with_44px_targets() -> None:
    css = read_ui()
    phone = _media_block(css, "max-width: 560px")

    assert "width: 100%" in _declarations(phone, ".start-bar .start-field")
    for selector in (
        ".start-bar #difficultySelect",
        ".start-bar #studentName",
        ".start-bar #startButton",
    ):
        declarations = _declarations(phone, selector)
        assert "width: 100%" in declarations
        assert "min-width: 0" in declarations
        assert "min-height: 44px" in declarations


def test_round2_the_start_handler_reads_the_scenario_select() -> None:
    html = read_ui()

    assert 'els.startForm.addEventListener("submit", startSession);' in html
    start = _js_function(html, "startSession")
    assert "els.scenarioSelect.value" in start
    assert "FormData" not in html


# --- FR-04 / FR-05: nothing else moved and no inline handlers -------------


def test_fr04_connection_end_and_panels_stay_in_place() -> None:
    document = parse_ui()
    setup_panel = find_all(document, "aside", "setup-panel")[0]
    clinical_panel = find_all(document, "aside", "clinical-panel")[0]

    for element_id in (
        "keyForm",
        "apiKeyInput",
        "scenarioMeta",
        "sessionInfo",
        "endButton",
        "healthButton",
        "healthResult",
        "historyList",
        "clearHistory",
    ):
        node = find_by_id(document, element_id)[0]
        assert setup_panel.contains(node), element_id

    for element_id in (
        "monitorStatus",
        "vitalsGrid",
        "actionList",
        "hintButton",
        "hintAsk",
    ):
        node = find_by_id(document, element_id)[0]
        assert clinical_panel.contains(node), element_id


def test_fr05_moved_controls_have_no_inline_event_handlers() -> None:
    html = read_ui()
    assert not re.search(r"\bon(?:click|change|submit|input|keydown|keyup)\s*=", html)
    document = parse_ui()
    start_form = find_by_id(document, "startForm")[0]
    start_button = find_by_id(document, "startButton")[0]
    assert start_button.attrs.get("type") == "submit"
    assert start_form.contains(start_button)
