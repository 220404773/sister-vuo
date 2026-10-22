from __future__ import annotations

import re
from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
from app.guidelines import GuidelineStep
from app.scoring import normalize_protocol_steps
from tests.conftest import FakeLLM, FakeWhatsApp
from tests.test_frontend import find_all, find_by_id, parse_ui, read_ui
from tests.test_guidelines import (
    full_guidelines,
    guideline_client,
    links_only_guidelines,
    start_chest,
    token_headers,
)


def test_ac01_serialized_steps_carry_the_stored_step_text(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    fake_llm.protocol_steps = [
        {"id": "step-2", "status": "partial", "evidence_turn": 3},
    ]
    with guideline_client(
        settings, tmp_path, fake_llm, fake_whatsapp, full_guidelines()
    ) as client:
        started = start_chest(client)
        ended = client.post(
            f"/session/{started['session_id']}/end", headers=token_headers(started)
        )
        assert ended.status_code == 200, ended.text
        evaluation = ended.json()["evaluation"]

        stored = {
            step["id"]: step["text"]
            for step in full_guidelines()["scenarios"]["chest_pain"]["steps"]
        }
        steps = evaluation["protocol_steps"]
        assert steps is not None
        assert len(steps) == 5
        for step in steps:
            assert set(step) == {"id", "status", "evidence_turn", "text"}
            assert step["text"] == stored[step["id"]]
        assert steps[1]["status"] == "partial"
        assert (
            steps[1]["text"]
            == "Give oxygen if the oxygen saturation is below 94 percent."
        )


def test_ac02_unknown_step_ids_are_still_dropped_without_inventing_text() -> None:
    configured = [
        GuidelineStep(id="s1", text="Stored wording one."),
        GuidelineStep(id="s2", text="Stored wording two."),
    ]
    raw = [
        {"id": "s1", "status": "met", "evidence_turn": 2},
        {"id": "not-stored", "status": "met", "evidence_turn": 1},
    ]
    result = normalize_protocol_steps(raw, configured)
    assert [step.id for step in result] == ["s1", "s2"]
    assert [step.text for step in result] == [
        "Stored wording one.",
        "Stored wording two.",
    ]
    assert result[1].status == "missed"


def test_ac03_a_session_without_a_guideline_has_no_protocol_steps(
    client: TestClient,
) -> None:
    started = client.post(
        "/session/start", json={"scenario_id": "abdominal_pain"}
    ).json()
    ended = client.post(
        f"/session/{started['session_id']}/end", headers=token_headers(started)
    )
    assert ended.status_code == 200, ended.text
    assert ended.json()["evaluation"]["protocol_steps"] is None


def test_fr01_links_only_mode_also_has_no_protocol_steps(
    settings: Settings,
    tmp_path: Path,
    fake_llm: FakeLLM,
    fake_whatsapp: FakeWhatsApp,
) -> None:
    with guideline_client(
        settings, tmp_path, fake_llm, fake_whatsapp, links_only_guidelines()
    ) as client:
        started = start_chest(client)
        ended = client.post(
            f"/session/{started['session_id']}/end", headers=token_headers(started)
        )
        assert ended.status_code == 200, ended.text
        assert ended.json()["evaluation"]["protocol_steps"] is None


def test_contract_end_returns_the_spec_evaluation_shape(
    client: TestClient,
) -> None:
    started = client.post(
        "/session/start", json={"scenario_id": "abdominal_pain"}
    ).json()
    ended = client.post(
        f"/session/{started['session_id']}/end", headers=token_headers(started)
    )
    assert ended.status_code == 200, ended.text
    assert set(ended.json()) == {"session_id", "evaluation"}

    evaluation = ended.json()["evaluation"]
    assert set(evaluation) == {
        "overall_score",
        "domain_scores",
        "strengths",
        "missed_red_flags",
        "unsafe_actions",
        "critical_errors",
        "recommended_learning",
        "feedback_to_student",
        "hint_penalty",
        "debrief",
        "protocol_steps",
    }
    assert set(evaluation["domain_scores"]) == {
        "red_flag_recognition",
        "history_completeness",
        "prioritisation",
        "safety",
        "communication_empathy",
        "time_to_escalation",
    }
    assert set(evaluation["debrief"]) == {
        "what_went_well",
        "what_was_missed",
        "critical_errors",
        "ideal_approach",
        "model_answer",
        "reflection_questions",
        "guideline_references",
        "timeline",
        "time_to_escalation_minutes",
        "escalation_target_minutes",
        "actions_taken",
    }


def test_contract_the_debrief_object_requires_the_session_token(
    client: TestClient,
) -> None:
    started = client.post(
        "/session/start", json={"scenario_id": "abdominal_pain"}
    ).json()
    session_id = started["session_id"]

    assert client.get(f"/session/{session_id}").status_code == 404
    assert (
        client.get(
            f"/session/{session_id}", headers={"X-Session-Token": "wrong-token"}
        ).status_code
        == 404
    )
    assert client.post(f"/session/{session_id}/end").status_code == 404
    assert (
        client.post(
            f"/session/{session_id}/end", headers={"X-Session-Token": "wrong-token"}
        ).status_code
        == 404
    )

    ended = client.post(
        f"/session/{session_id}/end", headers=token_headers(started)
    )
    assert ended.status_code == 200, ended.text
    assert ended.json()["evaluation"]["overall_score"] >= 0


DEBRIEF_HEADINGS = [
    "Your debrief",
    "Safety: needs attention",
    "What went well",
    "What to improve",
    "Skills",
    "Guideline check",
    "Go deeper",
]


def test_ac05_the_debrief_sections_appear_in_reading_order() -> None:
    html = read_ui()
    positions = [html.index(heading) for heading in DEBRIEF_HEADINGS]
    assert positions == sorted(positions), DEBRIEF_HEADINGS


def test_ac05_the_old_duplicate_headings_live_only_in_the_safety_block() -> None:
    html = read_ui()
    safety_start = html.index("function renderSafety(")
    progress_start = html.index("function renderProgress(")
    for heading in ("Critical errors", "Unsafe actions", "Red flags missed"):
        assert html.count(heading) == 1, heading
        assert safety_start < html.index(heading) < progress_start, heading


def test_ac06_the_four_disclosures_are_closed_native_details() -> None:
    document = parse_ui()
    details = find_all(document, "details")
    assert len(details) == 4
    for node in details:
        assert "open" not in node.attrs
        assert node.attrs.get("class") == "debrief-details"
        assert next(child.tag for child in node.children) == "summary"

    for element_id in (
        "evaluationPanel",
        "overallScore",
        "newSessionButton",
        "guidelineCard",
    ):
        assert len(find_by_id(document, element_id)) == 1, element_id


def test_ac06_render_evaluation_builds_text_without_inner_html() -> None:
    html = read_ui()
    for function_name in (
        "renderEvaluation",
        "renderSafety",
        "renderProgress",
        "renderSkills",
        "renderGuidelineCard",
        "renderDeeper",
    ):
        start = html.index(f"function {function_name}(")
        following = html.find("\n    function ", start + 1)
        body = html[start : following if following != -1 else len(html)]
        assert "innerHTML" not in body, function_name

    body_start = html.index("function renderEvaluation(")
    body_end = html.index("function readHistory(")
    body = html[body_start:body_end]
    assert "window.matchMedia" in body
    assert "prefers-reduced-motion: reduce" in body
    assert "els.debriefHeading.focus(" in body
    assert not re.search(r"\son[a-z]+\s*=", body)


def test_fr02_header_has_a_focusable_heading_and_subtitle() -> None:
    document = parse_ui()
    heading = find_by_id(document, "debriefHeading")[0]
    assert heading.tag == "h2"
    assert heading.text() == "Your debrief"
    assert heading.attrs.get("tabindex") == "-1"
    assert find_by_id(document, "debriefScenario")[0].attrs.get("class") == "debrief-scenario"
    panel = find_by_id(document, "evaluationPanel")[0]
    eyebrows = [
        node.text()
        for node in panel.descendants()
        if "eyebrow" in node.classes
    ]
    assert "Debrief" in eyebrows


def test_fr03_score_labels_follow_the_rounded_overall_score() -> None:
    html = read_ui()
    body = html[html.index("function renderEvaluation(") : html.index("function readHistory(")]
    labels = re.search(r"const scaleLabels = \[(.*?)\];", body, re.DOTALL)
    assert labels is not None
    assert re.findall(r'"([^"]+)"', labels.group(1)) == [
        "Absent or unsafe",
        "Very poor",
        "Below the expected level",
        "Competent",
        "Strong",
        "Excellent",
    ]
    assert "Math.max(0, Math.min(5, Math.round(overall)))" in body


def test_fr04_safety_block_shows_words_not_only_colour() -> None:
    html = read_ui()
    body = html[html.index("function renderSafety(") : html.index("function renderProgress(")]
    for heading in ("Critical errors", "Unsafe actions", "Red flags missed"):
        assert f'["{heading}"' in body
    assert (
        "No unsafe actions, critical errors or missed red flags were identified."
        in body
    )
    assert "dedupeInto(evaluation.critical_errors" in body
    assert "dedupeInto(evaluation.unsafe_actions" in body
    assert "dedupeInto(evaluation.missed_red_flags" in body
    assert "function dedupeInto(" in html
    assert "item.toLowerCase()" in html

    css = html.split("<style>", 1)[1].split("</style>", 1)[0]
    safety_rule = css.split(".debrief-safety {", 1)[1].split("}", 1)[0]
    assert "var(--danger-bg)" in safety_rule


def test_fr05_missed_items_share_the_safety_deduplication() -> None:
    html = read_ui()
    body = html[html.index("function renderProgress(") : html.index("function renderSkills(")]
    assert "dedupeInto(debrief.what_was_missed, safetySeen)" in body
    assert '["What was missed", missed]' in body
    assert '["Recommended learning", learning]' in body
    assert 'classList.toggle("single", hasWell !== hasImprove)' in body


def test_fr06_each_skill_shows_a_text_value_and_a_decorative_bar() -> None:
    html = read_ui()
    body = html[html.index("function renderSkills(") : html.index("function renderGuidelineCard(")]
    assert 'value.textContent = hasScore ? `${score}/5` : "Not scored";' in body
    assert 'bar.setAttribute("aria-hidden", "true")' in body
    assert 'fill.style.width' in body


def test_fr07_protocol_rows_use_words_and_the_stored_step_text() -> None:
    html = read_ui()
    body = html[
        html.index("function renderGuidelineCard(") : html.index("function renderDeeper(")
    ]
    assert '{ met: "Met", partial: "Partly met", missed: "Missed" }' in body
    assert "`Step ${step.id}`" in body
    assert "`message ${step.evidence_turn}`" in body
    assert "function safeGuidelineUrl(value)" in html
    assert "const url = safeGuidelineUrl(value);" in html
    assert '"Open the guideline"' in body
    assert '"Further reading"' in body
    assert '"Also referenced"' in body
    assert "ref.trim().toLowerCase() !== guidelineName" in body


def test_fr07b_go_deeper_uses_four_labelled_disclosures() -> None:
    html = read_ui()
    for title in (
        "Ideal approach",
        "Model answer",
        "Reflection questions",
        "Clinical timeline",
    ):
        assert f"<summary>{title}</summary>" in html
    assert ".debrief-details summary:focus-visible" in html


def test_ac07_responsive_rules_and_score_labels_exist() -> None:
    html = read_ui()
    tablet = html.split("@media (max-width: 920px)", 1)[1].split("@media", 1)[0]
    phone = html.split("@media (max-width: 560px)", 1)[1].split("</style>", 1)[0]

    assert ".debrief-summary" in tablet
    assert ".debrief-progress" in tablet
    assert "grid-template-columns: minmax(0, 1fr)" in tablet
    assert ".evaluation-panel" in phone
    assert ".protocol-item" in phone and "grid-template-columns: minmax(0, 1fr)" in phone

    css = html.split("<style>", 1)[1].split("</style>", 1)[0]
    for selector, block in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        if not re.search(
            r"debrief|evaluation|protocol|guideline|timeline|eval-", selector
        ):
            continue
        for prop, value in re.findall(
            r"(?<![\w-])(min-width|width)\s*:\s*([^;]+);", block
        ):
            for pixels in re.findall(r"(\d+(?:\.\d+)?)px", value):
                assert float(pixels) <= 320, (selector.strip(), prop, value)

    for label in (
        "Absent or unsafe",
        "Very poor",
        "Below the expected level",
        "Competent",
        "Strong",
        "Excellent",
    ):
        assert label in html
    assert "Not scored" in html
    assert 'bar.setAttribute("aria-hidden", "true")' in html
