"""Static checks for the debrief's all-clear safety card and small polish items."""

import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")


def render_safety_source() -> str:
    start = PAGE.index("function renderSafety(")
    return PAGE[start : PAGE.index("function renderProgress(", start)]


def test_the_safety_heading_is_dynamic_and_is_not_alarming_when_all_clear() -> None:
    assert 'id="debriefSafetyHeading"' in PAGE
    source = render_safety_source()
    assert 'allClear ? "Safety" : "Safety: needs attention"' in source
    assert 'classList.toggle("clear", allClear)' in source


def test_the_all_clear_card_uses_neutral_styles_not_the_danger_colours() -> None:
    clear = re.search(r"\.debrief-safety\.clear\s*\{([^}]*)\}", PAGE)
    clear_heading = re.search(r"\.debrief-safety\.clear h3\s*\{([^}]*)\}", PAGE)
    assert clear and clear_heading
    assert "danger" not in clear.group(1) and "danger" not in clear_heading.group(1)
    assert "--teal-soft" in clear.group(1)


def test_the_programmatic_focus_target_has_no_outline_and_the_guideline_groups_have_spacing() -> None:
    assert re.search(r"#debriefHeading:focus\s*\{\s*outline:\s*none;", PAGE)
    assert re.search(r"\.guideline-group > p \+ h4\s*\{\s*margin-top:\s*14px;", PAGE)


def test_the_clinical_actions_count_is_singular_for_one() -> None:
    assert 'clinical action${count === 1 ? "" : "s"}' in PAGE
    assert "} clinical actions`" not in PAGE
