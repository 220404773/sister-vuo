"""Written cases, the Learning tab and integrated questions (/v1). Acceptance AC-01 to AC-08."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.written import (
    ALLOWLIST,
    WrittenCaseRepository,
    find_abbreviations,
    parse_written_evaluation,
    validate_record,
)
from tests.conftest import FakeLLM

EXAMPLE = Path(__file__).resolve().parents[1] / "scenarios" / "written" / "example.json"


class WrittenFakeLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.json_calls: list[list[dict[str, str]]] = []
        self.json_reply = json.dumps({
            "domains": {"recognition": 4, "pathophysiology": 3, "scenario_data": 4,
                        "management": 3, "rationale": 4, "safety": 5},
            "strengths": ["Named the tachycardia."], "missed": [], "unsafe_actions": [],
            "critical_errors": ["None identified"], "feedback": "Good start. Give 900 milligrams of nothing.",
            "row_status": [{"question": "q1", "row_index": 0, "status": "met"},
                           {"question": "q2", "group_index": 0, "row_index": 0, "status": "partial"}],
        })
        self.fail = False

    async def complete_json(self, messages):
        self.json_calls.append(messages)
        if self.fail:
            raise RuntimeError("model down")
        return self.json_reply


def private_record(**overrides):
    record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    record.update({"id": "secret-case", "title": "Zebra fever", "label": "Written case 9",
                   "hidden": {"history": [{"keywords": [], "answer": "He travelled last week."}],
                              "examination": [{"keywords": [], "answer": "This is zebra fever."}]}})
    record.update(overrides)
    return record


@pytest.fixture
def private_dir(tmp_path):
    directory = tmp_path / "private"
    directory.mkdir()
    (directory / "secret-case.json").write_text(json.dumps(private_record()), encoding="utf-8")
    (directory / "broken.json").write_text("{not json", encoding="utf-8")
    (directory / "marks.json").write_text(json.dumps(private_record(id="with-marks", marks=5)), encoding="utf-8")
    (directory / "integrated.json").write_text(json.dumps({"schema_version": 1, "questions": [
        {"id": "mc1", "kind": "mcq", "prompt": "Which drug first?", "options": ["A drug", "B drug"],
         "answer": 1, "reason": "Because B."},
        {"id": "tf1", "kind": "true_false", "prompt": "Steroids first?", "answer": False,
         "justification": "Adrenaline first."},
        {"id": "cmp1", "kind": "compare", "prompt": "Compare.", "columns": ["X", "Y"],
         "rows": [["Rate", "slow", "fast"]]},
    ]}), encoding="utf-8")
    return directory


@pytest.fixture
def written(settings, private_dir):
    llm = WrittenFakeLLM()
    app = create_app(replace(settings, written_cases_path=private_dir), llm_service=llm)
    with TestClient(app) as client:
        yield client, llm, app


def start(client):
    cases = client.get("/v1/cases").json()["items"]
    written_case = next(item for item in cases if item["kind"] == "written")
    response = client.post(f"/v1/written-cases/{written_case['id']}/sessions", json={})
    assert response.status_code == 201
    data = response.json()
    return data, {"X-Session-Token": data["session_token"]}


ANSWERS = {"q1": {"rows": [{"left": "Tachycardia", "right": "Low stroke volume"}]},
           "q2": {"groups": [{"heading": "Fluids", "rows": [{"left": "Give a fluid bolus", "right": "Restores volume"}]}]}}


# AC-03: nothing leaks before submit ------------------------------------------------------

def test_nothing_reveals_the_condition_or_answers_before_submit(written):
    client, _, _ = written
    record = private_record()
    data, headers = start(client)
    asked = [client.post(f"/v1/written-cases/sessions/{data['session_id']}/ask", headers=headers,
                         json={"question": q, "topic": topic}).text
             for q, topic in (("Past history?", "history"), ("Examine him", "examination"), ("anything", "auto"))]
    log = client.get(f"/v1/written-cases/sessions/{data['session_id']}/log", headers=headers).text
    raw = " ".join([json.dumps(data), client.get("/v1/cases").text, *asked, log]).lower()
    assert "zebra fever" not in raw
    assert "secret-case" not in raw, "written cases use an opaque id, not the learning id"
    for row in record["model_answer"]["q1"]["rows"]:
        assert row["right"].lower() not in raw
    for group in record["model_answer"]["q2"]["groups"]:
        for row in group["rows"]:
            assert row["left"].lower() not in raw and row["right"].lower() not in raw
    assert "model_answer" not in raw and '"hidden"' not in raw
    written_items = [i for i in client.get("/v1/cases").json()["items"] if i["kind"] == "written"]
    assert all("title" not in item for item in written_items)
    assert "title" not in data["case"]


def test_ask_answers_word_for_word_and_withholds_a_named_diagnosis(written):
    client, llm, _ = written
    data, headers = start(client)
    url = f"/v1/written-cases/sessions/{data['session_id']}/ask"
    history = client.post(url, headers=headers, json={"question": "Any history?", "topic": "history"}).json()
    assert history == {**history, "answer": "He travelled last week.", "source": "case_data"}
    exam = client.post(url, headers=headers, json={"question": "Examine", "topic": "examination"}).json()
    assert exam["source"] == "not_available" and "zebra" not in exam["answer"].lower()
    missing = client.post(url, headers=headers, json={"question": "Scan?", "topic": "investigations"}).json()
    assert missing["source"] == "not_available"
    assert llm.patient_calls == [] and llm.json_calls == [], "ask never calls the model"


def test_ask_needs_the_session_token_and_has_a_cap(written):
    client, _, app = written
    data, headers = start(client)
    url = f"/v1/written-cases/sessions/{data['session_id']}/ask"
    assert client.post(url, json={"question": "x"}).status_code == 403
    assert client.post(url, headers={"X-Session-Token": "wrong"}, json={"question": "x"}).status_code == 403
    app.state.written_sessions.max_ask = 2
    codes = [client.post(url, headers=headers, json={"question": "history?"}).status_code for _ in range(3)]
    assert codes == [200, 200, 429]


def test_ask_idempotency_key_returns_the_same_reply_without_counting(written):
    client, _, _ = written
    data, headers = start(client)
    url = f"/v1/written-cases/sessions/{data['session_id']}/ask"
    first = client.post(url, headers={**headers, "Idempotency-Key": "k1"}, json={"question": "history?"}).json()
    again = client.post(url, headers={**headers, "Idempotency-Key": "k1"}, json={"question": "history?"}).json()
    assert first == again
    log = client.get(f"/v1/written-cases/sessions/{data['session_id']}/log", headers=headers).json()
    assert len(log["items"]) == 1


# AC-05: submit, reveal and evaluation ---------------------------------------------------

def test_submit_reveals_and_scores_six_domains_without_marks(written):
    client, llm, _ = written
    data, headers = start(client)
    result = client.post(f"/v1/written-cases/sessions/{data['session_id']}/submit", headers=headers,
                         json={"answers": ANSWERS}).json()
    assert result["reveal"]["condition_title"] == "Zebra fever"
    evaluation = result["evaluation"]
    assert [d["id"] for d in evaluation["domains"]] == [
        "recognition", "pathophysiology", "scenario_data", "management", "rationale", "safety"]
    assert evaluation["overall"] == round((4 + 3 + 4 + 3 + 4 + 5) / 6, 1)
    statuses = {(r["question"], r.get("group_index"), r["row_index"]): r["status"] for r in evaluation["row_status"]}
    assert statuses[("q1", None, 0)] == "met" and statuses[("q1", None, 1)] == "missed"
    assert statuses[("q2", 0, 0)] == "partial"
    assert "900 milligrams" not in evaluation["feedback"], "dose guard removes unsupported doses"
    assert "mark" not in json.dumps(result).lower()
    assert evaluation["critical_errors"] == [], "filler like 'None identified' is dropped"
    student_block = llm.json_calls[0][1]["content"]
    assert "<<<STUDENT>>>" in student_block and "Give a fluid bolus" in student_block


def test_submit_is_idempotent_and_answers_cannot_change(written):
    client, llm, _ = written
    data, headers = start(client)
    url = f"/v1/written-cases/sessions/{data['session_id']}/submit"
    first = client.post(url, headers=headers, json={"answers": ANSWERS}).json()
    assert client.post(url, headers=headers, json={"answers": ANSWERS}).json() == first
    assert len(llm.json_calls) == 1
    changed = {"q1": {"rows": [{"left": "Other", "right": "x"}]}, "q2": {"groups": []}}
    assert client.post(url, headers=headers, json={"answers": changed}).status_code == 409
    ask = client.post(f"/v1/written-cases/sessions/{data['session_id']}/ask", headers=headers, json={"question": "x"})
    assert ask.status_code == 409


def test_evaluator_failure_still_reveals_and_can_be_retried(written):
    client, llm, _ = written
    llm.fail = True
    data, headers = start(client)
    result = client.post(f"/v1/written-cases/sessions/{data['session_id']}/submit", headers=headers,
                         json={"answers": ANSWERS}).json()
    assert result["evaluation"]["status"] == "unavailable" and result["reveal"]["condition_title"]
    llm.fail = False
    retried = client.post(f"/v1/written-cases/sessions/{data['session_id']}/evaluate", headers=headers).json()
    assert retried["status"] == "complete"
    again = client.get(f"/v1/written-cases/sessions/{data['session_id']}/result", headers=headers).json()
    assert again["evaluation"]["status"] == "complete"


def test_result_before_submit_is_refused(written):
    client, _, _ = written
    data, headers = start(client)
    assert client.get(f"/v1/written-cases/sessions/{data['session_id']}/result", headers=headers).status_code == 409


def test_parse_clamps_scores_and_recomputes_the_overall():
    record = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    raw = json.dumps({"domains": {"recognition": 9, "pathophysiology": -2, "scenario_data": "4",
                                  "management": 3, "rationale": 3.6, "safety": None},
                      "overall": 5, "feedback": "ok"})
    result = parse_written_evaluation(raw, record)
    assert [d["score"] for d in result["domains"]] == [5, 0, 4, 3, 4, 0]
    assert result["overall"] == 2.7


# AC-04: abbreviations -------------------------------------------------------------------

def test_abbreviations_are_flagged_and_units_are_not(written):
    client, _, _ = written
    data, headers = start(client)
    texts = ["BP was low so give IV fluids", "ABG shows pH 7.28 and PaCO2 54 mmHg, HCO3 5 mmol/L, SpO2 85"]
    flags = client.post(f"/v1/written-cases/sessions/{data['session_id']}/abbreviations", headers=headers,
                        json={"texts": texts}).json()["flags"]
    terms = {(f["text_index"], f["term"]) for f in flags}
    assert {(0, "BP"), (0, "IV"), (1, "ABG")} <= terms
    assert not {f["term"] for f in flags} & ALLOWLIST
    assert next(f for f in flags if f["term"] == "BP")["suggestion"] == "blood pressure"


def test_plain_words_are_not_flagged():
    assert find_abbreviations("Give oxygen and call the doctor, then reassess the patient.") == []


# AC-02, AC-07: content loading ----------------------------------------------------------

def test_learning_tab_has_six_sections_in_order_and_no_case_data(written):
    client, _, _ = written
    items = client.get("/v1/learning/conditions").json()["items"]
    assert [item["id"] for item in items] == ["secret-case"]
    view = client.get("/v1/learning/conditions/secret-case").json()
    assert [s["id"] for s in view["sections"]] == [
        "what_it_is", "pathophysiology", "manifestations", "diagnostics", "interpretation", "actions"]
    assert "hidden" not in view and "model_answer" not in view and "scenario" not in view
    assert client.get("/v1/learning/conditions/nope").status_code == 404


def test_invalid_records_are_skipped_and_logged(written):
    _, _, app = written
    skipped = {item["file"] for item in app.state.written_cases.skipped}
    assert skipped == {"broken.json", "marks.json"}


def test_missing_private_folder_serves_only_the_example(tmp_path):
    repository = WrittenCaseRepository(tmp_path / "missing", EXAMPLE)
    assert [r["id"] for r in repository.conditions.values()] == ["example-hypovolaemia"]
    assert repository.integrated == []


def test_example_record_is_valid():
    validate_record(json.loads(EXAMPLE.read_text(encoding="utf-8")))


# AC-06: integrated questions --------------------------------------------------------------

def test_integrated_questions_hide_answers_and_mark_choices_in_code(written):
    client, llm, _ = written
    listing = client.get("/v1/integrated-questions").text
    assert "Because B" not in listing and "Adrenaline first" not in listing and '"answer"' not in listing
    right = client.post("/v1/integrated-questions/mc1/answer", json={"choice": "B"}).json()
    wrong = client.post("/v1/integrated-questions/mc1/answer", json={"choice": "a"}).json()
    assert right["correct"] is True and wrong["correct"] is False and right["marked_by"] == "code"
    llm.json_reply = json.dumps({"feedback": "Correct: adrenaline comes first."})
    tf = client.post("/v1/integrated-questions/tf1/answer", json={"value": False, "justification": "Adrenaline"}).json()
    assert tf["correct"] is True and tf["justification_evaluation"]["status"] == "complete"
    compare = client.post("/v1/integrated-questions/cmp1/answer", json={"text": "slow vs fast"}).json()
    assert compare["correct"] is None and compare["marked_by"] == "evaluator"
    assert client.post("/v1/integrated-questions/mc1/answer", json={"choice": "Z"}).status_code == 400


# Auth -------------------------------------------------------------------------------------

def test_v1_requires_the_api_key_when_set(settings, private_dir):
    app = create_app(replace(settings, written_cases_path=private_dir, app_api_key="k"), llm_service=WrittenFakeLLM())
    with TestClient(app) as client:
        assert client.get("/v1/cases").status_code == 401
        assert client.get("/v1/cases", headers={"X-API-Key": "k"}).status_code == 200
        assert client.get("/v1/learning/conditions").status_code == 401
        assert client.get("/v1/integrated-questions").status_code == 401


# AC-01: the page ----------------------------------------------------------------------------

def test_page_has_exactly_two_tabs_and_new_code_never_uses_inner_html():
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    assert html.count('role="tab"') == 2
    assert 'id="tabLearning"' in html and 'id="tabScenarios"' in html
    assert ">Learning</button>" in html and ">Clinical Scenarios</button>" in html
    new_code = html[html.index("// Learning and Clinical Scenarios tabs"):]
    assert ".innerHTML" not in new_code and "insertAdjacentHTML" not in new_code


# Chat cases: typed actions ------------------------------------------------------------------

def test_typed_actions_are_recognised_timed_and_never_complete_the_plan(client):
    started = client.post("/session/start", json={"scenario_id": "chest_pain"}).json()
    sid, headers = started["session_id"], {"X-Session-Token": started["session_token"]}
    url = f"/session/{sid}/action"
    vitals = client.post(url, headers=headers, json={"text": "Take a full set of observations"}).json()
    assert vitals["action_id"] == "vitals" and vitals["result"].startswith("Observations recorded")
    aspirin = client.post(url, headers=headers, json={"text": "Give aspirin 300 milligrams chewed"}).json()
    assert aspirin["action_id"] == "aspirin" and aspirin["result"] == "Recorded."
    assert aspirin["action_label"] == "Give aspirin 300 milligrams chewed"
    escalate = client.post(url, headers=headers, json={"text": "Call the doctor using SBAR"}).json()
    assert escalate["monitor"]["escalated"] is True
    noted = client.post(url, headers=headers, json={"text": "Reassure the patient and explain"}).json()
    assert noted["action_id"] == "noted" and noted["result"] == "Recorded."
    negated = client.post(url, headers=headers, json={"text": "Do not give aspirin"}).json()
    assert negated["action_id"] == "noted"
    assert client.post(url, headers=headers, json={}).status_code == 422
    snapshot = client.get(f"/session/{sid}", headers=headers).json()
    titles = [event["title"] for event in snapshot["timeline"]]
    assert "Give aspirin 300 milligrams chewed" in titles, "the evaluator sees the student's own words"


def test_monitor_alarms_state_readings_and_follow_age():
    from app.clinical import compute_monitor, vital_band

    infant = {"patient": {"age": 9, "age_unit": "months"},
              "case_file": {"baseline_vitals": {"heart_rate": 150, "systolic_bp": 80, "diastolic_bp": 45,
                                                "spo2": 85, "respiratory_rate": 45, "temperature": 37}}}
    monitor = compute_monitor(infant, 0, 0, [], "Intermediate", None, target_minutes=1)
    assert "Marked tachycardia" not in monitor.alerts, "150 is not an alarm in a 9-month-old"
    assert all("escalate" not in alert.lower() and "call for help" not in alert.lower() for alert in monitor.alerts)
    assert vital_band({"patient": {"age": 40}})["hr_high"] == 130


def test_scenarios_are_tagged_adult_or_paediatric(client):
    groups = {item["id"]: item["age_group"] for item in client.get("/scenarios").json()}
    assert groups["chest_pain"] == "adult" and groups["paediatric_dehydration"] == "paediatric"
    assert sum(1 for group in groups.values() if group == "paediatric") >= 6


def test_integrated_questions_filter_by_condition(written):
    client, _, _ = written
    assert len(client.get("/v1/integrated-questions").json()["items"]) == 3
    assert client.get("/v1/integrated-questions", params={"condition": "nope"}).json()["items"] == []


def test_learning_and_integrated_questions_are_locked_during_a_case():
    html = (Path(__file__).resolve().parents[1] / "app" / "static" / "index.html").read_text(encoding="utf-8")
    select_tab = html[html.index("function selectTab("):html.index("function setMode(")]
    assert 'name === "learning" && caseInProgress()' in select_tab
    set_mode = html[html.index("function setMode("):html.index("function syncStartBar(")]
    assert 'mode === "integrated" && caseInProgress()' in set_mode
    assert 'id="leaveButton"' in html and 'id="writtenLeave"' in html
