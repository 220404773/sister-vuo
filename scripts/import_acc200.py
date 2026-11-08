"""Turn the operator's ACC 200 transcription into private written-case records.

    python scripts/import_acc200.py <transcription.json> [<out dir>]

The transcription is a JSON file with "cases" (one object per condition) and "integrated",
made by hand from the paper and its memorandum (a PDF parse is too unreliable for model
answers). The out dir defaults to scenarios/private/ and must stay inside it: those files hold
model answers and are gitignored. The script prints counts only, never answer text, and writes
review_report.md listing every review flag for the operator.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.written import SCHEMA_VERSION, validate_integrated, validate_record

PRIVATE = (ROOT / "scenarios" / "private").resolve()
TOPIC_KEYS = {"history": "history", "examination": "examination",
              "investigations": "investigations", "repeat observations": "observations"}


def _row(pair: list[str]) -> dict[str, str]:
    return {"left": pair[0], "right": pair[1]}


def to_record(case: dict) -> dict:
    learning = case["learning"]
    groups = case["management"]["groups"]
    return {
        "schema_version": SCHEMA_VERSION,
        "id": case["id"],
        "number": case["number"],
        "title": case["title"],
        "label": f"Written case {case['number']}",
        "learning": {
            "what_it_is": {"text": learning["definition"], "review_flag": True,
                           "review_note": "Draft written for the app; awaiting clinician review."},
            "pathophysiology": {"text": learning["pathophysiology"], "review_flag": True,
                                "review_note": "Draft written for the app; awaiting clinician review."},
            "manifestations": {"items": [{"text": m, "pathophysiology": p} for m, p in case["manifestations"]],
                               "review_flag": False, "review_note": None},
            "diagnostics": {"items": [{"test": test} for test in learning["diagnostic_tests"]],
                            "review_flag": True,
                            "review_note": "Draft written for the app; awaiting clinician review."},
            "interpretation": {"items": [{"text": text} for text in learning["interpretation"]],
                               "review_flag": True,
                               "review_note": "Draft written for the app; awaiting clinician review."},
            "actions": {"items": [{"heading": heading, "action": a, "rationale": r}
                                  for heading, rows in groups for a, r in rows],
                        "review_flag": False, "review_note": None},
        },
        "scenario": {
            "age": case["patient"]["age"], "sex": case["patient"]["sex"], "text": case["scenario_text"],
            "monitored": [{"label": label, "value": value} for label, value in case["monitored"]],
            "abg": [{"label": label, "value": value} for label, value in case.get("blood_gas", {}).items()] or None,
            "bloods": [{"label": label, "value": value} for label, value in case.get("bloods", {}).items()] or None,
            "ventilator": case.get("ventilator"),
            "imaging": case.get("imaging"),
        },
        "hidden": {TOPIC_KEYS[topic]: [{"keywords": [], "answer": text}]
                   for topic, text in case.get("more_info", {}).items()},
        "model_answer": {
            "q1": {"rows": [_row(pair) for pair in case["manifestations"]]},
            "q2": {"ordered": bool(case["management"].get("ordered")),
                   "groups": [{"heading": heading, "rows": [_row(pair) for pair in rows]} for heading, rows in groups]},
        },
        "review_flags": [{"note": note} for note in case.get("review_flags", [])],
    }


def to_integrated(data: dict) -> dict:
    questions = [
        {"id": item["id"], "kind": "compare", "prompt": item["question"],
         "columns": item["columns"], "rows": item["rows"], "conditions": item.get("conditions", [])}
        for item in data.get("compare", [])
    ]
    questions += [
        {"id": item["id"], "kind": "mcq", "prompt": item["question"], "options": item["options"],
         "answer": item["answer"], "reason": item["reason"], "conditions": item.get("conditions", [])}
        for item in data.get("multiple_choice", [])
    ]
    questions += [
        {"id": item["id"], "kind": "true_false", "prompt": item["statement"], "answer": item["answer"],
         "justification": item["justification"], "conditions": item.get("conditions", [])}
        for item in data.get("true_false", [])
    ]
    return {"schema_version": SCHEMA_VERSION, "questions": questions,
            "review_flags": [{"note": note} for note in data.get("review_flags", [])]}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    source = Path(argv[0])
    out = Path(argv[1]).resolve() if len(argv) > 1 else PRIVATE
    if out != PRIVATE and PRIVATE not in out.parents:
        print(f"Refusing to write outside {PRIVATE}: model answers must stay private.")
        return 2
    data = json.loads(source.read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    report = ["# ACC 200 import: items to review before students use the app", ""]
    for case in data["cases"]:
        record = to_record(case)
        validate_record(record)
        (out / f"{record['id']}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report += [f"## {record['label']}"] + [f"- {flag['note']}" for flag in record["review_flags"]] + [""]
    integrated = to_integrated(data.get("integrated", {}))
    validate_integrated(integrated)
    (out / "integrated.json").write_text(json.dumps(integrated, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report += ["## Integrated questions"] + [f"- {flag['note']}" for flag in integrated["review_flags"]]
    (out / "review_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Wrote {len(data['cases'])} cases and {len(integrated['questions'])} integrated questions to {out}")
    print(f"Review flags listed in {out / 'review_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
