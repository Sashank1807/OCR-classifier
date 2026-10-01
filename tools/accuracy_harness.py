"""
Field-level accuracy AND run-to-run stability on the production documents.

This exists because bugs in this system were being found by the user, one at
a time, in production - and because a single passing run proves very little
about a stochastic extractor. Two different questions have to be answered
together:

  ACCURACY  - is each field's value right, compared to ground truth a human
              read off the document?
  STABILITY - does the same document give the same answer every time?

The second matters as much as the first and is invisible to single-run
testing. A field that is correct 70% of the time looks fine when you check
it once, and it is exactly what a user experiences as "it keeps getting
things wrong". Measuring it needs repeats, so every document is processed
`--runs` times and each field is reported as

    ok 3/3   stable      the same correct value every run
    ok 2/3   UNSTABLE    right twice, wrong once - the worst kind of bug
    ok 0/3   stable      consistently wrong, which is at least easy to fix

    python tools/accuracy_harness.py --runs 3
    python tools/accuracy_harness.py --runs 1 --only aadhaar   # quick check

Ground truth lives in tests/ground_truth/production/*.json. Match modes:

  exact         byte-for-byte
  exact_ci      case-insensitive, whitespace-collapsed
  digits        compare only the digits (so "6394 1918 1234" == "639419181234")
  contains      the expected string appears in the value
  contains_all  every expected string appears (for addresses and lists)

Field paths address nested values: `contact_info.email`,
`education[0].gpa_or_grade`, `skills.languages`.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GT_DIR = ROOT / "tests" / "ground_truth" / "production"
log = sys.stderr.write


# --------------------------------------------------------------------------- #
# Reading a value out of a nested payload
# --------------------------------------------------------------------------- #

_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


def resolve(data: Any, path: str) -> Any:
    """`education[0].gpa_or_grade` -> the value, or None if any step is absent."""
    current = data
    for part in path.split("."):
        m = _INDEX.match(part)
        index = None
        if m:
            part, index = m.group(1), int(m.group(2))
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
        if index is not None:
            if not isinstance(current, list) or index >= len(current):
                return None
            current = current[index]
    return current


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " | ".join(_text(v) for v in value)
    if isinstance(value, dict):
        return " | ".join(f"{k}={_text(v)}" for k, v in value.items())
    return str(value)


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def matches(value: Any, expect: Any, mode: str) -> bool:
    got = _text(value)
    if not got.strip():
        return False

    if mode == "exact":
        return got.strip() == str(expect).strip()
    if mode == "exact_ci":
        return _squash(got) == _squash(str(expect))
    if mode == "digits":
        return re.sub(r"\D", "", got) == re.sub(r"\D", "", str(expect))
    if mode == "contains":
        return _squash(str(expect)) in _squash(got)
    if mode == "contains_all":
        haystack = _squash(got)
        return all(_squash(str(e)) in haystack for e in expect)
    raise ValueError(f"unknown match mode '{mode}'")


# --------------------------------------------------------------------------- #
# One run of one document
# --------------------------------------------------------------------------- #

def process(db, path: Path) -> Dict[str, Any]:
    from app.models.document import DocumentRecord, DocumentResult
    from app.services.erp_payload import erp_from_record
    from app.services.ocr_pipeline import pipeline

    started = time.time()
    out = pipeline.process_file(db, path, path.name, requested_doc_type="AUTO")
    record = db.query(DocumentRecord).filter(DocumentRecord.id == out["id"]).first()
    stored = db.query(DocumentResult).filter(DocumentResult.document_id == out["id"]).first()
    from app.models.document import DocumentPage
    pages = (db.query(DocumentPage)
               .filter(DocumentPage.document_id == out["id"])
               .order_by(DocumentPage.page_number).all())
    return {
        "erp": erp_from_record(record, stored),
        "seconds": round(time.time() - started, 1),
        "document_id": out["id"],
        "page_text": {p.page_number: len((p.markdown or "").strip()) for p in pages},
    }


def evaluate(fixture: Dict[str, Any], run: Dict[str, Any]) -> Dict[str, Any]:
    """Score one run against the fixture. Returns per-field ok/value plus checks."""
    erp = run["erp"]
    data = erp.get("data", {})
    fields = {}
    for path, rule in fixture["fields"].items():
        value = resolve(data, path)
        fields[path] = {
            "ok": matches(value, rule["expect"], rule.get("match", "exact")),
            "value": _text(value),
        }

    checks = {}
    expected_type = fixture.get("document_type")
    if expected_type:
        checks["document_type"] = erp.get("document_type") == expected_type

    if fixture.get("must_not_need_review"):
        checks["no_review_needed"] = not erp.get("review_required", False)

    for path, forbidden in (fixture.get("must_not_contain") or {}).items():
        got = _squash(_text(resolve(data, path)))
        checks[f"{path}_clean"] = not any(_squash(str(f)) in got for f in forbidden)

    if fixture.get("page_count"):
        checks["page_count"] = len(run["page_text"]) == fixture["page_count"]
    for page in fixture.get("pages_must_have_text") or []:
        checks[f"page_{page}_has_text"] = run["page_text"].get(page, 0) > 50

    return {"fields": fields, "checks": checks,
            "seconds": run["seconds"], "confidence": erp.get("confidence")}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3,
                    help="repeats per document; >1 is what measures stability")
    ap.add_argument("--only", help="substring of a fixture name")
    ap.add_argument("--out", default="outputs/accuracy_report.json")
    args = ap.parse_args()

    fixtures = []
    for f in sorted(GT_DIR.glob("*.json")):
        if args.only and args.only.lower() not in f.stem.lower():
            continue
        fixtures.append((f.stem, json.loads(f.read_text(encoding="utf-8"))))
    if not fixtures:
        raise SystemExit(f"No fixtures in {GT_DIR}")

    from app.core.database import SessionLocal
    db = SessionLocal()

    report, all_fields_ok, all_fields_total = [], 0, 0
    unstable_total = 0

    for name, fixture in fixtures:
        path = ROOT / fixture["document"]
        if not path.exists():
            log(f"\n{name}: document missing ({fixture['document']}) - skipped\n")
            report.append({"fixture": name, "status": "MISSING_DOCUMENT"})
            continue

        log(f"\n{'=' * 84}\n{name}   {path.name}   {args.runs} run(s)\n{'=' * 84}\n")
        runs = []
        for i in range(args.runs):
            try:
                runs.append(evaluate(fixture, process(db, path)))
            except Exception as exc:
                log(f"  run {i + 1} FAILED: {type(exc).__name__}: {exc}\n")
                runs.append({"fields": {}, "checks": {"pipeline_ran": False},
                             "seconds": 0, "confidence": None})

        # Per field: how many runs were correct, and how many distinct values.
        field_rows = {}
        for path_key in fixture["fields"]:
            oks = [r["fields"].get(path_key, {}).get("ok", False) for r in runs]
            values = [r["fields"].get(path_key, {}).get("value", "") for r in runs]
            distinct = len(set(values))
            passed = sum(oks)
            field_rows[path_key] = {
                "passed": passed, "runs": len(runs), "distinct_values": distinct,
                "stable": distinct == 1, "values": sorted(set(values)),
            }
            all_fields_ok += passed
            all_fields_total += len(runs)
            if distinct > 1:
                unstable_total += 1

            state = "ok " if passed == len(runs) else ("BAD" if passed == 0 else "FLAKY")
            stability = "stable  " if distinct == 1 else f"UNSTABLE({distinct})"
            shown = (values[0] or "(empty)")[:52]
            log(f"  {state} {passed}/{len(runs)}  {stability}  {path_key:<30} {shown}\n")
            if distinct > 1:
                for v in sorted(set(values)):
                    log(f"         variant: {(v or '(empty)')[:70]}\n")

        check_rows = {}
        for key in {k for r in runs for k in r["checks"]}:
            passed = sum(1 for r in runs if r["checks"].get(key))
            check_rows[key] = {"passed": passed, "runs": len(runs)}
            mark = "ok " if passed == len(runs) else "BAD"
            log(f"  {mark} {passed}/{len(runs)}            check: {key}\n")

        secs = [r["seconds"] for r in runs if r["seconds"]]
        log(f"  -- confidence {[r['confidence'] for r in runs]}   "
            f"{min(secs) if secs else 0}-{max(secs) if secs else 0}s\n")

        report.append({"fixture": name, "document": fixture["document"],
                       "runs": args.runs, "fields": field_rows, "checks": check_rows,
                       "confidences": [r["confidence"] for r in runs]})

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    pct = (all_fields_ok / all_fields_total * 100) if all_fields_total else 0
    failed_checks = sum(1 for r in report for c in (r.get("checks") or {}).values()
                        if c["passed"] < c["runs"])
    log("\n" + "=" * 84 + "\n")
    log(f"  FIELD ACCURACY   {all_fields_ok}/{all_fields_total}  ({pct:.1f}%)\n")
    log(f"  UNSTABLE FIELDS  {unstable_total}   (same document, different answer)\n")
    log(f"  FAILED CHECKS    {failed_checks}\n")
    log("=" * 84 + f"\n\n  -> {out_path}\n")

    return 0 if (all_fields_ok == all_fields_total and not failed_checks
                 and not unstable_total) else 1


if __name__ == "__main__":
    sys.exit(main())
