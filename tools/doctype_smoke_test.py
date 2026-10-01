"""
Run every production document type through the pipeline and report what came out.

This exists because "it works" has repeatedly meant "it worked on the one
document I tried". The production set is PAN, Aadhaar, visiting card,
pamphlet and resume; each is processed end to end and the result printed as
the ERP would receive it, so a regression in routing, reading order or field
assignment is visible immediately rather than discovered by a user.

    python tools/doctype_smoke_test.py
    python tools/doctype_smoke_test.py --only resume --out outputs/smoke.json

Add a document by dropping it in test_data_prod/corpus - anything matching a
known type name is picked up.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = sys.stderr.write

# Where to look, in order of preference, for a document of each type.
CANDIDATES = {
    "PAN": ["test_data_prod/corpus/pan_real.jpg", "test_data_prod/corpus/pan.jpg"],
    "AADHAAR": ["test_data_prod/corpus/aadhaar_real.jpg", "test_data_prod/corpus/aadhaar.jpg"],
    "RESUME": ["test_data_prod/corpus/resume.pdf", "uploads/*RESUME*.pdf"],
    "VISITING_CARD": ["test_data_prod/corpus/*visiting*", "test_data_prod/corpus/*card*",
                      "uploads/*visiting*"],
    "PAMPHLET": ["test_data_prod/corpus/pamphlet.webp", "uploads/*pamphlet*"],
}

# What a correct extraction of each type must contain. Not exact values -
# these are smoke checks on SHAPE, so they keep working when the sample
# document changes.
EXPECTED_FIELDS = {
    "PAN": ["name", "pan_number"],
    "AADHAAR": ["name", "aadhaar_number", "dob", "address"],
    "RESUME": ["candidate_name", "education", "skills"],
    "VISITING_CARD": ["name", "phone"],
    "PAMPHLET": [],
}


def find(patterns):
    for pattern in patterns:
        matches = sorted(ROOT.glob(pattern))
        if matches:
            return matches[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--out", default="outputs/doctype_smoke_test.json")
    args = ap.parse_args()

    from app.core.database import SessionLocal
    from app.models.document import DocumentRecord, DocumentResult
    from app.services.erp_payload import erp_from_record
    from app.services.ocr_pipeline import pipeline

    db = SessionLocal()
    results = []

    for doc_type, patterns in CANDIDATES.items():
        if args.only and args.only.lower() not in doc_type.lower():
            continue
        path = find(patterns)
        if not path:
            log(f"\n{doc_type:<14} no sample found - skipped\n")
            results.append({"type": doc_type, "status": "NO_SAMPLE"})
            continue

        log(f"\n{'=' * 78}\n{doc_type}  <-  {path.name}\n{'=' * 78}\n")
        t0 = time.time()
        try:
            out = pipeline.process_file(db, path, path.name, requested_doc_type="AUTO")
        except Exception as exc:
            log(f"  FAILED: {type(exc).__name__}: {exc}\n")
            results.append({"type": doc_type, "status": "ERROR", "error": str(exc)})
            continue
        elapsed = round(time.time() - t0, 1)

        record = db.query(DocumentRecord).filter(DocumentRecord.id == out["id"]).first()
        stored = db.query(DocumentResult).filter(DocumentResult.document_id == out["id"]).first()
        erp = erp_from_record(record, stored)
        data = erp.get("data", {})

        missing = [f for f in EXPECTED_FIELDS.get(doc_type, []) if not data.get(f)]
        verdict = "OK" if not missing else f"MISSING {', '.join(missing)}"

        log(f"  detected  : {erp.get('document_type')}   ({elapsed}s, {out.get('pages')} page(s))\n")
        log(f"  confidence: {erp.get('confidence')}   review_required={erp.get('review_required')}\n")
        log(f"  fields    : {verdict}\n")
        for key, value in list(data.items())[:9]:
            rendered = json.dumps(value, ensure_ascii=False)
            log(f"      {key:<22} {rendered[:96]}\n")
        if erp.get("line_items"):
            log(f"      line_items             {len(erp['line_items'])} row(s)\n")
        # An "N/A" reaching the ERP is a regression - absent must mean absent.
        blob = json.dumps(data, ensure_ascii=False).lower()
        for marker in ('"n/a"', '"none"', '"not available"', '"-"'):
            if marker in blob:
                log(f"      !! placeholder {marker} reached the ERP payload\n")
                verdict += f" +PLACEHOLDER{marker}"

        results.append({
            "type": doc_type, "sample": path.name, "document_id": out["id"],
            "detected": erp.get("document_type"), "seconds": elapsed,
            "pages": out.get("pages"), "confidence": erp.get("confidence"),
            "review_required": erp.get("review_required"),
            "status": "OK" if verdict == "OK" else verdict,
            "erp_data": data,
        })

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    log("\n" + "=" * 78 + "\n")
    for r in results:
        log(f"  {r['type']:<16} {r.get('detected', '-'):<16} {r['status']}\n")
    log(f"\n  -> {out_path}\n")
    return 0 if all(r["status"] in ("OK", "NO_SAMPLE") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
