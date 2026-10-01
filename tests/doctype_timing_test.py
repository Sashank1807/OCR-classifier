"""Timed extraction + verification test across the production document types."""
import json, sys, time
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from app.core.database import SessionLocal
from app.services.ocr_pipeline import pipeline

CASES = [
    ("PAN card",      "uploads/6b75bf34_PAN-CARD1.jpg"),
    ("Aadhaar card",  "uploads/11825e08_aadhaar1.jpg"),
    ("Resume (PDF)",  "uploads/cee61981_SUBHAJIT_PATI_RESUME.pdf"),
    ("Pamphlet A",    "test_data_prod/pamphlet_health_services.webp"),
    ("Pamphlet B",    "test_data_prod/pamphlet_medical_service.webp"),
]

out = []
for label, path in CASES:
    p = Path(path)
    if not p.exists():
        print(f"  SKIP {label}: {p} missing", flush=True); continue
    db = SessionLocal(); t0 = time.time()
    try:
        res = pipeline.process_file(db, p, p.name, requested_doc_type="AUTO")
        el = time.time() - t0
        sd = res.get("structured_data") or {}
        fv = sd.get("field_verification") or {}
        rec = {"label": label, "seconds": round(el, 2),
               "type": res.get("document_type"),
               "confidence": sd.get("overall_confidence"),
               "needs_review": sd.get("needs_manual_review"),
               "counts": fv.get("counts"),
               "legible": (fv.get("source_legibility") or {}).get("legible"),
               "legibility_reasons": (fv.get("source_legibility") or {}).get("reasons"),
               "flagged": [f"{f['field']}: {f['reason']}" for f in (fv.get("flagged") or [])][:5],
               "fields": sd.get("fields")}
        print(f"  {label:<14} {el:>6.1f}s  {rec['type']:<14} conf={rec['confidence']}  review={rec['needs_review']}  {rec['counts']}", flush=True)
    except Exception as e:
        rec = {"label": label, "seconds": round(time.time()-t0,2), "error": f"{type(e).__name__}: {e}"}
        print(f"  {label:<14} ERROR {e}", flush=True)
    finally:
        db.close()
    out.append(rec)

Path("outputs/doctype_test_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote outputs/doctype_test_results.json")
