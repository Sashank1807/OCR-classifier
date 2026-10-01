"""
Ground-truth-free accuracy audit.

A stock statement checks itself: every product row satisfies
Closing = Opening + In - Out, and the total row equals the sum of its product
rows. So extraction quality can be measured on ANY document without a hand-made
fixture - if the numbers came out right, the arithmetic closes; if a digit was
misread, it does not.

That matters for two reasons:
  1. Only 11 of the 234 documents in test_data_june have ground truth, so
     fixture-based scoring can only ever see ~5% of the real corpus.
  2. The rows that FAIL arithmetic here are exactly the pool that an
     arithmetic-driven self-correction feature could repair, so this run
     doubles as a sizing estimate for that work.

    python -m tests.arithmetic_consistency_audit --limit 12 [--pattern jpg]
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.core.database import SessionLocal
from app.services.ocr_pipeline import OCRPipeline

# Column-role synonyms as they appear across the different ERPs in the corpus.
ROLE_PATTERNS = {
    "opening": [r"^op\b", r"opening", r"op\.?bal", r"o\.?bal"],
    "inflow":  [r"receipt", r"^in\b", r"purchase", r"inward"],
    "outflow": [r"issue", r"^out\b", r"sale", r"outward", r"dispatch"],
    "closing": [r"closing", r"^bal", r"balance", r"clos"],
}


def map_roles(columns: List[str]) -> Dict[str, int]:
    roles: Dict[str, int] = {}
    for idx, name in enumerate(columns):
        n = (name or "").strip().lower()
        for role, pats in ROLE_PATTERNS.items():
            if role in roles:
                continue
            if any(re.search(p, n) for p in pats):
                roles[role] = idx
                break
    return roles


def num(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip().replace(",", "")
    if s in ("", "-", "--", "---"):
        return 0.0          # a printed dash means "nothing moved", i.e. zero
    try:
        return float(s)
    except ValueError:
        return None


def audit_table(columns: List[str], rows: List[List[str]]) -> Optional[Dict[str, Any]]:
    roles = map_roles(columns)
    if not {"opening", "inflow", "outflow", "closing"} <= set(roles):
        return None  # not a stock-movement table; nothing to verify

    o, i, u, c = roles["opening"], roles["inflow"], roles["outflow"], roles["closing"]
    checked = balanced = 0
    off_by: List[float] = []

    for r in rows:
        if max(o, i, u, c) >= len(r):
            continue
        vals = [num(r[o]), num(r[i]), num(r[u]), num(r[c])]
        if any(v is None for v in vals):
            continue
        ov, iv, uv, cv = vals
        if ov == iv == uv == cv == 0.0:
            continue  # an entirely empty row proves nothing
        checked += 1
        diff = round((ov + iv - uv) - cv, 2)
        if abs(diff) < 0.05:
            balanced += 1
        else:
            off_by.append(diff)

    if not checked:
        return None
    return {
        "rows_checked": checked,
        "rows_balanced": balanced,
        "balance_rate_pct": round(balanced / checked * 100.0, 1),
        "discrepancies": off_by[:10],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--pattern", default="", help="only files whose name contains this")
    ap.add_argument("--ext", default="jpg,jpeg", help="comma-separated extensions")
    args = ap.parse_args()

    exts = {"." + e.strip().lstrip(".").lower() for e in args.ext.split(",")}
    src = Path("test_data_june")
    files = sorted(p for p in src.iterdir()
                   if p.suffix.lower() in exts and (not args.pattern or args.pattern.lower() in p.name.lower()))
    files = files[: args.limit]

    print("=" * 96)
    print(f"ARITHMETIC CONSISTENCY AUDIT - {len(files)} document(s), no ground truth required")
    print("=" * 96)

    pipeline = OCRPipeline()
    results = []
    for f in files:
        t0 = time.time()
        db = SessionLocal()
        try:
            out = pipeline.process_file(
                db=db, file_path=f, original_filename=f.name, requested_doc_type="AUTO"
            )
        except Exception as e:
            print(f"{f.name[:38]:38} ERROR {type(e).__name__}: {str(e)[:40]}")
            continue
        finally:
            db.close()

        tables = (out or {}).get("tables") or []
        verdict = None
        for t in tables:
            verdict = audit_table(t.get("columns") or [], t.get("rows") or [])
            if verdict:
                break

        elapsed = time.time() - t0
        if not verdict:
            print(f"{f.name[:38]:38} no stock-movement columns detected      {elapsed:5.0f}s")
            continue

        verdict["document"] = f.name
        verdict["elapsed_s"] = round(elapsed, 1)
        results.append(verdict)
        print(f"{f.name[:38]:38} {verdict['rows_balanced']:>3}/{verdict['rows_checked']:<3} rows balance "
              f"= {verdict['balance_rate_pct']:5.1f}%   {elapsed:5.0f}s")

    if results:
        tot_c = sum(r["rows_checked"] for r in results)
        tot_b = sum(r["rows_balanced"] for r in results)
        print("-" * 96)
        print(f"OVERALL: {tot_b}/{tot_c} rows satisfy Closing = Opening + In - Out "
              f"({tot_b / max(1, tot_c) * 100:.1f}%)")
        print(f"Unbalanced rows are the addressable pool for arithmetic self-correction: {tot_c - tot_b}")
        Path("outputs/arithmetic_audit.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
        print("Saved: outputs/arithmetic_audit.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
