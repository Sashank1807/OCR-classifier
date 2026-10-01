"""
Ground-truth-free structural bug finder.

Every extraction bug fixed on 2026-09-10 (a product row absorbed into the
header, a TOTAL label glued onto the product above it, ERP chrome parsed as
data) left an obvious STRUCTURAL fingerprint in the output - long before anyone
knew the document's correct values. None of them were visible in the 11-fixture
benchmark, because those fixtures happen not to contain the patterns.

So this scans unseen documents for shapes a real table cannot have, which finds
bugs without needing ground truth for any of them.

    python -m tests.structural_anomaly_scan --limit 10
    python -m tests.structural_anomaly_scan --files 1000411293.jpg,1000517651.jpg
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from app.services.table_ocr_service import table_ocr_service

TOTAL_WORDS = ("total", "s-total", "subtotal", "grand total")
CHROME = re.compile(
    r"[\w.+-]+@[\w-]+\.\w+|www\.|\bf\d{1,2}\s*[-:]|\bmobile\s*:|\bhelp\s*line|\besc\b|jline",
    re.IGNORECASE,
)
HEADER_WORDS = ("description", "particulars", "item", "product", "packing", "opening",
                "receipt", "issue", "closing", "balance", "qty", "rate", "amount")


def scan_table(columns: List[str], rows: List[List[str]]) -> List[str]:
    issues: List[str] = []

    # A column NAME that also carries a product's text means the header band
    # swallowed the first data row - the row is gone and the column is mislabelled.
    for j, c in enumerate(columns):
        if re.match(r"^Col_Num_\d+_\d+$", c or ""):
            issues.append(f"column {j} is an invented placeholder: {c!r}")
        words = (c or "").split()
        header_hits = sum(1 for w in words if any(h in w.lower() for h in HEADER_WORDS))
        if len(words) >= 4 and header_hits < len(words) - 1:
            issues.append(f"column name looks like it absorbed a data row: {c!r}")
        if CHROME.search(c or ""):
            issues.append(f"column name contains application chrome: {c!r}")

    # Adjacent columns carrying the same label mean one column was split in two,
    # halving its data across both. Repeats far apart are legitimate ("QTY."
    # under OPENING and again under RECEIPT), so only adjacency counts.
    def _norm_label(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", (x or "").lower())
    for j in range(len(columns) - 1):
        a, b = _norm_label(columns[j]), _norm_label(columns[j + 1])
        if a and a == b:
            # Only suspicious if one of the pair carries no data. Two adjacent
            # columns sharing a label is normally a super-header ("Opn") spanning
            # real Qty/Value sub-columns - both hold data and both are correct.
            fa = sum(1 for r in rows if j < len(r) and (r[j] or "").strip())
            fb = sum(1 for r in rows if j + 1 < len(r) and (r[j + 1] or "").strip())
            if min(fa, fb) <= max(1, len(rows) // 5):
                issues.append(
                    f"columns {j}/{j+1} share label {columns[j]!r} and one is empty "
                    f"({fa} vs {fb} rows) - possible split column")

    desc_seen: Dict[str, int] = {}
    for i, r in enumerate(rows):
        if not r:
            continue
        desc = (r[0] or "").strip()
        rest = [c for c in r[1:] if (c or "").strip()]
        low = desc.lower()

        # "VETORY P TAB TOTAL": a total label merged onto a product description.
        if any(w in low for w in TOTAL_WORDS) and len(desc.split()) > 2:
            non_total = re.sub(r"(?i)\b(grand\s+)?(s-)?(sub)?total\b[: ]*", "", desc).strip()
            if non_total and not non_total.isdigit():
                issues.append(f"row {i}: total label merged into a product row: {desc!r}")

        if CHROME.search(desc) or any(CHROME.search(c or "") for c in r[1:]):
            issues.append(f"row {i}: application chrome parsed as table data: {r}")

        # A row with a description but not one value anywhere.
        if desc and not rest:
            issues.append(f"row {i}: description with no values at all: {desc!r}")

        # Numbers that are not numbers, in cells that should hold numbers.
        # Which columns those are is decided by the column NAME - packing/unit
        # columns legitimately read "5GM", and "0.000" is a perfectly good zero.
        for j, cell in enumerate(r[1:], start=1):
            s = (cell or "").strip()
            cname = (columns[j] if j < len(columns) else "").lower()
            is_text_col = any(k in cname for k in ("pack", "unit", "desc", "name", "item",
                                                   "batch", "expiry", "day", "sh.exp"))
            if s and re.match(r"^0{2,}[1-9]", s.replace(".", "")):
                issues.append(f"row {i} col {j}: leading-zero garble {s!r}")
            if s and not is_text_col and re.search(r"\d", s) and re.search(r"[A-Za-z]{2,}", s):
                issues.append(f"row {i} col {j}: letters inside a numeric cell {s!r}")

        if desc:
            desc_seen[low] = desc_seen.get(low, 0) + 1

    for d, n in desc_seen.items():
        # Twice is normal (product + its S-Total). Three times is not a table shape.
        if n > 2:
            issues.append(f"description repeats {n}x (rows duplicated?): {d!r}")

    return issues


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--files", default="", help="comma-separated filenames instead of a sample")
    ap.add_argument("--ext", default="jpg,jpeg,jfif")
    args = ap.parse_args()

    src = Path("test_data_june")
    # Documents already covered by fixtures - the point is to look elsewhere.
    known = {"1000411295.jpg", "1000411296.jpg", "1000517666.jpg", "1000517796.jpg",
             "Agarwal Jaipur.jpeg", "Bansal barelly.jpeg"}

    if args.files:
        files = [src / f.strip() for f in args.files.split(",") if f.strip()]
    else:
        exts = {"." + e.strip().lstrip(".").lower() for e in args.ext.split(",")}
        files = [p for p in sorted(src.iterdir())
                 if p.suffix.lower() in exts and p.name not in known][: args.limit]

    print("=" * 100)
    print(f"STRUCTURAL ANOMALY SCAN - {len(files)} unseen document(s), no ground truth used")
    print("=" * 100)

    report = []
    for f in files:
        if not f.exists():
            print(f"\n--- {f.name}: NOT FOUND")
            continue
        t0 = time.time()
        try:
            res = table_ocr_service.extract_table(f)
        except Exception as e:
            print(f"\n--- {f.name}: EXTRACTION ERROR {type(e).__name__}: {e}")
            report.append({"document": f.name, "error": f"{type(e).__name__}: {e}"})
            continue

        tables = res.get("tables") or []
        if not tables:
            print(f"\n--- {f.name}: no table detected ({time.time()-t0:.0f}s)")
            report.append({"document": f.name, "issues": ["no table detected"]})
            continue

        cols = tables[0].get("columns") or []
        rows = tables[0].get("rows") or []
        issues = scan_table(cols, rows)
        report.append({"document": f.name, "columns": cols,
                       "row_count": len(rows), "issues": issues})
        print(f"\n--- {f.name}  ({len(rows)} rows, {len(cols)} cols, {time.time()-t0:.0f}s)")
        print(f"    columns: {cols}")
        if issues:
            for it in issues[:8]:
                print(f"    [!] {it}")
            if len(issues) > 8:
                print(f"    ... {len(issues)-8} more")
        else:
            print("    clean")

    Path("outputs/structural_scan.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    total = sum(len(r.get("issues") or []) for r in report)
    print("\n" + "=" * 100)
    print(f"{total} structural anomalies across {len(report)} documents -> outputs/structural_scan.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
