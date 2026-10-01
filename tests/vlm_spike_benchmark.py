"""
Architecture spike: can a vision-language model extract these tables directly,
well enough to replace the geometric table-reconstruction pipeline?

Sends each benchmark document's page image to a local Ollama VLM, asks for the
table as structured JSON, and scores the result with the SAME helpers the
production benchmark uses (map_columns_semantically / align_rows_to_ground_truth
/ compare_cell_values) so the comparison is like-for-like on cell accuracy.

This measures the question rather than arguing it: the current pipeline spends
~3,500 lines reconstructing table structure from OCR text boxes, and every
structural failure found so far (tilt, page curvature, header detection, column
invention, row merging) is a class of bug a VLM does not have. What a VLM may
have instead is hallucination, dropped rows on long tables, and no per-cell
provenance - all of which show up in these numbers.

    python -m tests.vlm_spike_benchmark [--model qwen3-vl:8b] [--docs 2]
"""

import argparse
import base64
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from tests.comprehensive_cell_benchmark import (
    align_rows_to_ground_truth,
    compare_cell_values,
    get_cell_semantic,
    map_columns_semantically,
    parse_numeric,
)

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

GT_FILES = [
    "1000411295.json",
    "1000411296.json",
    "1000517666.json",
    "1000517796.json",
    "agarwal_jaipur.json",
    "bansal_barelly.json",
]

# Headline numbers from the production pipeline (benchmark_run_postfix20.log)
# so the spike prints a side-by-side instead of a bare number.
PIPELINE_BASELINE = {
    "1000411295.jpg": 74.8,
    "1000411296.jpg": 86.7,
    "1000517666.jpg": 79.5,
    "1000517796.jpg": 95.0,
    "Agarwal Jaipur.jpeg": 53.5,
    "Bansal barelly.jpeg": 75.0,
}

SEARCH_DIRS = [Path("test_data_june"), Path("samples/benchmark_suite"), Path("uploads")]


def build_prompt(columns: List[str]) -> str:
    """
    The anti-hallucination wording is the important part: a VLM's characteristic
    failure on these documents is completing the pattern it expects rather than
    reporting what is printed, so the prompt has to make "leave it blank" the
    explicitly correct answer.
    """
    col_list = ", ".join(f'"{c}"' for c in columns)
    return (
        "You are reading a printed stock/sales statement table from a photograph.\n\n"
        f"Return EVERY data row of the table, using exactly these columns in this order: [{col_list}].\n\n"
        "Rules:\n"
        "- Transcribe ONLY what is visibly printed. Never infer, calculate, or complete a value.\n"
        "- If a cell is blank, return an empty string \"\".\n"
        "- If a cell shows a dash, return \"-\".\n"
        "- Preserve the exact characters and decimals as printed (e.g. \"51.000\", not \"51\").\n"
        "- Include subtotal and total rows as their own rows, in the order they appear.\n"
        "- Do not skip rows. Do not merge two rows into one.\n\n"
        "Respond with ONLY a JSON object, no commentary:\n"
        '{"rows": [["cell1","cell2",...], ["cell1","cell2",...]]}'
    )


def call_vlm(model: str, image_path: Path, prompt: str, timeout: int = 900) -> Tuple[str, float]:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        # Reasoning models otherwise spend the whole token budget in "thinking"
        # and return an empty response (done_reason: length). Transcription
        # needs the output, not the deliberation.
        "think": False,
        "options": {"temperature": 0, "num_predict": 8192},
    }
    t0 = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    out = data.get("response") or ""
    if not out and data.get("thinking"):
        # Model ignored think=False; salvage any JSON it reasoned its way to.
        out = data["thinking"]
    return out, time.time() - t0


def parse_rows(raw: str) -> List[List[str]]:
    """Pull the rows array out of the model's reply, tolerating stray prose or fencing."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    rows = data.get("rows") or []
    out = []
    for r in rows:
        if isinstance(r, list):
            out.append(["" if c is None else str(c).strip() for c in r])
    return out


def score(gt_data: Dict[str, Any], extracted_rows: List[List[str]]) -> Dict[str, Any]:
    """
    Mirrors the production benchmark's cell-accuracy computation using its own
    helpers, so the headline number means the same thing in both places.
    """
    expected_columns = gt_data["columns"]
    gt_rows = gt_data["rows"]
    extracted_columns = [c["name"] for c in expected_columns]  # column order was dictated in the prompt

    col_mapping, col_sem = map_columns_semantically(expected_columns, extracted_columns)
    row_mapping = align_rows_to_ground_truth(gt_rows, extracted_rows, expected_columns, col_mapping)

    total = correct = incorrect = missing = 0
    numeric_total = numeric_correct = 0
    hallucinated = 0  # value produced where ground truth is blank

    for r_idx, gt_r in enumerate(gt_rows):
        gt_cells = gt_r.get("cells", {})
        mapped = row_mapping.get(r_idx)
        ext_r = extracted_rows[mapped] if mapped is not None else []

        for c_idx, col_def in enumerate(expected_columns):
            gt_val = gt_cells.get(col_def["name"], "")
            total += 1
            is_num = col_def["type"] in ("integer", "decimal", "percentage") or parse_numeric(gt_val) is not None
            if is_num:
                numeric_total += 1

            if mapped is None:
                missing += 1
                continue

            ext_col = col_mapping.get(c_idx)
            if ext_col is None or ext_col >= len(ext_r):
                missing += 1
                continue

            comp = compare_cell_values(gt_val, ext_r[ext_col], col_def["type"])
            if comp["is_correct"]:
                correct += 1
                if is_num:
                    numeric_correct += 1
            else:
                incorrect += 1
                if get_cell_semantic(gt_val) == "EMPTY" and get_cell_semantic(ext_r[ext_col]) != "EMPTY":
                    hallucinated += 1

    return {
        "total_cells": total,
        "correct": correct,
        "incorrect": incorrect,
        "missing": missing,
        "cell_accuracy_pct": round(correct / max(1, total) * 100.0, 1),
        "numeric_accuracy_pct": round(numeric_correct / max(1, numeric_total) * 100.0, 1),
        "invented_cells": hallucinated,
        "column_semantic_pct": col_sem,
    }


def find_doc(name: str) -> Optional[Path]:
    return next((d / name for d in SEARCH_DIRS if (d / name).exists()), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--docs", type=int, default=0, help="limit number of documents (0 = all)")
    args = ap.parse_args()

    gt_dir = Path("tests/ground_truth")
    targets = GT_FILES[: args.docs] if args.docs else GT_FILES

    print("=" * 104)
    print(f"VLM ARCHITECTURE SPIKE - model={args.model}")
    print("Scored with the production benchmark's own comparison helpers.")
    print("=" * 104)

    results = []
    for gt_f in targets:
        gt_path = gt_dir / gt_f
        if not gt_path.exists():
            continue
        gt_data = json.loads(gt_path.read_text(encoding="utf-8"))
        doc_name = gt_data["document"]
        doc_file = find_doc(doc_name)
        if not doc_file:
            print(f"[SKIP] file not found: {doc_name}")
            continue
        if doc_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            print(f"[SKIP] not an image: {doc_name}")
            continue

        cols = [c["name"] for c in gt_data["columns"]]
        print(f"\n---> {doc_name}  ({len(gt_data['rows'])} GT rows, {len(cols)} cols)")
        try:
            raw, elapsed = call_vlm(args.model, doc_file, build_prompt(cols))
        except Exception as e:
            print(f"     VLM call failed: {type(e).__name__}: {e}")
            continue

        rows = parse_rows(raw)
        if not rows:
            print(f"     Unparseable reply ({elapsed:.0f}s). First 200 chars: {raw[:200]!r}")
            continue

        s = score(gt_data, rows)
        s["document"] = doc_name
        s["elapsed_s"] = round(elapsed, 1)
        s["rows_returned"] = len(rows)
        s["rows_expected"] = len(gt_data["rows"])
        results.append(s)
        print(
            f"     rows {s['rows_returned']}/{s['rows_expected']} | cell {s['cell_accuracy_pct']}% "
            f"| numeric {s['numeric_accuracy_pct']}% | invented {s['invented_cells']} | {s['elapsed_s']}s"
        )

    if not results:
        print("\nNo documents scored.")
        return 1

    print("\n" + "=" * 104)
    print(f"{'Document':24} {'VLM':>7} {'Pipeline':>9} {'Delta':>7} {'Rows':>9} {'Numeric':>8} {'Invented':>9} {'Time':>7}")
    print("-" * 104)
    for r in results:
        base = PIPELINE_BASELINE.get(r["document"])
        delta = f"{r['cell_accuracy_pct'] - base:+.1f}" if base is not None else "  n/a"
        base_s = f"{base:.1f}" if base is not None else "n/a"
        print(
            f"{r['document'][:24]:24} {r['cell_accuracy_pct']:6.1f}% {base_s:>9} {delta:>7} "
            f"{r['rows_returned']:>4}/{r['rows_expected']:<4} {r['numeric_accuracy_pct']:7.1f}% "
            f"{r['invented_cells']:>9} {r['elapsed_s']:6.0f}s"
        )

    avg_vlm = sum(r["cell_accuracy_pct"] for r in results) / len(results)
    scored = [r for r in results if r["document"] in PIPELINE_BASELINE]
    avg_pipe = sum(PIPELINE_BASELINE[r["document"]] for r in scored) / max(1, len(scored))
    print("-" * 104)
    print(f"{'AVERAGE':24} {avg_vlm:6.1f}% {avg_pipe:9.1f} {avg_vlm - avg_pipe:+7.1f}")
    print("\nInvented = cells the model filled where ground truth is blank (hallucination proxy).")
    print("Rows returned vs expected shows dropped/merged rows, the known VLM weakness on long tables.")

    out = Path("outputs/vlm_spike_results.json")
    out.write_text(json.dumps({"model": args.model, "results": results}, indent=2), encoding="utf-8")
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
