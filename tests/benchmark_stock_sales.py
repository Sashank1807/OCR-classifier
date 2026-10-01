"""
LEGACY / NON-AUTHORITATIVE BENCHMARK - do not use to judge current accuracy.

This script matches numbers by bag-of-numbers membership against combined
markdown/text output: it ignores which row or column a number came from, so a
value can score "correct" even if it landed under the wrong header entirely.
It predates comprehensive_cell_benchmark.py's semantic column mapping and
row-level cell comparison.

tests/comprehensive_cell_benchmark.py is the authoritative benchmark for all
pipeline changes going forward. Keep this file only for quick ad-hoc smoke
checks; do not compare its output to comprehensive_cell_benchmark.py's, and
do not use its historical numbers as evidence of current pipeline accuracy.
"""

import sys
import os
import re
import json
import time
import pathlib

# Add project root to sys.path
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, Any, List, Tuple
from app.services.ocr_pipeline import pipeline
from app.core.database import SessionLocal
from tests.ground_truth.dataset import GROUND_TRUTH_DATA


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes the Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def extract_numbers_from_text(text: str) -> List[float]:
    """Extracts all float/integer numbers from a text string."""
    raw_nums = re.findall(r'-?\b\d+(?:\.\d+)?\b', text)
    nums = []
    for n in raw_nums:
        try:
            val = float(n)
            nums.append(val)
        except ValueError:
            pass
    return nums


import uuid

def evaluate_document(file_path: pathlib.Path, gt_info: Dict[str, Any]) -> Dict[str, Any]:
    """Executes OCR pipeline and calculates accuracy vs ground truth."""
    db = SessionLocal()
    start_t = time.time()
    stored_name = f"bench_{uuid.uuid4().hex[:8]}_{file_path.name}"
    try:
        res = pipeline.process_file(db, file_path, stored_name, requested_doc_type="SPREADSHEET")
    finally:
        db.close()
    elapsed = time.time() - start_t

    # markdown is the whole text now - plain_text was the same content with the
    # formatting stripped and is no longer published.
    combined_text = res.get("markdown", "")

    # 1. Distributor & Header Recognition
    distributor_gt = gt_info.get("distributor_name", "").upper()
    distributor_found = distributor_gt in combined_text.upper() if distributor_gt else True

    # 2. Numerical Values Match Rate
    target_numbers = gt_info.get("sample_numeric_values", [])
    extracted_numbers = extract_numbers_from_text(combined_text)

    matched_nums = 0
    for target in target_numbers:
        # Check if target is present in extracted numbers (allowing 0.01 floating point tolerance)
        if any(abs(target - ex) < 0.01 for ex in extracted_numbers):
            matched_nums += 1

    numeric_accuracy = (matched_nums / len(target_numbers) * 100.0) if target_numbers else 100.0

    # 3. Summary Financial Totals Match
    summary_gt = gt_info.get("summary_totals", {})
    matched_totals = 0
    for k, v in summary_gt.items():
        if isinstance(v, (int, float)):
            if any(abs(v - ex) < 0.05 for ex in extracted_numbers):
                matched_totals += 1
    summary_accuracy = (matched_totals / len(summary_gt) * 100.0) if summary_gt else 100.0

    # 4. Overall Estimated OCR Character Accuracy
    # Rough estimate based on key terms and numerical precision
    combined_accuracy = (numeric_accuracy * 0.7) + (summary_accuracy * 0.2) + ((100.0 if distributor_found else 50.0) * 0.1)

    return {
        "file_name": file_path.name,
        "doc_type": gt_info["doc_type"],
        "processing_time_sec": round(elapsed, 2),
        "distributor_found": distributor_found,
        "total_target_numbers": len(target_numbers),
        "matched_numbers": matched_nums,
        "numeric_accuracy_pct": round(numeric_accuracy, 2),
        "summary_totals_accuracy_pct": round(summary_accuracy, 2),
        "overall_accuracy_pct": round(combined_accuracy, 2),
        "extracted_markdown": combined_text
    }


def run_benchmark():
    suite_dir = pathlib.Path("samples/benchmark_suite")
    results = []
    print("=" * 80)
    print("STARTING PHARMACEUTICAL STOCKIST SALES OCR BENCHMARK SUITE")
    print("=" * 80)

    for file_name, gt_info in GROUND_TRUTH_DATA.items():
        file_path = suite_dir / file_name
        if not file_path.exists():
            print(f"[SKIP] File not found: {file_path}")
            continue

        print(f"\n---> Benchmarking: {file_name} ({gt_info['doc_type']})")
        eval_result = evaluate_document(file_path, gt_info)
        results.append(eval_result)
        print(f"     Processing Time: {eval_result['processing_time_sec']}s")
        print(f"     Distributor Extracted: {'YES' if eval_result['distributor_found'] else 'NO'}")
        print(f"     Numerical Accuracy: {eval_result['numeric_accuracy_pct']}% ({eval_result['matched_numbers']}/{eval_result['total_target_numbers']} matched)")
        print(f"     Summary Totals Accuracy: {eval_result['summary_totals_accuracy_pct']}%")
        print(f"     Overall Estimated Accuracy: {eval_result['overall_accuracy_pct']}%")

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS TABLE")
    print("=" * 80)
    print(f"{'Document File':<38} | {'Type':<25} | {'Numeric Acc':<12} | {'Overall Acc':<12} | {'Time (s)':<8}")
    print("-" * 105)
    for r in results:
        print(f"{r['file_name']:<38} | {r['doc_type'][:25]:<25} | {r['numeric_accuracy_pct']:>10.2f}% | {r['overall_accuracy_pct']:>10.2f}% | {r['processing_time_sec']:>8.2f}")

    # Save results to json for reporting
    with open("outputs/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nBenchmark results saved to outputs/benchmark_results.json")


if __name__ == "__main__":
    run_benchmark()
