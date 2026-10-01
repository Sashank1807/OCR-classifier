"""
Comprehensive Production Cell-Level Benchmarking Engine
Evaluates:
- Physical lines, Logical rows, Product rows
- Strict Column Count & Semantic Order Mapping
- Cell-level Exact String, Normalized Text, Numeric, Blank, Zero, Dash Accuracy
- Deterministic Validation Consistency
- Multi-Factor Confidence Verification
- Standardized Production Status Taxonomy
"""

import sys
import os
import re
import json
import time
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.database import SessionLocal
from app.services.ocr_pipeline import pipeline
from app.services.table_ocr_service import table_ocr_service

# Bump this string whenever a pipeline change intended to affect OCR/alignment
# accuracy lands, so results from different phases/revisions are never
# compared as if they were runs of the same code (see OCR_ENGINEERING_CONTEXT
# handoff doc section 13.3 - historical milestone numbers across different
# code states are not directly comparable).
PIPELINE_REVISION = "phase0-benchmark-integrity-fix"


def normalize_str(s: str) -> str:
    """Normalizes string by stripping special chars, extra whitespace, lowercasing, and normalizing letter-digit transitions."""
    if not s:
        return ""
    # Normalize unicode dashes and quotes
    clean = s.replace("\u4e00", "-").replace("—", "-").replace("–", "-")
    # Normalize OCR digit-letter confusions inside product names
    clean = re.sub(r'(\d+)[oO]\b', r'\g<1>0', clean)
    clean = re.sub(r'\b[iIlL]oog\b', '100g', clean, flags=re.IGNORECASE)
    # Normalize alphanumeric boundaries where dot-matrix pin dropouts fused words (e.g. BILASET40MG -> BILASET 40 MG)
    clean = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', clean)
    clean = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', clean)
    clean = re.sub(r'(\d)%', r'\1 %', clean)
    clean = re.sub(r'%([a-zA-Z])', r'% \1', clean)
    clean = re.sub(r'([a-zA-Z]{3,})(tab|tabs|cap|caps|gel|oint|cream|lotion|solution|susp|inj|drops|syrup)\b', r'\1 \2', clean, flags=re.IGNORECASE)
    clean = re.sub(r'\b([a-zA-Z]+)(sb)(\d+)\b', r'\1 \2 \3', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[\s\-_.,/()\'"]+', ' ', clean).strip().lower()
    return clean



def parse_numeric(s: str) -> Optional[float]:
    """Extracts numeric float value if cell is numeric or percentage."""
    if not s:
        return None
    clean = s.replace(",", "").replace("$", "").replace("₹", "").replace("%", "").strip()
    clean = clean.replace("\u4e00", "-").replace("—", "-")
    if clean in ["-", "--", "---", "na", "n/a", ""]:
        return None
    try:
        return float(clean)
    except ValueError:
        m = re.search(r'^-?\d+(?:\.\d+)?', clean)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
        return None


def get_cell_semantic(val: str) -> str:
    """Classifies cell semantic category."""
    v = (val or "").strip()
    if not v:
        return "EMPTY"
    if v in ["-", "--", "---", "一", "—", "NA", "N/A"]:
        return "NOT_REPORTED"
    if v in ["0", "0.0", "0.00"]:
        return "ZERO"
    if parse_numeric(v) is not None:
        return "NUMERIC"
    return "TEXT"


def compare_cell_values(gt_val: str, ocr_val: str, col_type: str = "text") -> Dict[str, Any]:
    """
    Compares Ground Truth cell value vs Extracted OCR cell value.
    Returns detailed match breakdown.
    """
    gt_clean = (gt_val or "").strip()
    ocr_clean = (ocr_val or "").strip()

    gt_sem = get_cell_semantic(gt_clean)
    ocr_sem = get_cell_semantic(ocr_clean)

    # 1. Exact match
    exact_match = (gt_clean == ocr_clean)

    # 2. Normalized match
    norm_gt = normalize_str(gt_clean)
    norm_ocr = normalize_str(ocr_clean)
    normalized_match = (norm_gt == norm_ocr) if (norm_gt or norm_ocr) else (gt_sem == ocr_sem)

    # 3. Numeric match
    gt_num = parse_numeric(gt_clean)
    ocr_num = parse_numeric(ocr_clean)
    numeric_match = False
    if gt_num is not None and ocr_num is not None:
        numeric_match = abs(gt_num - ocr_num) <= 0.02
    elif gt_sem == ocr_sem and gt_sem in ["EMPTY", "NOT_REPORTED", "ZERO"]:
        numeric_match = True

    # 4. Blank / Dash / Zero preservation
    blank_preserved = (gt_sem == "EMPTY" and ocr_sem == "EMPTY")
    dash_preserved = (gt_sem == "NOT_REPORTED" and ocr_sem == "NOT_REPORTED")
    zero_preserved = (gt_sem == "ZERO" and ocr_sem in ["ZERO", "NUMERIC"] and ocr_num == 0.0)

    # Overall correctness determination
    is_correct = False
    if col_type in ["integer", "decimal", "percentage"]:
        is_correct = numeric_match or normalized_match
    elif gt_sem in ["EMPTY", "NOT_REPORTED", "ZERO"]:
        is_correct = (gt_sem == ocr_sem) or (gt_sem == "ZERO" and zero_preserved)
    else:
        is_correct = exact_match or normalized_match

    return {
        "is_correct": is_correct,
        "exact_match": exact_match,
        "normalized_match": normalized_match,
        "numeric_match": numeric_match,
        "gt_semantic": gt_sem,
        "ocr_semantic": ocr_sem,
        "blank_preserved": blank_preserved,
        "dash_preserved": dash_preserved,
        "zero_preserved": zero_preserved
    }


def map_columns_semantically(
    gt_cols: List[Dict[str, Any]],
    extracted_cols: List[str]
) -> Tuple[Dict[int, int], float]:
    """
    Maps extracted column indices to ground truth column indices.
    Returns mapping {gt_col_idx: ext_col_idx} and semantic accuracy percentage.
    """
    gt_map: Dict[int, int] = {}
    matched_gt_indices = set()
    used_ext_indices = set()

    # Pass 1: exact normalized match
    for gt_i, g_col in enumerate(gt_cols):
        g_name_norm = normalize_str(g_col["name"])
        for ext_i, ext_name in enumerate(extracted_cols):
            if ext_i in used_ext_indices:
                continue
            e_name_norm = normalize_str(ext_name)
            if g_name_norm == e_name_norm or (g_name_norm and g_name_norm in e_name_norm) or (e_name_norm and e_name_norm in g_name_norm):
                gt_map[gt_i] = ext_i
                matched_gt_indices.add(gt_i)
                used_ext_indices.add(ext_i)
                break

    # Pass 2: Positional alignment for unmapped columns if count matches
    if len(gt_cols) == len(extracted_cols):
        for gt_i in range(len(gt_cols)):
            if gt_i not in gt_map and gt_i not in used_ext_indices:
                gt_map[gt_i] = gt_i
                matched_gt_indices.add(gt_i)
                used_ext_indices.add(gt_i)

    semantic_accuracy = (len(matched_gt_indices) / max(1, len(gt_cols))) * 100.0
    return gt_map, round(semantic_accuracy, 2)


def _find_description_column(expected_columns: List[Dict[str, Any]]) -> Optional[int]:
    """Locates the GT column index most likely to hold a row-identifying description."""
    for i, c in enumerate(expected_columns):
        name_l = str(c.get("name", "")).lower()
        if any(k in name_l for k in ["desc", "particular", "product", "item", "name"]):
            return i
    for i, c in enumerate(expected_columns):
        if c.get("type") == "text":
            return i
    return 0 if expected_columns else None


def align_rows_to_ground_truth(
    gt_rows: List[Dict[str, Any]],
    extracted_rows: List[List[str]],
    expected_columns: List[Dict[str, Any]],
    col_mapping: Dict[int, int]
) -> Dict[int, Optional[int]]:
    """
    Maps each GT row index to the extracted row index that actually corresponds to it,
    using order-preserving description similarity, instead of assuming index r == index r.

    A single dropped/duplicated/extra row anywhere before the end of the table would
    otherwise cascade into every subsequent GT row being compared against the wrong
    extracted row, collapsing the whole document's score even when extraction is
    substantively correct. Falls back to identity (today's behavior) when row counts
    already match, so already-correct documents are scored identically to before.
    """
    n, m = len(gt_rows), len(extracted_rows)
    if n == m:
        return {i: i for i in range(n)}
    if n == 0:
        return {}
    if m == 0:
        return {i: None for i in range(n)}

    desc_gt_idx = _find_description_column(expected_columns)
    desc_ext_idx = col_mapping.get(desc_gt_idx) if desc_gt_idx is not None else None

    def gt_desc(i: int) -> str:
        if desc_gt_idx is None:
            return ""
        col_name = expected_columns[desc_gt_idx]["name"]
        return normalize_str(str(gt_rows[i].get("cells", {}).get(col_name, "")))

    def ext_desc(j: int) -> str:
        if desc_ext_idx is None or desc_ext_idx >= len(extracted_rows[j]):
            return ""
        return normalize_str(str(extracted_rows[j][desc_ext_idx]))

    def sim(i: int, j: int) -> float:
        a, b = gt_desc(i), ext_desc(j)
        if not a and not b:
            # Both blank (e.g. subtotal/party rows with no description overlap) -
            # neutral score, neither confidently a match nor a mismatch.
            return 0.5
        return SequenceMatcher(None, a, b).ratio()

    # Needleman-Wunsch style order-preserving alignment: row order is a meaningful
    # signal (rows don't get reshuffled), so this is sequence alignment, not free
    # assignment. cost(match) = 1 - similarity; a fixed gap cost handles
    # inserted/dropped rows.
    GAP = 0.7
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP
        back[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP
        back[0][j] = "left"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match_cost = dp[i - 1][j - 1] + (1.0 - sim(i - 1, j - 1))
            del_cost = dp[i - 1][j] + GAP
            ins_cost = dp[i][j - 1] + GAP
            best = min(match_cost, del_cost, ins_cost)
            dp[i][j] = best
            back[i][j] = "diag" if best == match_cost else ("up" if best == del_cost else "left")

    mapping: Dict[int, Optional[int]] = {}
    i, j = n, m
    while i > 0 or j > 0:
        move = back[i][j] if (i > 0 and j > 0) else ("up" if i > 0 else "left")
        if move == "diag":
            mapping[i - 1] = (j - 1) if sim(i - 1, j - 1) >= 0.35 else None
            i -= 1
            j -= 1
        elif move == "up":
            mapping[i - 1] = None
            i -= 1
        else:
            j -= 1

    return mapping


def evaluate_document_cell_benchmark(
    gt_path: Path,
    file_path: Path
) -> Dict[str, Any]:
    """
    Executes full pipeline and performs comprehensive cell-level comparison against frozen ground truth.
    """
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    doc_name = gt_data["document"]
    expected_logical_rows = gt_data["logical_row_count"]
    expected_product_rows = gt_data["product_row_count"]
    expected_columns = gt_data["columns"]
    gt_rows = gt_data["rows"]

    # A GT fixture that declares rows but ships an empty "rows" array is an
    # incomplete fixture, not a real 0%/100% measurement. Score it as
    # unscorable rather than letting it silently pollute results with a
    # count-only vacuous pass/fail.
    gt_incomplete = (not gt_rows) and (expected_logical_rows > 0 or expected_product_rows > 0)

    db = SessionLocal()
    start_t = time.time()
    try:
        res = pipeline.process_file(
            db=db,
            file_path=file_path,
            original_filename=file_path.name,
            requested_doc_type=gt_data.get("document_type", "AUTO")
        )
    finally:
        db.close()
    elapsed = round(time.time() - start_t, 2)

    # Extract primary table from response
    envelope = res.get("structured_data", {})
    envelope_fields = envelope.get("fields", {}) or res.get("fields", {}) or envelope
    all_tables = res.get("tables", [])

    if res.get("document_type") in ["PAN", "AADHAAR"]:
        extracted_columns = ["field_name", "field_value"]
        # Match field_name keys semantically from structured envelope fields
        extracted_rows = []
        for gt_r in gt_rows:
            f_key = gt_r.get("cells", {}).get("field_name", "")
            f_val = envelope_fields.get(f_key, "")
            extracted_rows.append([f_key, str(f_val) if f_val is not None else ""])
    elif all_tables and isinstance(all_tables[0], dict) and all_tables[0].get("rows"):
        extracted_columns = all_tables[0].get("columns", [])
        extracted_rows = all_tables[0].get("rows", [])
    elif envelope_fields.get("rows"):
        extracted_columns = envelope_fields.get("columns", [])
        extracted_rows = envelope_fields.get("rows", [])
    else:
        tbl_res = table_ocr_service.extract_table(file_path)
        if tbl_res.get("tables"):
            extracted_columns = tbl_res["tables"][0].get("columns", [])
            extracted_rows = tbl_res["tables"][0].get("rows", [])
        else:
            extracted_columns = []
            extracted_rows = []

    extracted_row_count = len(extracted_rows)
    extracted_col_count = len(extracted_columns)

    if gt_incomplete:
        reason = (
            f"Ground truth declares {expected_logical_rows} logical rows "
            f"({expected_product_rows} product rows) but 'rows' is empty in {gt_path.name} - "
            f"fixture needs backfilling before this document can be scored."
        )
        return {
            "document": doc_name,
            "document_type_detected": res.get("document_type", "UNKNOWN"),
            "expected_logical_rows": expected_logical_rows,
            "extracted_logical_rows": extracted_row_count,
            "expected_columns": len(expected_columns),
            "extracted_columns": extracted_col_count,
            "elapsed_seconds": elapsed,
            "unscorable": True,
            "reason": reason,
            "status_tags": ["UNSCORABLE_GT_INCOMPLETE"],
            "overall_status": "UNSCORABLE_GT_INCOMPLETE",
            "production_pass": False,
            "extracted_rows": extracted_rows
        }

    # Calculate row and column counts accuracy
    row_accuracy = min(extracted_row_count, expected_logical_rows) / max(1, max(extracted_row_count, expected_logical_rows)) * 100.0
    col_accuracy = min(extracted_col_count, len(expected_columns)) / max(1, max(extracted_col_count, len(expected_columns))) * 100.0

    # Column Semantic Accuracy
    col_mapping, col_semantic_acc = map_columns_semantically(expected_columns, extracted_columns)

    # Logical row alignment (order-preserving, similarity-based - not raw index)
    row_mapping = align_rows_to_ground_truth(gt_rows, extracted_rows, expected_columns, col_mapping)

    # Cell-Level Evaluation
    total_expected_cells = 0
    correct_cells = 0
    incorrect_cells = 0
    missing_cells = 0
    extra_cells = 0

    gt_non_empty_cells = 0
    ocr_non_empty_cells = 0
    correct_non_empty_cells = 0
    false_empty_cells = 0
    false_dash_cells = 0

    numeric_total = 0
    numeric_correct = 0
    numeric_extracted_total = 0
    numeric_correct_extracted = 0

    blank_total = 0
    blank_correct = 0
    dash_total = 0
    dash_correct = 0
    zero_total = 0
    zero_correct = 0

    cell_failures = []
    debug_images_exported = 0

    # Load image for cell debug visualization if available
    img_cv = None
    if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        import cv2
        from app.services.cell_ocr_service import cell_ocr_service
        img_cv = cv2.imread(str(file_path))

    # Iterate over ground truth rows
    for r_idx, gt_r in enumerate(gt_rows):
        gt_cells = gt_r.get("cells", {})
        mapped_ext_idx = row_mapping.get(r_idx)
        has_ext_row = mapped_ext_idx is not None
        ext_r = extracted_rows[mapped_ext_idx] if has_ext_row else []

        for c_idx, col_def in enumerate(expected_columns):
            c_name = col_def["name"]
            c_type = col_def["type"]
            gt_val = gt_cells.get(c_name, "")
            total_expected_cells += 1
            gt_sem = get_cell_semantic(gt_val)
            if gt_sem != "EMPTY":
                gt_non_empty_cells += 1

            is_numeric_cell = (c_type in ["integer", "decimal", "percentage"] or parse_numeric(gt_val) is not None)
            if is_numeric_cell:
                numeric_total += 1

            if not has_ext_row:
                missing_cells += 1
                if gt_sem != "EMPTY":
                    false_empty_cells += 1
                cell_failures.append({
                    "row": r_idx + 1,
                    "column": c_name,
                    "gt_val": gt_val,
                    "ocr_val": "<MISSING_ROW>",
                    "reason": "Extracted row is missing"
                })
                continue

            # Find matching extracted column
            ext_col_idx = col_mapping.get(c_idx)
            if ext_col_idx is None or ext_col_idx >= len(ext_r):
                missing_cells += 1
                if gt_sem != "EMPTY":
                    false_empty_cells += 1
                cell_failures.append({
                    "row": r_idx + 1,
                    "column": c_name,
                    "gt_val": gt_val,
                    "ocr_val": "<MISSING_COLUMN>",
                    "reason": "Column unmapped or out of bounds"
                })
                continue

            ocr_val = ext_r[ext_col_idx]
            comp = compare_cell_values(gt_val, ocr_val, c_type)
            ocr_sem = comp["ocr_semantic"]
            ocr_clean = (ocr_val or "").strip()

            if ocr_sem != "EMPTY":
                ocr_non_empty_cells += 1

            ocr_has_num = (ocr_sem in ["NUMERIC", "ZERO"] or parse_numeric(ocr_clean) is not None)
            if is_numeric_cell and ocr_has_num:
                numeric_extracted_total += 1

            # Check False Empty (GT had ink/content, but OCR extracted empty)
            if gt_sem != "EMPTY" and ocr_sem == "EMPTY":
                false_empty_cells += 1

            # Check False Dash (GT was NOT a dash, but OCR extracted a dash)
            if gt_sem != "NOT_REPORTED" and ocr_sem == "NOT_REPORTED":
                false_dash_cells += 1

            # Track Blank / Dash / Zero preservation
            if comp["gt_semantic"] == "EMPTY":
                blank_total += 1
                if comp["blank_preserved"]:
                    blank_correct += 1
            elif comp["gt_semantic"] == "NOT_REPORTED":
                dash_total += 1
                if comp["dash_preserved"]:
                    dash_correct += 1
            elif comp["gt_semantic"] == "ZERO":
                zero_total += 1
                if comp["zero_preserved"]:
                    zero_correct += 1

            if is_numeric_cell:
                if comp["numeric_match"] or comp["is_correct"]:
                    numeric_correct += 1
                    if ocr_has_num:
                        numeric_correct_extracted += 1

            if comp["is_correct"]:
                correct_cells += 1
                if gt_sem != "EMPTY":
                    correct_non_empty_cells += 1
            else:
                incorrect_cells += 1
                cell_failures.append({
                    "row": r_idx + 1,
                    "column": c_name,
                    "gt_val": gt_val,
                    "ocr_val": ocr_val,
                    "reason": f"Mismatch (GT: '{gt_val}' vs OCR: '{ocr_val}')"
                })

                # Export debug visualization for review-required cells (up to 5 per doc)
                if img_cv is not None and debug_images_exported < 5:
                    from app.services.cell_ocr_service import cell_ocr_service
                    # Attempt to find bounding box from cells_metadata if available
                    cell_bbox = (20, 20 + r_idx * 30, 150, 45 + r_idx * 30)
                    if all_tables and all_tables[0].get("cells_metadata") and mapped_ext_idx is not None:
                        meta_grid = all_tables[0]["cells_metadata"]
                        if mapped_ext_idx < len(meta_grid) and ext_col_idx < len(meta_grid[mapped_ext_idx]):
                            toks = meta_grid[mapped_ext_idx][ext_col_idx].get("tokens", [])
                            if toks:
                                cell_bbox = (
                                    int(min(t["x0"] for t in toks)),
                                    int(min(t["y0"] for t in toks)),
                                    int(max(t["x1"] for t in toks)),
                                    int(max(t["y1"] for t in toks))
                                )
                    cell_ocr_service.export_cell_debug_image(
                        image=img_cv,
                        cell_bbox=cell_bbox,
                        doc_name=doc_name,
                        row_idx=r_idx,
                        col_idx=ext_col_idx,
                        col_name=c_name,
                        gt_val=gt_val,
                        ocr_val=ocr_val
                    )
                    debug_images_exported += 1

    # Extra cells if extracted has more rows
    if extracted_row_count > len(gt_rows):
        extra_cells = (extracted_row_count - len(gt_rows)) * max(1, extracted_col_count)

    cell_accuracy = (correct_cells / max(1, total_expected_cells)) * 100.0 if total_expected_cells > 0 else (100.0 if (row_accuracy >= 95.0 and col_accuracy >= 95.0) else 0.0)
    numeric_accuracy = (numeric_correct / max(1, numeric_total)) * 100.0 if numeric_total > 0 else 100.0
    blank_acc = (blank_correct / max(1, blank_total)) * 100.0 if blank_total > 0 else 100.0
    dash_acc = (dash_correct / max(1, dash_total)) * 100.0 if dash_total > 0 else 100.0
    zero_acc = (zero_correct / max(1, zero_total)) * 100.0 if zero_total > 0 else 100.0

    # Production Completeness & Quality Metrics
    cell_recall = (correct_non_empty_cells / max(1, gt_non_empty_cells)) * 100.0 if gt_non_empty_cells > 0 else 100.0
    cell_precision = (correct_non_empty_cells / max(1, ocr_non_empty_cells)) * 100.0 if ocr_non_empty_cells > 0 else 100.0
    false_empty_rate = (false_empty_cells / max(1, gt_non_empty_cells)) * 100.0 if gt_non_empty_cells > 0 else 0.0
    false_dash_rate = (false_dash_cells / max(1, total_expected_cells)) * 100.0 if total_expected_cells > 0 else 0.0
    numeric_recall = (numeric_correct / max(1, numeric_total)) * 100.0 if numeric_total > 0 else 100.0
    numeric_precision = (numeric_correct_extracted / max(1, numeric_extracted_total)) * 100.0 if numeric_extracted_total > 0 else 100.0

    # Strict Production Pass Criteria (Row>=95%, Col>=95%, ColSemantic>=95%, Cell>=95%, Num>=98%)
    layout_pass = (row_accuracy >= 95.0 and col_accuracy >= 95.0 and col_semantic_acc >= 95.0)
    cell_pass = (cell_accuracy >= 95.0)
    numeric_pass = (numeric_accuracy >= 98.0)
    
    validation_status = "N/A"
    audit = envelope.get("_audit", {})
    val_info = audit.get("validation", {})
    if res.get("document_type") in ["STOCK_STATEMENT", "SPREADSHEET"]:
        mismatches = val_info.get("arithmetic_mismatches", 0)
        validation_pass = (mismatches == 0)
        validation_status = "VALIDATION_PASS" if validation_pass else f"DISCREPANCY ({mismatches} mismatches)"
    else:
        validation_pass = True
        validation_status = "VALID_ID"

    overall_confidence = float(envelope.get("overall_confidence", res.get("confidence", 0.0)))
    needs_manual_review = bool(envelope.get("needs_manual_review", False))

    production_pass = (
        layout_pass and cell_pass and numeric_pass and validation_pass and (not needs_manual_review) and overall_confidence >= 0.85
    )

    status_tags = []
    status_tags.append("LAYOUT_PASS" if layout_pass else "LAYOUT_FAIL")
    status_tags.append("CELL_PASS" if cell_pass else "CELL_FAIL")
    status_tags.append("NUMERIC_PASS" if numeric_pass else "NUMERIC_FAIL")
    status_tags.append("VALIDATION_PASS" if validation_pass else "VALIDATION_FAIL")
    if production_pass:
        status_tags.append("PRODUCTION_PASS")
    else:
        status_tags.append("REVIEW_REQUIRED")

    return {
        "document": doc_name,
        "document_type_detected": res.get("document_type", "UNKNOWN"),
        "expected_logical_rows": expected_logical_rows,
        "extracted_logical_rows": extracted_row_count,
        "row_accuracy_pct": round(row_accuracy, 2),
        "expected_columns": len(expected_columns),
        "extracted_columns": extracted_col_count,
        "column_accuracy_pct": round(col_accuracy, 2),
        "column_semantic_accuracy_pct": col_semantic_acc,
        "total_expected_cells": total_expected_cells,
        "correct_cells": correct_cells,
        "incorrect_cells": incorrect_cells,
        "missing_cells": missing_cells,
        "extra_cells": extra_cells,
        "gt_non_empty_cells": gt_non_empty_cells,
        "ocr_non_empty_cells": ocr_non_empty_cells,
        "correct_non_empty_cells": correct_non_empty_cells,
        "false_empty_cells": false_empty_cells,
        "false_dash_cells": false_dash_cells,
        "cell_accuracy_pct": round(cell_accuracy, 2),
        "cell_recall_pct": round(cell_recall, 2),
        "cell_precision_pct": round(cell_precision, 2),
        "false_empty_rate_pct": round(false_empty_rate, 2),
        "false_dash_rate_pct": round(false_dash_rate, 2),
        "numeric_accuracy_pct": round(numeric_accuracy, 2),
        "numeric_recall_pct": round(numeric_recall, 2),
        "numeric_precision_pct": round(numeric_precision, 2),
        "blank_accuracy_pct": round(blank_acc, 2),
        "dash_accuracy_pct": round(dash_acc, 2),
        "zero_accuracy_pct": round(zero_acc, 2),
        "layout_status": "LAYOUT_PASS" if layout_pass else "LAYOUT_FAIL",
        "cell_ocr_status": "CELL_PASS" if cell_pass else "CELL_FAIL",
        "numeric_status": "NUMERIC_PASS" if numeric_pass else "NUMERIC_FAIL",
        "validation_status": "VALIDATION_PASS" if validation_pass else "VALIDATION_FAIL",
        "overall_confidence": overall_confidence,
        "needs_manual_review": needs_manual_review,
        "elapsed_seconds": elapsed,
        "status_tags": status_tags,
        "production_pass": production_pass,
        "overall_status": "PRODUCTION_PASS" if production_pass else "REVIEW_REQUIRED",
        "failures": cell_failures,
        "extracted_rows": extracted_rows
    }



def run_comprehensive_benchmark(output_path: str = "outputs/cell_benchmark_results.json"):
    """Runs benchmark across all frozen ground truth documents."""
    gt_dir = Path("tests/ground_truth")
    test_data_dir = Path("test_data_june")
    samples_dir = Path("samples/benchmark_suite")
    uploads_dir = Path("uploads")

    all_gt_files = [
        "1000411295.json",
        "1000411296.json",
        "1000517666.json",
        "1000517796.json",
        "agarwal_jaipur.json",
        "bansal_barelly.json",
        "anshul_sikar.json",
        "saraswati_drug.json",
        "hetroder.json",
        "pan_card.json",
        "aadhaar_card.json"
    ]

    print("=" * 110)
    print("STARTING COMPREHENSIVE PRODUCTION CELL-LEVEL & GEOMETRY BENCHMARK")
    print("=" * 110)

    results = []

    for gt_f in all_gt_files:
        gt_path = gt_dir / gt_f
        if not gt_path.exists():
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            gt_info = json.load(f)

        target_doc = gt_info["document"]
        # Locate target file
        cand_paths = [
            test_data_dir / target_doc,
            samples_dir / target_doc,
            uploads_dir / target_doc
        ]
        doc_file = next((p for p in cand_paths if p.exists()), None)

        if not doc_file:
            print(f"[SKIP] Document file not found: {target_doc}")
            continue

        print(f"\n---> Benchmarking: {target_doc} ({gt_info['document_type']})")
        report = evaluate_document_cell_benchmark(gt_path, doc_file)
        results.append(report)

        if report.get("unscorable"):
            print(f"     UNSCORABLE_GT_INCOMPLETE: {report.get('reason', '')}")
            continue

        print(f"     Rows (Exp/Ext): {report['expected_logical_rows']} / {report['extracted_logical_rows']} ({report['row_accuracy_pct']}%)")
        print(f"     Cols (Exp/Ext): {report['expected_columns']} / {report['extracted_columns']} ({report['column_accuracy_pct']}%) | Semantic: {report['column_semantic_accuracy_pct']}%")
        print(f"     Cells: Total {report['total_expected_cells']} | Correct {report['correct_cells']} | Wrong {report['incorrect_cells']} | Missing {report['missing_cells']}")
        print(f"     Cell Accuracy: {report['cell_accuracy_pct']}% | Recall: {report['cell_recall_pct']}% | Precision: {report['cell_precision_pct']}%")
        print(f"     Numeric Accuracy: {report['numeric_accuracy_pct']}% | Num Recall: {report['numeric_recall_pct']}% | Num Precision: {report['numeric_precision_pct']}%")
        print(f"     False Empty Rate: {report['false_empty_rate_pct']}% | False Dash Rate: {report['false_dash_rate_pct']}%")
        print(f"     Confidence: {report['overall_confidence']} | Review Required: {report['needs_manual_review']}")
        print(f"     STATUS: {' | '.join(report['status_tags'])}")

    print("\n" + "=" * 124)
    print(f"{'Document':<22} | {'Row Acc':<7} | {'Col Acc':<7} | {'Cell Acc':<8} | {'Recall':<7} | {'F-Empty':<7} | {'Num Acc':<7} | {'Status'}")
    print("-" * 124)
    scorable_results = [r for r in results if not r.get("unscorable")]
    unscorable_results = [r for r in results if r.get("unscorable")]
    for r in scorable_results:
        status_summary = "PRODUCTION_PASS" if r["production_pass"] else "REVIEW_REQUIRED"
        print(f"{r['document'][:22]:<22} | {r['row_accuracy_pct']:>6.1f}% | {r['column_accuracy_pct']:>6.1f}% | {r['cell_accuracy_pct']:>7.1f}% | {r['cell_recall_pct']:>6.1f}% | {r['false_empty_rate_pct']:>6.1f}% | {r['numeric_accuracy_pct']:>6.1f}% | {status_summary}")
    for r in unscorable_results:
        print(f"{r['document'][:22]:<22} | {'--':>6}  | {'--':>6}  | {'--':>7}  | {'--':>6}  | {'--':>6}  | {'--':>6}  | UNSCORABLE_GT_INCOMPLETE")
    if unscorable_results:
        print(f"\n({len(unscorable_results)} document(s) excluded from scoring above due to incomplete ground truth - see UNSCORABLE_GT_INCOMPLETE entries.)")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_payload = {
        "pipeline_revision": PIPELINE_REVISION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed benchmark report saved to {output_path} (revision: {PIPELINE_REVISION})")

    return results


if __name__ == "__main__":
    run_comprehensive_benchmark()
