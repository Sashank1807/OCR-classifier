import pytest
from tests.comprehensive_cell_benchmark import (
    normalize_str,
    parse_numeric,
    get_cell_semantic,
    compare_cell_values,
    map_columns_semantically,
    align_rows_to_ground_truth
)


def test_cell_semantic_classification():
    assert get_cell_semantic("") == "EMPTY"
    assert get_cell_semantic("   ") == "EMPTY"
    assert get_cell_semantic("-") == "NOT_REPORTED"
    assert get_cell_semantic("--") == "NOT_REPORTED"
    assert get_cell_semantic("一") == "NOT_REPORTED"  # Chinese dash from RapidOCR
    assert get_cell_semantic("NA") == "NOT_REPORTED"
    assert get_cell_semantic("0") == "ZERO"
    assert get_cell_semantic("0.00") == "ZERO"
    assert get_cell_semantic("123.45") == "NUMERIC"
    assert get_cell_semantic("BILASET 40MG TAB") == "TEXT"


def test_cell_comparison_exact_and_normalized():
    # Exact text match
    res1 = compare_cell_values("BILASET 40MG TAB", "BILASET 40MG TAB", "text")
    assert res1["is_correct"] is True
    assert res1["exact_match"] is True

    # Normalized text match (minor whitespace/case differences)
    res2 = compare_cell_values("BILASET 40MG TAB", "bilaset 40mg  tab", "text")
    assert res2["is_correct"] is True
    assert res2["normalized_match"] is True

    # Numeric match with decimal tolerance
    res3 = compare_cell_values("114.09", "114.090", "decimal")
    assert res3["is_correct"] is True
    assert res3["numeric_match"] is True

    # Numeric match with currency symbol
    res4 = compare_cell_values("2281.76", "₹2,281.76", "decimal")
    assert res4["is_correct"] is True
    assert res4["numeric_match"] is True

    # Blank preservation
    res5 = compare_cell_values("", "", "text")
    assert res5["is_correct"] is True
    assert res5["blank_preserved"] is True

    # Dash preservation
    res6 = compare_cell_values("-", "一", "text")
    assert res6["is_correct"] is True
    assert res6["dash_preserved"] is True

    # Zero vs Blank should NOT be silently identical unless allowed
    res7 = compare_cell_values("0", "", "integer")
    assert res7["is_correct"] is False


def test_column_semantic_mapping():
    gt_cols = [
        {"name": "ITEM DESCRIPTION", "type": "text", "x_order": 1},
        {"name": "Packing", "type": "text", "x_order": 2},
        {"name": "OPENING", "type": "decimal", "x_order": 3},
        {"name": "RECEIPT", "type": "decimal", "x_order": 4},
        {"name": "ISSUE", "type": "decimal", "x_order": 5},
        {"name": "CLOSING", "type": "decimal", "x_order": 6}
    ]
    extracted_cols = ["Item Description", "Packing", "Opening", "Receipt", "Issue", "Closing"]
    mapping, acc = map_columns_semantically(gt_cols, extracted_cols)
    assert acc == 100.0
    assert len(mapping) == 6
    assert mapping[0] == 0
    assert mapping[5] == 5


def test_column_order_mismatch_fails_semantic_mapping():
    gt_cols = [
        {"name": "OPENING", "type": "decimal", "x_order": 1},
        {"name": "RECEIPT", "type": "decimal", "x_order": 2},
        {"name": "ISSUE", "type": "decimal", "x_order": 3},
        {"name": "CLOSING", "type": "decimal", "x_order": 4}
    ]
    # Swapped columns: Issue before Receipt
    extracted_cols = ["OPENING", "ISSUE", "RECEIPT", "CLOSING"]
    mapping, acc = map_columns_semantically(gt_cols, extracted_cols)
    # They should be mapped to the actual swapped indices
    assert mapping[1] == 2  # GT RECEIPT is mapped to extracted index 2
    assert mapping[2] == 1  # GT ISSUE is mapped to extracted index 1


def test_cell_completeness_and_recall_metrics():
    """Verifies that cell recall, precision, and false empty rate correctly evaluate completeness."""
    # Suppose 10 cells total:
    # 8 non-empty GT cells, 2 empty GT cells
    # OCR extracts 7 non-empty correctly, 1 non-empty as "" (false empty),
    # 2 empty GT correctly preserved as ""
    gt_vals = ["ITEM A", "10TAB", "100.00", "50.00", "20.00", "130.00", "0.00", "15", "", ""]
    ocr_vals = ["item a", "10TAB", "100.00", "50.00", "20.00", "130.00", "0.00", "", "", ""]
    col_types = ["text", "text", "decimal", "decimal", "decimal", "decimal", "decimal", "integer", "text", "text"]

    gt_non_empty = 0
    ocr_non_empty = 0
    correct_non_empty = 0
    false_empty = 0

    for gt_v, ocr_v, c_t in zip(gt_vals, ocr_vals, col_types):
        gt_sem = get_cell_semantic(gt_v)
        ocr_sem = get_cell_semantic(ocr_v)
        comp = compare_cell_values(gt_v, ocr_v, c_t)

        if gt_sem != "EMPTY":
            gt_non_empty += 1
        if ocr_sem != "EMPTY":
            ocr_non_empty += 1
        if gt_sem != "EMPTY" and ocr_sem == "EMPTY":
            false_empty += 1
        if comp["is_correct"] and gt_sem != "EMPTY":
            correct_non_empty += 1

    assert gt_non_empty == 8
    assert ocr_non_empty == 7
    assert false_empty == 1
    assert correct_non_empty == 7

    cell_recall = (correct_non_empty / gt_non_empty) * 100.0
    cell_precision = (correct_non_empty / ocr_non_empty) * 100.0
    false_empty_rate = (false_empty / gt_non_empty) * 100.0

    assert cell_recall == 87.5
    assert cell_precision == 100.0
    assert false_empty_rate == 12.5


def _make_gt_row(desc: str) -> dict:
    return {"cells": {"Description": desc}}


EXPECTED_COLUMNS = [
    {"name": "Description", "type": "text"},
    {"name": "Qty", "type": "integer"},
]


def test_row_alignment_survives_identity_case():
    """When counts already match, alignment must be identity (unchanged behavior)."""
    gt_rows = [_make_gt_row("PARACETAMOL 500MG"), _make_gt_row("AMOXICILLIN 250MG")]
    extracted_rows = [["PARACETAMOL 500MG", "10"], ["AMOXICILLIN 250MG", "20"]]
    col_mapping = {0: 0, 1: 1}
    mapping = align_rows_to_ground_truth(gt_rows, extracted_rows, EXPECTED_COLUMNS, col_mapping)
    assert mapping == {0: 0, 1: 1}


def test_row_alignment_survives_single_dropped_row_mid_table():
    """
    A single row dropped mid-table must not cascade into every subsequent row
    being compared against the wrong extracted row (the bug pure index-based
    matching had).
    """
    gt_rows = [
        _make_gt_row("PARACETAMOL 500MG"),
        _make_gt_row("AMOXICILLIN 250MG"),
        _make_gt_row("IBUPROFEN 400MG"),
        _make_gt_row("CETIRIZINE 10MG"),
    ]
    # "AMOXICILLIN 250MG" (GT row 1) was dropped by the OCR pipeline.
    extracted_rows = [
        ["PARACETAMOL 500MG", "10"],
        ["IBUPROFEN 400MG", "30"],
        ["CETIRIZINE 10MG", "40"],
    ]
    col_mapping = {0: 0, 1: 1}
    mapping = align_rows_to_ground_truth(gt_rows, extracted_rows, EXPECTED_COLUMNS, col_mapping)

    assert mapping[0] == 0  # PARACETAMOL still lines up
    assert mapping[1] is None  # AMOXICILLIN correctly reported as missing, not misaligned
    assert mapping[2] == 1  # IBUPROFEN correctly re-aligned to its real extracted row
    assert mapping[3] == 2  # CETIRIZINE correctly re-aligned, not cascaded off by one


def test_row_alignment_survives_single_extra_row_mid_table():
    """An extra/duplicate extracted row must not shift every later GT row's comparison."""
    gt_rows = [
        _make_gt_row("PARACETAMOL 500MG"),
        _make_gt_row("AMOXICILLIN 250MG"),
        _make_gt_row("IBUPROFEN 400MG"),
    ]
    # An extra noise row was inserted after the first product.
    extracted_rows = [
        ["PARACETAMOL 500MG", "10"],
        ["S-TOTAL", ""],
        ["AMOXICILLIN 250MG", "20"],
        ["IBUPROFEN 400MG", "30"],
    ]
    col_mapping = {0: 0, 1: 1}
    mapping = align_rows_to_ground_truth(gt_rows, extracted_rows, EXPECTED_COLUMNS, col_mapping)

    assert mapping[0] == 0
    assert mapping[1] == 2  # AMOXICILLIN correctly skips over the inserted noise row
    assert mapping[2] == 3

