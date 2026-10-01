import pytest
from app.services.validation_service import validation_service
from app.utils.structured_schemas import create_semantic_cell


def test_validation_stock_arithmetic_verified():
    columns = ["Product Name", "Opening Qty", "Receipt Qty", "Issue Qty", "Closing Qty"]
    rows = [
        ["Paracetamol 500mg", "100.0", "50.0", "30.0", "120.0"],
        ["Amoxicillin 250mg", "50.0", "0.0", "20.0", "30.0"],
        ["Totals:", "150.0", "50.0", "50.0", "150.0"]
    ]

    res = validation_service.validate_stock_table(columns, rows, ocr_confidence=0.98, layout_confidence=0.95)

    assert res["arithmetic_checks"] == 2
    assert res["arithmetic_matches"] == 2
    assert res["arithmetic_mismatches"] == 0
    assert res["needs_manual_review"] is False
    assert res["overall_confidence"] >= 0.90

    # Check verified row data
    row0 = res["validated_rows"][0]
    assert row0["product_name"] == "Paracetamol 500mg"
    assert row0["calculated_closing_qty"] == 120.0
    assert row0["discrepancy"] == 0.0
    assert row0["validation_status"] == "VERIFIED"

    # Check totals validation
    totals = res["totals_validation"]
    assert totals["status"] == "VERIFIED"
    assert totals["column_sums"]["opening_qty"] == 150.0
    assert totals["printed_totals"]["closing_qty"] == 150.0


def test_validation_stock_arithmetic_mismatch():
    columns = ["Item Description", "Op. Qty", "Pur. Qty", "Sale Qty", "Bal. Qty"]
    rows = [
        # 10 + 5 - 2 should be 13, but OCR says 8 (column shift or misread)
        ["Cough Syrup 100ml", "10", "5", "2", "8"]
    ]

    res = validation_service.validate_stock_table(columns, rows, ocr_confidence=0.90, layout_confidence=0.85)

    assert res["arithmetic_checks"] == 1
    assert res["arithmetic_matches"] == 0
    assert res["arithmetic_mismatches"] == 1
    assert res["needs_manual_review"] is True

    row0 = res["validated_rows"][0]
    assert row0["validation_status"] == "MISMATCH"
    assert row0["calculated_closing_qty"] == 13.0
    assert row0["discrepancy"] == 5.0
    # Crucial golden rule: raw cells must NEVER be mutated!
    assert row0["raw_cells"] == ["Cough Syrup 100ml", "10", "5", "2", "8"]


def test_semantic_cell_classification():
    # Empty
    c_empty = create_semantic_cell("")
    assert c_empty["semantic"] == "EMPTY"
    assert c_empty["normalized"] is None

    c_none = create_semantic_cell(None)
    assert c_none["semantic"] == "EMPTY"

    # Not reported (dash)
    c_dash = create_semantic_cell("-")
    assert c_dash["semantic"] == "NOT_REPORTED"
    assert c_dash["raw"] == "-"

    # Explicit Zero
    c_zero = create_semantic_cell("0")
    assert c_zero["semantic"] == "ZERO"
    assert c_zero["normalized"] == 0.0

    c_zero_dec = create_semantic_cell("0.00")
    assert c_zero_dec["semantic"] == "ZERO"
    assert c_zero_dec["normalized"] == 0.0

    # Numeric
    c_num = create_semantic_cell("1,250.50")
    assert c_num["semantic"] == "NUMERIC"
    assert c_num["normalized"] == 1250.50

    # Text
    c_text = create_semantic_cell("Bilastine 20mg Tab")
    assert c_text["semantic"] == "TEXT"
    assert c_text["normalized"] == "Bilastine 20mg Tab"
