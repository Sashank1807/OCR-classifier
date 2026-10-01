import pytest
import numpy as np
from pathlib import Path
from app.services.table_ocr_service import table_ocr_service
from app.services.document_analyzer import document_analyzer, normalize_erp_header_string
from app.services.validation_service import validation_service
from app.services.ocr_pipeline import pipeline


def test_erp_header_normalization_compact_stock():
    # Tests ERP header normalization stripping spaces, hyphens, and casing
    assert normalize_erp_header_string("STOCKSUMMARY") == "stocksummary"
    assert normalize_erp_header_string("STOCK & SALES ANALYSIS REPORT") == "stocksalesanalysisreport"
    assert normalize_erp_header_string("HETERO (DERMA GLOW) STOCK & SALES STATEMENT") == "heterodermaglowstocksalesstatement"

    # Verify compact token triggers classification as STOCK_STATEMENT
    res = document_analyzer.analyze(Path("test_data_june/1000411296.jpg"))
    assert res["document_type"] == "STOCK_STATEMENT"
    assert res["is_table_heavy"] is True
    assert res["recommended_pipeline"] == "COORDINATE_TABLE_PIPELINE"


def test_multi_tier_header_reconstruction_stacked():
    # Simulates stacked 2-tier header:
    # Line 1: [PRODUCT NAME] [PACKING] [OPENING] [RECEIPT] [ISSUE] [CLOSING]
    # Line 2:                          [QTY]     [QTY]     [QTY]   [QTY]
    hdr_lines = [
        {
            "y_avg": 100.0,
            "tokens": [
                {"text": "PRODUCT NAME", "x0": 50.0, "x1": 150.0},
                {"text": "PACKING", "x0": 170.0, "x1": 230.0},
                {"text": "OPENING", "x0": 260.0, "x1": 320.0},
                {"text": "RECEIPT", "x0": 340.0, "x1": 400.0},
                {"text": "ISSUE", "x0": 420.0, "x1": 480.0},
                {"text": "CLOSING", "x0": 500.0, "x1": 560.0},
            ]
        },
        {
            "y_avg": 120.0,
            "tokens": [
                {"text": "QTY", "x0": 270.0, "x1": 310.0},
                {"text": "QTY", "x0": 350.0, "x1": 390.0},
                {"text": "QTY", "x0": 430.0, "x1": 470.0},
                {"text": "QTY", "x0": 510.0, "x1": 550.0},
            ]
        }
    ]

    cols = table_ocr_service._synthesize_columns_from_band(hdr_lines, [])
    assert len(cols) == 6
    assert "PRODUCT NAME" in cols[0]["name"]
    assert "PACKING" in cols[1]["name"]
    assert "OPENING" in cols[2]["name"]
    assert "RECEIPT" in cols[3]["name"]
    assert "ISSUE" in cols[4]["name"]
    assert "CLOSING" in cols[5]["name"]


def test_compound_header_token_splitting():
    # Dot-matrix merged tokens split on the evidence that they straddle two of
    # the vertical bands the data forms (passed here as anchors), never on a
    # list of label spellings.
    t1 = {"text": "Op.Bal.Receipt", "x0": 300.0, "x1": 400.0}
    splits1 = table_ocr_service._split_compound_header_token(t1, anchors=[320.0, 380.0])
    assert len(splits1) == 2
    assert "Op.Bal" in splits1[0]["text"]
    assert "Receipt" in splits1[1]["text"]

    t2 = {"text": "Issue Closing", "x0": 450.0, "x1": 550.0}
    splits2 = table_ocr_service._split_compound_header_token(t2, anchors=[470.0, 530.0])
    assert len(splits2) == 2
    assert "Issue" in splits2[0]["text"]
    assert "Closing" in splits2[1]["text"]

    t3 = {"text": "aty.Balance", "x0": 460.0, "x1": 540.0}
    splits3 = table_ocr_service._split_compound_header_token(t3, anchors=[475.0, 525.0])
    assert len(splits3) == 2
    assert "aty." in splits3[0]["text"]
    assert "Balance" in splits3[1]["text"]


def test_line_clustering_collision_awareness():
    # Two tokens on same line (non-overlapping X) vs two tokens stacked vertically (colliding X)
    tokens = [
        # Line 1
        {"text": "BILASET-20MG", "x0": 100.0, "x1": 200.0, "y_center": 300.0, "height": 16.0},
        {"text": "10TAB", "x0": 220.0, "x1": 280.0, "y_center": 305.0, "height": 16.0},  # dy=5.0
        {"text": "50", "x0": 320.0, "x1": 350.0, "y_center": 302.0, "height": 15.0},
        # Line 2 (collides in X with BILASET-20MG)
        {"text": "BILASET-40MG", "x0": 100.0, "x1": 200.0, "y_center": 325.0, "height": 16.0},
        {"text": "10TAB", "x0": 220.0, "x1": 280.0, "y_center": 326.0, "height": 16.0},
        {"text": "45", "x0": 320.0, "x1": 350.0, "y_center": 324.0, "height": 15.0},
    ]

    lines = table_ocr_service._cluster_lines_by_y(tokens)
    assert len(lines) == 2
    assert "BILASET-20MG" in lines[0]["text"]
    assert "50" in lines[0]["text"]
    assert "BILASET-40MG" in lines[1]["text"]
    assert "45" in lines[1]["text"]


def test_logical_row_consolidation_continuation_lines():
    # Simulates visual lines where product description wraps across 2 lines
    col_bounds = [
        ("Description", 0.0, 200.0),
        ("Pack", 200.0, 300.0),
        ("Op.Qty", 300.0, 400.0),
        ("Cl.Qty", 400.0, 500.0)
    ]

    lines = [
        # Header (idx 0)
        {"text": "Description Pack Op.Qty Cl.Qty", "tokens": []},
        # Row 1 (Visual line 1: has numbers)
        {
            "text": "BILASET 40 MG TAB 10 TAB 35 64",
            "tokens": [
                {"text": "BILASET 40 MG", "x0": 20.0, "x1": 150.0},
                {"text": "10 TAB", "x0": 220.0, "x1": 280.0},
                {"text": "35", "x0": 330.0, "x1": 360.0},
                {"text": "64", "x0": 430.0, "x1": 460.0},
            ]
        },
        # Visual line 2: description continuation (NO numbers)
        {
            "text": "EXP: 11/2026 BATCH: 25S2GCA517",
            "tokens": [
                {"text": "EXP: 11/2026 BATCH: 25S2GCA517", "x0": 20.0, "x1": 180.0}
            ]
        }
    ]

    grid_rows, grid_metadata, footers = table_ocr_service._assemble_logical_rows(lines, 0, col_bounds)
    assert len(grid_rows) == 1
    assert "BILASET 40 MG" in grid_rows[0][0]
    assert "BATCH: 25S2GCA517" in grid_rows[0][0]
    assert grid_rows[0][2] == "35"
    assert grid_rows[0][3] == "64"


def test_logical_row_pending_title_completion():
    # In legacy ERPs (e.g. 1000411296), Line 1 has title with 0 numbers, Line 2 brings the numbers
    col_bounds = [
        ("Sn.", 0.0, 100.0),
        ("Description", 100.0, 300.0),
        ("Opening", 300.0, 400.0),
        ("Closing", 400.0, 500.0)
    ]

    lines = [
        {"text": "Header", "tokens": []},
        # Visual line 1: Title line with no numbers
        {
            "text": "1 BORIT SB 130",
            "tokens": [
                {"text": "1", "x0": 20.0, "x1": 50.0},
                {"text": "BORIT SB 130", "x0": 120.0, "x1": 250.0}
            ]
        },
        # Visual line 2: Data line bringing the numbers
        {
            "text": "BORIT SB 130 51.00 47.00",
            "tokens": [
                {"text": "BORIT SB 130", "x0": 120.0, "x1": 250.0},
                {"text": "51.00", "x0": 320.0, "x1": 360.0},
                {"text": "47.00", "x0": 420.0, "x1": 460.0}
            ]
        }
    ]

    grid_rows, grid_metadata, footers = table_ocr_service._assemble_logical_rows(lines, 0, col_bounds)
    assert len(grid_rows) == 1
    assert grid_rows[0][0] == "1"
    assert "BORIT SB 130" in grid_rows[0][1]
    assert grid_rows[0][2] == "51.00"
    assert grid_rows[0][3] == "47.00"


def test_description_similarity_rejects_different_dosage_skus():
    """
    Pure substring containment (the old row-merge check) would wrongly treat
    "PARA500" as the same product as "PARA500XR", and pure character-overlap
    similarity alone still scores "MINOSTRONG 1.25" vs "MINOSTRONG 2.5" above
    a typical 0.85 threshold since the shared prefix dominates the ratio. Both
    must be rejected as different products; a genuine OCR-noise duplicate of
    the SAME description must still be accepted.
    """
    from app.services.table_ocr_service import _description_similarity

    assert _description_similarity("para500", "para500xr") == 0.0
    assert _description_similarity("minostrong125", "minostrong25") == 0.0
    assert _description_similarity("boritsb65", "boritsb130") == 0.0
    # Same product, identical description - must still be recognized as a match.
    assert _description_similarity("boritsb130", "boritsb130") == 1.0
    # Same product, minor OCR noise (single dropped trailing letter, no digits involved).
    assert _description_similarity("bilaset40mgtab", "bilaset40mgta") >= 0.85


def test_description_similarity_rejects_interior_dosage_skus():
    """
    SKU pairs whose distinguishing number sits in the MIDDLE and which both end
    in letters. Comparing only the trailing digit run reports "no digits" for
    both sides, so the guard passes, the rows merge, and one product is dropped
    from the table entirely - observed as real missing rows on two benchmark
    documents. Every digit run must be compared, not just the trailing one.
    """
    from app.services.table_ocr_service import _description_similarity

    assert _description_similarity("minostrong125tab", "minostrong25tab") == 0.0
    assert _description_similarity("trebor0025cream", "trebor005cream") == 0.0
    assert _description_similarity("zebor10gel", "zebor20gel") == 0.0
    # An identical description containing interior digits must still merge.
    assert _description_similarity("minostrong125tab", "minostrong125tab") == 1.0


def test_logical_row_never_merges_different_dosage_skus():
    # "PARA500" (title line, no numbers) must NOT be merged with the immediately
    # following "PARA500XR" data line - these are different SKUs despite one
    # description being a prefix of the other.
    col_bounds = [
        ("Sn.", 0.0, 100.0),
        ("Description", 100.0, 300.0),
        ("Opening", 300.0, 400.0),
        ("Closing", 400.0, 500.0)
    ]

    lines = [
        {"text": "Header", "tokens": []},
        {
            "text": "1 PARA500",
            "tokens": [
                {"text": "1", "x0": 20.0, "x1": 50.0},
                {"text": "PARA500", "x0": 120.0, "x1": 250.0}
            ]
        },
        {
            "text": "PARA500XR 51.00 47.00",
            "tokens": [
                {"text": "PARA500XR", "x0": 120.0, "x1": 250.0},
                {"text": "51.00", "x0": 320.0, "x1": 360.0},
                {"text": "47.00", "x0": 420.0, "x1": 460.0}
            ]
        }
    ]

    grid_rows, grid_metadata, footers = table_ocr_service._assemble_logical_rows(lines, 0, col_bounds)
    assert len(grid_rows) == 2, "Different SKUs must remain separate rows, not merge into one"
    assert grid_rows[0][1].strip() == "PARA500"
    assert grid_rows[0][2] == "" and grid_rows[0][3] == "", "PARA500's own row must not inherit PARA500XR's numbers"
    assert grid_rows[1][1].strip() == "PARA500XR"
    assert grid_rows[1][2] == "51.00"
    assert grid_rows[1][3] == "47.00"


def test_strict_column_interval_bounds_preserves_blanks():
    # 4 columns: [Product] [Opening] [Receipt] [Closing]
    # Blank Receipt must NOT cause Closing to left-shift into Receipt!
    col_bounds = [
        ("Product", 0.0, 150.0),
        ("Opening", 150.0, 250.0),
        ("Receipt", 250.0, 350.0),
        ("Closing", 350.0, 450.0)
    ]

    tokens = [
        {"text": "AD-16ngSCH.", "x0": 20.0, "x1": 120.0},
        {"text": "48", "x0": 180.0, "x1": 210.0},   # Opening
        # Receipt is BLANK
        {"text": "39", "x0": 380.0, "x1": 410.0},   # Closing
    ]

    cells, meta = table_ocr_service._project_tokens_to_columns(tokens, col_bounds)
    assert cells == ["AD-16ngSCH.", "48", "", "39"]
    assert meta[2]["semantic"] == "EMPTY"
    assert meta[3]["normalized"] == 39.0


def test_closed_loop_reocr_suspicious_cell_identification():
    columns = ["Product", "Opening", "Receipt", "Issue", "Closing"]
    rows = [
        # Op 48, Rec 0, Iss 9 -> Expected Closing = 39, but OCR gave 30 (mismatch)
        ["Item A", "48", "0", "9", "30"]
    ]

    res = validation_service.validate_stock_table(columns, rows)
    assert res["arithmetic_mismatches"] == 1
    assert len(res["suspicious_cells"]) == 1
    sc = res["suspicious_cells"][0]
    assert sc["field"] == "closing_qty"
    assert sc["current_val"] == 30.0
    assert sc["expected_val"] == 39.0
    assert sc["discrepancy"] == 9.0


def test_closed_loop_reocr_never_fabricates():
    # Verify that without visual evidence, OCR pipeline NEVER silently overwrites numbers
    primary_table = {
        "columns": ["Product", "Opening", "Receipt", "Issue", "Closing"],
        "rows": [["Drug X", "50", "0", "10", "35"]],  # 50 - 10 = 40, OCR says 35
        "cells_metadata": [[
            {"tokens": []}, {"tokens": []}, {"tokens": []}, {"tokens": []},
            {"tokens": [{"text": "35", "x0": 400.0, "y0": 200.0, "x1": 430.0, "y1": 220.0}]}
        ]]
    }

    suspicious_cells = [{
        "row_index": 0,
        "column_index": 4,
        "current_val": 35.0,
        "expected_val": 40.0
    }]

    # Provide a non-existent image path to ensure Re-OCR fails gracefully
    repaired = pipeline._attempt_closed_loop_reocr(
        Path("non_existent_image.png"),
        primary_table,
        suspicious_cells
    )
    assert repaired is False
    # Value must remain strictly 35 (NO silent fabrication!)
    assert primary_table["rows"][0][4] == "35"


def test_multi_factor_confidence_scoring():
    columns = ["Product", "Opening", "Receipt", "Issue", "Closing"]
    rows = [
        ["Item A", "100", "0", "20", "80"],
        ["Item B", "50", "10", "0", "60"],
        ["Item C", "20", "0", "5", "15"],
    ]

    res = validation_service.validate_stock_table(
        columns, rows, ocr_confidence=0.95, layout_confidence=0.90
    )

    assert res["arithmetic_matches"] == 3
    assert res["arithmetic_mismatches"] == 0
    assert res["needs_manual_review"] is False
    assert res["overall_confidence"] >= 0.90
    assert res["row_structure_confidence"] == 1.0
    assert res["col_structure_confidence"] == 1.0


def test_empty_table_confidence_penalty():
    # If a table-heavy document produces an empty table, confidence MUST be heavily penalized
    columns = ["Product", "Opening", "Closing"]
    rows = []

    res = validation_service.validate_stock_table(
        columns, rows, ocr_confidence=0.95, layout_confidence=0.90
    )

    assert res["overall_confidence"] <= 0.35
    assert res["needs_manual_review"] is True
    assert res["row_structure_confidence"] == 0.0


def test_mandatory_manual_review_triggers():
    columns = ["Product", "Opening", "Receipt", "Issue", "Closing"]
    rows_with_mismatch = [
        ["Item A", "100", "0", "20", "50"]  # Expected 80, OCR says 50
    ]

    res = validation_service.validate_stock_table(
        columns, rows_with_mismatch, ocr_confidence=0.95, layout_confidence=0.90
    )

    assert res["needs_manual_review"] is True
    assert res["arithmetic_mismatches"] == 1
