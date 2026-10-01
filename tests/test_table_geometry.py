import pytest
from app.services.table_ocr_service import table_ocr_service


def test_table_geometry_y_clustering():
    # Tokens on line 1 (y ~ 50)
    # Tokens on line 2 (y ~ 75)
    tokens = [
        {"text": "ITEM", "y_center": 50.0, "height": 14.0, "x0": 20.0, "x1": 80.0},
        {"text": "QTY", "y_center": 51.5, "height": 13.5, "x0": 100.0, "x1": 140.0},
        {"text": "RATE", "y_center": 49.5, "height": 14.0, "x0": 160.0, "x1": 200.0},
        {"text": "Paracetamol", "y_center": 75.0, "height": 14.0, "x0": 20.0, "x1": 90.0},
        {"text": "100", "y_center": 76.0, "height": 13.0, "x0": 100.0, "x1": 130.0},
        {"text": "25.00", "y_center": 74.8, "height": 14.0, "x0": 160.0, "x1": 195.0},
    ]

    lines = table_ocr_service._cluster_lines_by_y(tokens)
    assert len(lines) == 2
    assert lines[0]["text"] == "ITEM QTY RATE"
    assert lines[1]["text"] == "Paracetamol 100 25.00"


def test_table_geometry_column_projection_preserves_blanks():
    # Define 4 column boundaries
    col_bounds = [
        ("Product Name", 0.0, 100.0),
        ("Opening", 100.0, 200.0),
        ("Issue", 200.0, 300.0),
        ("Closing", 300.0, 400.0)
    ]

    # Row 1: All cells present
    tokens_row1 = [
        {"text": "Aspirin", "x0": 10.0, "x1": 60.0},
        {"text": "50", "x0": 120.0, "x1": 140.0},
        {"text": "10", "x0": 220.0, "x1": 240.0},
        {"text": "40", "x0": 320.0, "x1": 340.0},
    ]
    cells1, meta1 = table_ocr_service._project_tokens_to_columns(tokens_row1, col_bounds)
    assert cells1 == ["Aspirin", "50", "10", "40"]
    assert meta1[0]["semantic"] == "TEXT"
    assert meta1[1]["semantic"] == "NUMERIC"
    assert meta1[1]["normalized"] == 50.0

    # Row 2: Opening and Issue are BLANK (only Product and Closing are present)
    # This must NOT shift "40" to the Opening column!
    tokens_row2 = [
        {"text": "Bilastine", "x0": 10.0, "x1": 80.0},
        {"text": "40", "x0": 330.0, "x1": 350.0},
    ]
    cells2, meta2 = table_ocr_service._project_tokens_to_columns(tokens_row2, col_bounds)
    assert cells2 == ["Bilastine", "", "", "40"]
    assert meta2[1]["semantic"] == "EMPTY"
    assert meta2[2]["semantic"] == "EMPTY"
    assert meta2[3]["semantic"] == "NUMERIC"
    assert meta2[3]["normalized"] == 40.0

    # Row 3: Dash / Not reported in Issue column
    tokens_row3 = [
        {"text": "Cetirizine", "x0": 10.0, "x1": 70.0},
        {"text": "100", "x0": 110.0, "x1": 130.0},
        {"text": "-", "x0": 240.0, "x1": 250.0},
        {"text": "100", "x0": 320.0, "x1": 350.0},
    ]
    cells3, meta3 = table_ocr_service._project_tokens_to_columns(tokens_row3, col_bounds)
    assert cells3 == ["Cetirizine", "100", "-", "100"]
    assert meta3[2]["semantic"] == "NOT_REPORTED"
    assert meta3[2]["normalized"] is None
