"""
Generalization Test Suite for Unseen Documents
Verifies that all cell-level OCR rules and disambiguation logic are based on
general document characteristics, image quality metrics, and topology,
and function properly on completely unseen documents without filename-specific
or benchmark-specific hardcoding.
"""

import pytest
import numpy as np
import cv2
from app.services.cell_ocr_service import cell_ocr_service, CellType


def test_unseen_dot_matrix_pin_dropout():
    """Verifies dot-matrix character disambiguation on unseen dot-matrix patterns."""
    # S vs 5
    assert cell_ocr_service.disambiguate_numeric("12S.50", CellType.DECIMAL_VALUE) == "125.50"
    # O vs 0
    assert cell_ocr_service.disambiguate_numeric("1O5", CellType.INTEGER_QTY) == "105"
    # l/I vs 1
    assert cell_ocr_service.disambiguate_numeric("l05", CellType.INTEGER_QTY) == "105"
    # B vs 8
    assert cell_ocr_service.disambiguate_numeric("b40.00", CellType.AMOUNT) == "840.00"
    # Packaging 10.9 / 10.8 / 10.5 / 6.8 / 6.9 -> 10.S / 6.S
    assert cell_ocr_service.disambiguate_numeric("10.9", CellType.PACKING) == "10.S"
    assert cell_ocr_service.disambiguate_numeric("6.8", CellType.PACKING) == "6.S"
    assert cell_ocr_service.disambiguate_numeric("20.5", CellType.PACKING) == "20.S"
    # Normalizing dot-matrix middle dots
    assert cell_ocr_service.disambiguate_numeric("15·75", CellType.RATE) == "15.75"


def test_unseen_varying_column_counts():
    """Verifies column classification and spanning splitting on 3, 7, and 14 column tables."""
    # 3-column table
    cols_3 = [("Product Name", 0.0, 300.0), ("Quantity", 300.0, 450.0), ("Amount", 450.0, 600.0)]
    assert cell_ocr_service.classify_cell_type("Product Name") == CellType.DESCRIPTION
    assert cell_ocr_service.classify_cell_type("Quantity") == CellType.DECIMAL_VALUE
    assert cell_ocr_service.classify_cell_type("Amount") == CellType.AMOUNT

    toks_3 = [
        {"text": "50 12500.00", "x0": 320.0, "x1": 550.0, "y0": 20.0, "y1": 40.0}
    ]
    split_3 = cell_ocr_service.split_spanning_tokens(toks_3, cols_3)
    assert len(split_3) == 2
    assert split_3[0]["text"] == "50"
    assert split_3[1]["text"] == "12500.00"

    # 14-column complex ledger
    cols_14_names = [
        "Sn.", "Code", "Particulars", "Pack", "Batch", "Exp",
        "MRP", "Rate", "Qty", "Free", "Disc%", "Tax%", "Cess%", "Net Amount"
    ]
    expected_types_14 = [
        CellType.SERIAL, CellType.DESCRIPTION, CellType.DESCRIPTION, CellType.PACKING,
        CellType.DESCRIPTION, CellType.DATE, CellType.RATE, CellType.RATE,
        CellType.DECIMAL_VALUE, CellType.INTEGER_QTY, CellType.PERCENTAGE,
        CellType.PERCENTAGE, CellType.PERCENTAGE, CellType.AMOUNT
    ]
    for col_n, exp_t in zip(cols_14_names, expected_types_14):
        actual_t = cell_ocr_service.classify_cell_type(col_n)
        assert actual_t == exp_t, f"Column '{col_n}' classified as '{actual_t}', expected '{exp_t}'"


def test_unseen_multi_word_product_names_with_hyphens():
    """Verifies that multi-word hyphenated product names are protected from splitting and packaging is extracted."""
    # Spanning check: descriptions must NEVER be split even if they contain numbers and spaces
    cols = [("Description", 0.0, 300.0), ("Qty", 300.0, 400.0)]
    text_toks = [
        {"text": "CEF-PET 200 DT TAB", "x0": 20.0, "x1": 320.0, "y0": 10.0, "y1": 30.0},
        {"text": "TELMI-H 40/12.5 MG", "x0": 10.0, "x1": 310.0, "y0": 40.0, "y1": 60.0}
    ]
    # Because 'CEF-PET 200 DT TAB' contains letters, it must never be split across columns
    res_toks = cell_ocr_service.split_spanning_tokens(text_toks, cols)
    assert len(res_toks) == 2
    assert res_toks[0]["text"] == "CEF-PET 200 DT TAB"
    assert res_toks[1]["text"] == "TELMI-H 40/12.5 MG"

    # Fused packaging extraction on complex descriptions
    desc1, pack1 = cell_ocr_service.extract_fused_packing("AZITHRAL 500 TABLET 1X5")
    assert desc1 == "AZITHRAL 500 TABLET"
    assert pack1 == "1X5"

    desc2, pack2 = cell_ocr_service.extract_fused_packing("BETADINE OINTMENT 20GM")
    assert desc2 == "BETADINE OINTMENT"
    assert pack2 == "20GM"


def test_unseen_screen_photograph_moire_and_subpixel_stripes():
    """Verifies that periodic 1-2px CRT/screen subpixel stripes on blank cells do not trigger false character detection."""
    h, w = 32, 70
    screen_cell = np.full((h, w, 3), 235, dtype=np.uint8)
    # Simulate CRT / screen subpixel stripes (vertical lines of width 1-2px spaced every 4px)
    for x in range(0, w, 4):
        screen_cell[:, x:x+1] = 190

    # No character components exist, only periodic stripes
    is_empty = cell_ocr_service.is_cell_visually_empty(screen_cell, "Rate", CellType.RATE, [])
    assert is_empty is True, "Screen subpixel stripes on empty cell should be classified as visually empty"


def test_unseen_faint_paper_scans_contrast_below_30():
    """Verifies that faint ink (contrast ~25) is recognized as visual content and not treated as blank paper."""
    h, w = 30, 80
    bg_val = 230
    ink_val = 205  # Contrast = 25
    faint_cell = np.full((h, w, 3), bg_val, dtype=np.uint8)
    # Draw character stroke (width 12, height 16)
    cv2.rectangle(faint_cell, (25, 7), (37, 23), (ink_val, ink_val, ink_val), -1)

    is_empty = cell_ocr_service.is_cell_visually_empty(faint_cell, "Opening", CellType.DECIMAL_VALUE, [])
    assert is_empty is False, "Faint printed character stroke must not be discarded as empty paper"


def test_unseen_broken_dot_matrix_stroke_not_hallucinated_as_stripe():
    """
    A faint/broken dot-matrix digit can fragment into narrow (width<=2) pieces that the
    screen-stripe rejection bucket would otherwise discard entirely as noise, permanently
    losing real ink. Two narrow fragments confined to the same x-band (a broken vertical
    stroke) must be recognized as content, unlike the periodic-stripe case above where
    fragments are spread across the whole cell width.
    """
    h, w = 32, 70
    bg_val = 235
    ink_val = 150
    broken_stroke_cell = np.full((h, w, 3), bg_val, dtype=np.uint8)
    # Two vertically-stacked narrow fragments at (nearly) the same x position, separated
    # by a faint gap - simulating a broken "1"/"I" stroke, not a periodic scan-line pattern.
    cv2.rectangle(broken_stroke_cell, (34, 4), (35, 13), (ink_val, ink_val, ink_val), -1)
    cv2.rectangle(broken_stroke_cell, (34, 18), (35, 27), (ink_val, ink_val, ink_val), -1)

    is_empty = cell_ocr_service.is_cell_visually_empty(broken_stroke_cell, "Closing", CellType.INTEGER_QTY, [])
    assert is_empty is False, "Broken dot-matrix stroke fragments in one x-band must not be treated as empty/stripe noise"


def test_unseen_genuine_blanks_vs_dashes_grid():
    """Verifies reliable disambiguation between genuine blank cells and printed dashes across unseen grids."""
    # 1. Truly blank cell
    blank_cell = np.full((28, 65, 3), 245, dtype=np.uint8)
    val_b, sem_b, conf_b = cell_ocr_service.detect_cell_dash_vs_empty(blank_cell, (0, 0, 65, 28))
    assert val_b == ""
    assert sem_b == "EMPTY"
    assert conf_b == 1.0

    # 2. Printed dash cell (aspect ratio >= 2.0, centered)
    dash_cell = np.full((28, 65, 3), 245, dtype=np.uint8)
    cv2.rectangle(dash_cell, (22, 12), (42, 15), (50, 50, 50), -1)  # w=20, h=3
    val_d, sem_d, conf_d = cell_ocr_service.detect_cell_dash_vs_empty(dash_cell, (0, 0, 65, 28))
    assert val_d == "-"
    assert sem_d == "NOT_REPORTED"
    assert conf_d >= 0.85
