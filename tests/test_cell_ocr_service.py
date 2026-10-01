import pytest
import numpy as np
import cv2
from app.services.cell_ocr_service import cell_ocr_service


def test_split_spanning_tokens_two_columns():
    col_bounds = [
        ("Description", 0.0, 200.0),
        ("Issue", 200.0, 300.0),
        ("Closing", 300.0, 400.0),
    ]

    tokens = [
        # Normal token inside Description
        {"text": "ITEM A", "x0": 20.0, "x1": 150.0, "y0": 100.0, "y1": 120.0, "y_center": 110.0},
        # Spanning token crossing Issue (200-300) and Closing (300-400)
        {"text": "9 39", "x0": 240.0, "x1": 360.0, "y0": 100.0, "y1": 120.0, "y_center": 110.0}
    ]

    refined = cell_ocr_service.split_spanning_tokens(tokens, col_bounds)
    assert len(refined) == 3
    assert refined[0]["text"] == "ITEM A"
    assert refined[1]["text"] == "9"
    assert 200.0 <= refined[1]["xc"] < 300.0
    assert refined[2]["text"] == "39"
    assert 300.0 <= refined[2]["xc"] < 400.0


def test_split_spanning_tokens_qty_and_amount():
    col_bounds = [
        ("Product", 0.0, 250.0),
        ("Closing Qty", 250.0, 350.0),
        ("Closing Amount", 350.0, 500.0)
    ]

    tokens = [
        {"text": "101 16725.60", "x0": 280.0, "x1": 460.0, "y0": 50.0, "y1": 70.0, "y_center": 60.0}
    ]

    refined = cell_ocr_service.split_spanning_tokens(tokens, col_bounds)
    assert len(refined) == 2
    assert refined[0]["text"] == "101"
    assert refined[1]["text"] == "16725.60"
    assert 250.0 <= refined[0]["xc"] < 350.0
    assert 350.0 <= refined[1]["xc"] < 500.0


def test_split_spanning_tokens_never_splits_text():
    col_bounds = [
        ("Sn.", 0.0, 80.0),
        ("Description", 80.0, 350.0),
        ("Unit", 350.0, 450.0)
    ]

    # Product text starting at 70px should NOT be split into Sn. and Description
    tokens = [
        {"text": "BORIT SB 130", "x0": 70.0, "x1": 250.0, "y0": 100.0, "y1": 120.0, "y_center": 110.0},
        {"text": "MINOSTRONG 60ML", "x0": 60.0, "x1": 260.0, "y0": 130.0, "y1": 150.0, "y_center": 140.0}
    ]

    refined = cell_ocr_service.split_spanning_tokens(tokens, col_bounds)
    assert len(refined) == 2
    assert refined[0]["text"] == "BORIT SB 130"
    assert refined[1]["text"] == "MINOSTRONG 60ML"


def test_extract_fused_packing():
    desc, pack = cell_ocr_service.extract_fused_packing("MOISTE CREAM 100GM")
    assert desc == "MOISTE CREAM"
    assert pack.upper() == "100GM"

    desc, pack = cell_ocr_service.extract_fused_packing("ADABOR GEL 15GM")
    assert desc == "ADABOR GEL"
    assert pack.upper() == "15GM"

    desc, pack = cell_ocr_service.extract_fused_packing("BILASET-40 TAB 10TAB")
    assert desc == "BILASET-40 TAB"
    assert pack.upper() == "10TAB"

    desc, pack = cell_ocr_service.extract_fused_packing("XTANZ-TAB 10'S")
    assert desc == "XTANZ-TAB"
    assert pack.upper() == "10'S"


def test_compute_cell_quality():
    # Synthetic high-contrast image
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    cv2.putText(img, "48", (20, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    qual = cell_ocr_service.compute_cell_quality(img)
    assert qual["contrast"] > 100.0
    assert qual["mean_intensity"] < 255.0
    assert qual["foreground_ratio"] > 0.0


def test_dash_detection_on_synthetic_dash():
    # Create white canvas with a single horizontal dark dash in the center
    cell_img = np.full((30, 80, 3), 255, dtype=np.uint8)
    # Draw horizontal black line: width 16px, height 3px
    cv2.rectangle(cell_img, (32, 13), (48, 16), (30, 30, 30), -1)

    val, sem, conf = cell_ocr_service.detect_cell_dash_vs_empty(cell_img, (0, 0, 80, 30))
    assert val == "-"
    assert sem == "NOT_REPORTED"
    assert conf >= 0.85


def test_empty_cell_detection_on_blank_crop():
    # Create completely blank white canvas
    cell_img = np.full((30, 80, 3), 255, dtype=np.uint8)

    val, sem, conf = cell_ocr_service.detect_cell_dash_vs_empty(cell_img, (0, 0, 80, 30))
    assert val == ""
    assert sem == "EMPTY"
    assert conf == 1.0


def test_numeric_disambiguation_rules():
    assert cell_ocr_service.disambiguate_numeric("10.9", "packing") == "10.S"
    assert cell_ocr_service.disambiguate_numeric("10.8", "packing") == "10.S"
    assert cell_ocr_service.disambiguate_numeric("1GRAN", "packing") == "1GRAM"
    assert cell_ocr_service.disambiguate_numeric("θ", "numeric") == "0"
    assert cell_ocr_service.disambiguate_numeric("O", "numeric") == "0"
    assert cell_ocr_service.disambiguate_numeric("l", "numeric") == "1"
    assert cell_ocr_service.disambiguate_numeric("S", "numeric") == "5"
    assert cell_ocr_service.disambiguate_numeric("24·50", "numeric") == "24.50"


def test_extract_cell_value_cascade_level1():
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    res = cell_ocr_service.extract_cell_value(
        image=img,
        cell_bbox=(0, 0, 80, 30),
        col_name="Opening",
        expected_type="numeric",
        existing_candidate="45.00",
        existing_conf=0.92,
        allow_qwen=False
    )
    assert res["raw_value"] == "45.00"
    assert res["engine"] == "rapidocr_level1"
    assert res["review_required"] is False


def test_dash_detection_rejects_uniform_paper_noise():
    # Synthetic flat grey paper with tiny sensor noise (variance < 80, contrast < 40)
    rng = np.random.default_rng(42)
    noise_img = rng.integers(180, 195, (30, 80, 3), dtype=np.uint8)

    val, sem, conf = cell_ocr_service.detect_cell_dash_vs_empty(noise_img, (0, 0, 80, 30))
    assert val == ""
    assert sem == "EMPTY"
    assert conf == 1.0


def test_extract_cell_value_blank_preservation():
    # When existing_candidate is empty and image is flat grey paper
    rng = np.random.default_rng(42)
    flat_paper = rng.integers(185, 195, (30, 80, 3), dtype=np.uint8)

    res = cell_ocr_service.extract_cell_value(
        image=flat_paper,
        cell_bbox=(0, 0, 80, 30),
        col_name="Amount",
        expected_type="numeric",
        existing_candidate="",
        existing_conf=0.0,
        allow_qwen=False
    )
    assert res["raw_value"] == ""
    assert res["normalized_value"] == ""
    assert res["engine"] == "blank_preservation"
    assert res["review_required"] is False


def test_genuinely_blank_cell_zero_qwen():
    """Test 1: Genuinely blank cell returns empty string, visually empty True, zero Qwen."""
    blank_crop = np.full((30, 80, 3), 255, dtype=np.uint8)
    is_empty = cell_ocr_service.is_cell_visually_empty(blank_crop, "Closing", "numeric", [])
    assert is_empty is True

    res = cell_ocr_service.extract_cell_value(
        image=blank_crop,
        cell_bbox=(0, 0, 80, 30),
        col_name="Closing",
        expected_type="numeric",
        existing_candidate="",
        existing_conf=0.0,
        allow_qwen=True  # Gated: empty cells should never trigger Qwen even if allow_qwen is True
    )
    assert res["raw_value"] == ""
    assert res["engine"] == "blank_preservation"
    assert "qwen2.5vl_crop" not in res.get("variants_used", [])


def test_faint_printed_numeric_value_recovery():
    """Test 2: Faint printed numeric value recovers correct digits."""
    img = np.full((32, 90, 3), 245, dtype=np.uint8)
    # Faint text (color 170 on 245 background)
    cv2.putText(img, "150", (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (170, 170, 170), 2)
    quality = cell_ocr_service.compute_cell_quality(img)
    assert quality["contrast"] > 20.0

    disamb = cell_ocr_service.disambiguate_numeric("15O", "numeric", "Opening")
    assert disamb == "150"


def test_low_contrast_screen_numeric_value():
    """Test 3: Low-contrast screen numeric value selects correct disambiguated digits."""
    # Screen character with dot-matrix font confusion
    res = cell_ocr_service.disambiguate_numeric("24·50", "numeric", "Rate")
    assert res == "24.50"
    res_b = cell_ocr_service.disambiguate_numeric("b4", "numeric", "Quantity")
    assert res_b == "84"


def test_faint_dash_not_hallucinated_as_digit():
    """Test 4: Faint dash is normalized to '-' and does not hallucinate a digit."""
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    # Draw standard dash: width 20px, height 4px, centered
    cv2.rectangle(img, (30, 13), (50, 17), (40, 40, 40), -1)

    val, sem, conf = cell_ocr_service.detect_cell_dash_vs_empty(img, (0, 0, 80, 30))
    assert val == "-"
    assert sem == "NOT_REPORTED"
    # Never hallucinated as '1' or '7'
    assert val not in ["1", "7", "0"]


def test_faint_zero_preserved_not_eight():
    """Test 5: Faint zero is normalized to '0', not empty or 8."""
    assert cell_ocr_service.disambiguate_numeric("O", "numeric", "Issue") == "0"
    assert cell_ocr_service.disambiguate_numeric("o", "numeric", "Issue") == "0"
    assert cell_ocr_service.disambiguate_numeric("θ", "numeric", "Issue") == "0"
    norm_zero = cell_ocr_service._normalize_val("0", "numeric")
    assert norm_zero == 0.0


def test_fast_accept_triggers_fallback_on_suspicion():
    """Test 6: Fast accept triggers fallback when arithmetic mismatch or suspicion flag is raised."""
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    cv2.putText(img, "45.00", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # With suspicion (arithmetic mismatch = True), Fast Accept is bypassed
    res_suspicious = cell_ocr_service.extract_cell_value(
        image=img,
        cell_bbox=(0, 0, 80, 30),
        col_name="Closing",
        expected_type="numeric",
        existing_candidate="45.00",
        existing_conf=0.95,
        row_arithmetic_mismatch=True,
        allow_qwen=False
    )
    assert res_suspicious["engine"] != "rapidocr_level1"

    # Without suspicion, Fast Accept accepts directly
    res_ok = cell_ocr_service.extract_cell_value(
        image=img,
        cell_bbox=(0, 0, 80, 30),
        col_name="Closing",
        expected_type="numeric",
        existing_candidate="45.00",
        existing_conf=0.95,
        row_arithmetic_mismatch=False,
        allow_qwen=False
    )
    assert res_ok["engine"] == "rapidocr_level1"


def test_candidate_ranking_prefers_variant_agreement():
    """Test 7: Candidate ranking prefers variant agreement over standalone low-confidence."""
    candidates = [
        {"raw": "28", "text": "28", "confidence": 0.72, "variant": "clahe"},
        {"raw": "28", "text": "28", "confidence": 0.75, "variant": "otsu"},
        {"raw": "23", "text": "23", "confidence": 0.81, "variant": "sharpened"}
    ]
    quality = {"contrast": 80.0, "variance": 120.0, "mean_intensity": 200.0, "foreground_ratio": 0.08}
    best = cell_ocr_service.rank_candidates(candidates, "numeric", quality, col_name="Closing")
    assert best is not None
    # '28' has 2 variant agreements vs standalone '23'
    assert best["text"] == "28"


def test_bbox_xc_ratio_computation():
    """xc_ratio must be a real geometric measurement of bbox-center / crop-width, not a stub."""
    # Box spans x=[10,30] in a 100px-wide crop -> center=20 -> ratio 0.20
    box = [[10, 5], [30, 5], [30, 15], [10, 15]]
    ratio = cell_ocr_service._bbox_xc_ratio(box, 100)
    assert ratio is not None
    assert abs(ratio - 0.20) < 1e-6
    # Missing/invalid input must not raise, and must signal "unknown" rather than a fake value
    assert cell_ocr_service._bbox_xc_ratio(None, 100) is None
    assert cell_ocr_service._bbox_xc_ratio(box, 0) is None


def test_candidate_ranking_prefers_geometrically_centered_bbox():
    """
    Two otherwise-identical candidates (same text format, confidence, agreement count)
    must be disambiguated by real bbox geometry: a candidate whose recognized text sits
    near the horizontal center of its own cell crop (clean, well-isolated read) must
    outrank one pushed to the crop's edge (more likely bleed from a neighboring column).
    This locks in that "centering" is an actual geometric computation, not the old
    unconditional flat constant that never differentiated candidates.
    """
    candidates = [
        {"raw": "45", "text": "45", "confidence": 0.80, "variant": "original", "xc_ratio": 0.50},
        {"raw": "45", "text": "45", "confidence": 0.80, "variant": "clahe", "xc_ratio": 0.95},
    ]
    quality = {"contrast": 80.0, "variance": 120.0, "mean_intensity": 200.0, "foreground_ratio": 0.08}
    best = cell_ocr_service.rank_candidates(candidates, "numeric", quality, col_name="Closing")
    assert best is not None
    assert best["variant"] == "original", "Centered bbox candidate must be preferred over an edge-clipped one"


def test_adaptive_qwen_budget_bounds():
    """Test 8: Adaptive Qwen budget stops calls when budget is exceeded."""
    # When allow_qwen is False, extract_cell_value never calls Qwen
    img = np.full((30, 80, 3), 255, dtype=np.uint8)
    res = cell_ocr_service.extract_cell_value(
        image=img,
        cell_bbox=(0, 0, 80, 30),
        col_name="Closing",
        expected_type="numeric",
        existing_candidate="??",
        existing_conf=0.10,
        allow_qwen=False
    )
    assert res["engine"] != "level3_qwen"


def test_classify_cell_type_word_boundaries():
    """Test 9: 'Packing' must NOT match numeric 'in' substring and must classify as PACKING."""
    from app.services.cell_ocr_service import CellType
    assert cell_ocr_service.classify_cell_type("Packing") == CellType.PACKING
    assert cell_ocr_service.classify_cell_type("Pack") == CellType.PACKING
    assert cell_ocr_service.classify_cell_type("Pkg") == CellType.PACKING
    assert cell_ocr_service.classify_cell_type("Unit") == CellType.UNIT
    assert cell_ocr_service.classify_cell_type("Sn.") == CellType.SERIAL
    assert cell_ocr_service.classify_cell_type("Closing Balance") == CellType.DECIMAL_VALUE
    assert cell_ocr_service.classify_cell_type("Rate") == CellType.RATE
    assert cell_ocr_service.classify_cell_type("In") == CellType.DECIMAL_VALUE
    assert cell_ocr_service.classify_cell_type("Out") == CellType.DECIMAL_VALUE


def test_packing_preservation_and_fast_accept():
    """Test 10: Valid packing formats (1*10, 15GM, 10.S) fast-accept and are never corrupted."""
    from app.services.cell_ocr_service import CellType
    img = np.full((30, 80, 3), 255, dtype=np.uint8)

    for pack_val in ["1*10", "15GM", "20GM", "10.S", "100ML"]:
        res = cell_ocr_service.extract_cell_value(
            image=img,
            cell_bbox=(0, 0, 80, 30),
            col_name="Packing",
            expected_type=CellType.PACKING,
            existing_candidate=pack_val,
            existing_conf=0.90,
            allow_qwen=False
        )
        assert res["raw_value"] == pack_val
        assert res["engine"] == "rapidocr_level1"
        assert res["review_required"] is False


def test_refine_table_cells_never_fabricates_packing_from_description():
    """
    A Packing/Unit cell OCR'd as a bare "1" (a likely narrow-column truncation) must
    never be force-converted into a guessed value like "1BOT" purely because the
    product description mentions "syrup"/"bottle" - that is fabrication from context,
    not OCR evidence. On a blank image (no visual evidence anywhere), the cell must
    resolve to UNKNOWN with the raw "1" preserved, never a specific invented packing.
    """
    img_w, img_h = 500, 40
    image = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    col_bounds = [
        ("Description", 0.0, 300.0),
        ("Packing", 300.0, 400.0),
        ("Qty", 400.0, 500.0),
    ]
    logical_row_bboxes = [(0.0, float(img_h))]
    # Description mentions "syrup"/"bottle" - exactly the vocabulary the old code used
    # to force Packing into "1BOT". No digit-based packing token exists in the text
    # for the (legitimate) fused-packing fallback to relocate either.
    grid_rows = [["COUGH SYRUP BOTTLE", "1", "10"]]
    grid_meta = [[
        {"raw": "COUGH SYRUP BOTTLE", "normalized": "COUGH SYRUP BOTTLE", "status": "VALUE", "tokens": [], "repaired": False},
        {"raw": "1", "normalized": None, "status": "VALUE", "tokens": [], "repaired": False},
        {"raw": "10", "normalized": 10.0, "status": "VALUE", "tokens": [], "repaired": False},
    ]]

    out_rows, out_meta = cell_ocr_service.refine_table_cells(
        image=image,
        col_bounds=col_bounds,
        logical_row_bboxes=logical_row_bboxes,
        grid_rows=grid_rows,
        grid_meta=grid_meta,
    )

    packing_val = out_rows[0][1]
    packing_status = out_meta[0][1].get("status")
    assert packing_val != "1BOT", "Packing must never be guessed from description vocabulary"
    assert packing_val != "10TAB", "Packing must never be guessed from description vocabulary"
    assert packing_val == "1", "Raw OCR value must be preserved, not overwritten with a guess"
    assert packing_status == "UNKNOWN", "Unresolved packing truncation must be flagged UNKNOWN, not silently marked VALUE"


def test_refine_table_cells_arithmetic_conflict_never_overwrites_without_reocr_confirmation():
    """
    When a product row's Out/Issue cell conflicts with what the subtotal row's arithmetic
    implies (Opening + In - Out = Balance), the old code directly copied the neighbor's
    string into the cell. That must now only happen if an independent targeted re-OCR of
    the cell itself confirms the value - on a blank image (no visual evidence anywhere),
    the re-OCR finds nothing, so the original raw OCR value must be preserved and the
    cell flagged CONFLICT for review, never silently overwritten with the neighbor's digits.
    """
    img_w, img_h = 500, 40
    image = np.full((img_h, img_w, 3), 255, dtype=np.uint8)

    col_bounds = [
        ("Description", 0.0, 100.0),
        ("Opening", 100.0, 200.0),
        ("Receipt", 200.0, 300.0),
        ("Issue", 300.0, 400.0),
        ("Closing", 400.0, 500.0),
    ]
    logical_row_bboxes = [(0.0, 20.0), (20.0, 40.0)]

    # Product row: OCR read Issue="6", but Opening(50) + Receipt(0) - Issue - Closing(34) = 0
    # only balances if Issue is actually 16 (as the subtotal row's own Issue cell shows).
    grid_rows = [
        ["DRUG X", "50", "0", "6", "34"],
        ["DRUG X S-Total", "", "", "16", ""],
    ]

    def _meta(val):
        return {"raw": val, "normalized": None, "status": "VALUE", "tokens": [], "repaired": False}

    grid_meta = [
        [_meta("DRUG X"), _meta("50"), _meta("0"), _meta("6"), _meta("34")],
        [_meta("DRUG X S-Total"), _meta(""), _meta(""), _meta("16"), _meta("")],
    ]

    out_rows, out_meta = cell_ocr_service.refine_table_cells(
        image=image,
        col_bounds=col_bounds,
        logical_row_bboxes=logical_row_bboxes,
        grid_rows=grid_rows,
        grid_meta=grid_meta,
    )

    issue_val = out_rows[0][3]
    issue_meta = out_meta[0][3]
    assert issue_val == "6", "Original raw OCR value must be preserved without independent re-OCR confirmation"
    assert issue_meta.get("arithmetic_status") == "CONFLICT"
    assert issue_meta.get("needs_review") is True
    assert issue_meta.get("repair_source") != "arithmetic_conflict_reocr_confirmed"


def test_dot_matrix_s_after_decimal():
    """Test 11: Dot-matrix digit confusions after decimal (10.9, 10.8, 10.5, 6.8, 6.9 -> 10.S, 6.S)."""
    from app.services.cell_ocr_service import CellType
    assert cell_ocr_service.disambiguate_numeric("10.9", CellType.PACKING) == "10.S"
    assert cell_ocr_service.disambiguate_numeric("10.8", CellType.PACKING) == "10.S"
    assert cell_ocr_service.disambiguate_numeric("10.5", CellType.PACKING) == "10.S"
    assert cell_ocr_service.disambiguate_numeric("10:8", CellType.PACKING) == "10.S"
    assert cell_ocr_service.disambiguate_numeric("6.8", CellType.PACKING) == "6.S"
    assert cell_ocr_service.disambiguate_numeric("6.9", CellType.PACKING) == "6.S"
    assert cell_ocr_service.disambiguate_numeric("10.S.", CellType.PACKING) == "10.S"


def test_serial_candidate_ranking_rejects_alphabetic_noise():
    """Test 12: Serial column candidate ranking strongly prefers digits over alphabetic text."""
    from app.services.cell_ocr_service import CellType
    candidates = [
        {"raw": "No", "text": "No", "confidence": 0.85, "variant": "original"},
        {"raw": "1", "text": "1", "confidence": 0.70, "variant": "clahe"}
    ]
    quality = {"contrast": 70.0, "variance": 100.0, "mean_intensity": 200.0, "foreground_ratio": 0.05}
    best = cell_ocr_service.rank_candidates(candidates, CellType.SERIAL, quality, col_name="Sn.")
    assert best is not None
    assert best["text"] == "1"


def test_faint_dot_matrix_dash_detection():
    """Test 13: Faint dot-matrix dash with contrast around 28 is correctly detected."""
    # Background 220, dash ink 192 (contrast 28)
    cell_img = np.full((30, 80, 3), 220, dtype=np.uint8)
    # Draw horizontal line: width 18px, height 3px, intensity 192
    cv2.rectangle(cell_img, (31, 13), (49, 16), (192, 192, 192), -1)

    val, sem, conf = cell_ocr_service.detect_cell_dash_vs_empty(cell_img, (0, 0, 80, 30))
    assert val == "-"
    assert sem == "NOT_REPORTED"
    assert conf >= 0.85



