import pytest
import numpy as np
from pathlib import Path
from app.services.table_ocr_service import table_ocr_service


def test_detect_table_bounds_filters_bezel_and_watermark():
    # Widths/positions here are calibrated against a real 1600px-wide screen-photo
    # document, not picked arbitrarily: on that document, legitimate rightmost table
    # column data (a CLOSING column) sits up to x-ratio ~0.77 while the genuinely
    # separate side panel starts at ~0.80 - too close together for a loosely-chosen
    # ratio to get right, so this test pins both boundaries explicitly.
    dummy_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    tokens = [
        {"text": "COMPAQ 18.5 LED Monitor", "x0": 50.0, "x1": 250.0, "y0": 100.0, "y1": 140.0, "xc": 150.0, "yc": 120.0, "y_center": 120.0, "height": 40.0, "width": 200.0},
        {"text": "ITEM DESCRIPTION", "x0": 80.0, "x1": 300.0, "y0": 260.0, "y1": 285.0, "xc": 190.0, "yc": 272.5, "y_center": 272.5, "height": 25.0, "width": 220.0},
        {"text": "PARACETAMOL 500", "x0": 80.0, "x1": 280.0, "y0": 320.0, "y1": 345.0, "xc": 180.0, "yc": 332.5, "y_center": 332.5, "height": 25.0, "width": 200.0},
        # Rightmost legitimate table column (e.g. CLOSING) at x-ratio ~0.75 - must survive.
        {"text": "127", "x0": 1440.0, "x1": 1480.0, "y0": 320.0, "y1": 345.0, "xc": 1460.0, "yc": 332.5, "y_center": 332.5, "height": 25.0, "width": 40.0},
        {"text": "Shot on OnePlus By Chandan4u", "x0": 100.0, "x1": 400.0, "y0": 1020.0, "y1": 1060.0, "xc": 250.0, "yc": 1040.0, "y_center": 1040.0, "height": 40.0, "width": 300.0},
        # Genuinely separate side panel at x-ratio ~0.83 - must be filtered.
        {"text": "D/NOTE CDNB00030", "x0": 1600.0, "x1": 1800.0, "y0": 400.0, "y1": 420.0, "xc": 1700.0, "yc": 410.0, "y_center": 410.0, "height": 20.0, "width": 200.0},
    ]

    t_bbox, filtered = table_ocr_service._detect_table_bounds(tokens, dummy_img)
    filtered_texts = [t["text"] for t in filtered]

    # COMPAQ bezel filtered out
    assert not any("compaq" in t.lower() for t in filtered_texts)
    # Camera watermark filtered out
    assert not any("oneplus" in t.lower() for t in filtered_texts)
    # Side panel on right filtered out
    assert not any("d/note" in t.lower() for t in filtered_texts)
    # Main table content retained, including the rightmost legitimate column
    assert "ITEM DESCRIPTION" in filtered_texts
    assert "PARACETAMOL 500" in filtered_texts
    assert "127" in filtered_texts


def test_split_compound_header_tokens():
    # A fused header label is split on GEOMETRY, not on a list of known labels:
    # the token must straddle two bands the data actually forms. Those bands are
    # supplied here as anchors, and the cut lands at the token's own casing or
    # punctuation boundary nearest the gap between them.
    t1 = {"text": "Op.Bal.Receipt", "x0": 100.0, "x1": 200.0}
    splits = table_ocr_service._split_compound_header_token(t1, anchors=[120.0, 180.0])
    assert len(splits) == 2
    assert splits[0]["text"].lower().startswith("op.bal")
    assert splits[1]["text"].lower().startswith("receipt")
    assert splits[0]["x1"] == splits[1]["x0"]
    assert 100.0 < splits[0]["x1"] < 200.0

    t2 = {"text": "Issue Closing", "x0": 200.0, "x1": 300.0}
    splits2 = table_ocr_service._split_compound_header_token(t2, anchors=[220.0, 280.0])
    assert len(splits2) == 2
    assert splits2[0]["text"].lower() == "issue"
    assert splits2[1]["text"].lower() == "closing"

    # A label never seen before splits on the same evidence - no vocabulary.
    t3 = {"text": "Qoh Liquidation", "x0": 400.0, "x1": 520.0}
    splits3 = table_ocr_service._split_compound_header_token(t3, anchors=[420.0, 500.0])
    assert [s["text"] for s in splits3] == ["Qoh", "Liquidation"]

    # Without two bands underneath it there is no evidence of a fusion, so the
    # token is left alone rather than being cut on a guess.
    assert len(table_ocr_service._split_compound_header_token(t1, anchors=[150.0])) == 1
    assert len(table_ocr_service._split_compound_header_token(t1)) == 1


def test_synthesize_columns_from_band():
    hdr_lines = [
        {
            "tokens": [
                {"text": "PRODUCT", "x0": 50.0, "x1": 150.0, "y0": 100.0},
                {"text": "NANE", "x0": 160.0, "x1": 220.0, "y0": 100.0},
                {"text": "PACKING", "x0": 250.0, "x1": 350.0, "y0": 100.0},
                {"text": "Op.Bal.", "x0": 380.0, "x1": 460.0, "y0": 100.0},
                {"text": "Qty.", "x0": 380.0, "x1": 440.0, "y0": 125.0},
            ]
        }
    ]
    cols = table_ocr_service._synthesize_columns_from_band(hdr_lines, [])
    col_names = [c["name"] for c in cols]

    assert "PRODUCT" in col_names or any("product" in c.lower() for c in col_names)
    assert any("pack" in c.lower() for c in col_names)
    assert any("op.bal" in c.lower() for c in col_names)


def test_global_column_model_infers_missing_numeric_columns():
    hdr_lines = [
        {
            "tokens": [
                {"text": "ITEM DESCRIPTION", "x0": 100.0, "x1": 400.0, "y0": 100.0},
                {"text": "ISSUE", "x0": 700.0, "x1": 780.0, "y0": 100.0},
                {"text": "CLOSING", "x0": 850.0, "x1": 950.0, "y0": 100.0},
            ]
        }
    ]
    # Data lines have 4 distinct numeric columns: Opening (~500), Receipt (~600), Issue (~740), Closing (~900)
    data_lines = [
        {
            "tokens": [
                {"text": "DRUG A", "x0": 100.0, "x1": 250.0, "xc": 175.0},
                {"text": "10 TAB", "x0": 420.0, "x1": 460.0, "xc": 440.0},
                {"text": "100", "x0": 490.0, "x1": 520.0, "xc": 505.0},
                {"text": "50", "x0": 590.0, "x1": 620.0, "xc": 605.0},
                {"text": "20", "x0": 730.0, "x1": 755.0, "xc": 742.5},
                {"text": "130", "x0": 890.0, "x1": 920.0, "xc": 905.0},
            ]
        },
        {
            "tokens": [
                {"text": "DRUG B", "x0": 100.0, "x1": 250.0, "xc": 175.0},
                {"text": "10 TAB", "x0": 420.0, "x1": 460.0, "xc": 440.0},
                {"text": "200", "x0": 490.0, "x1": 520.0, "xc": 505.0},
                {"text": "0", "x0": 590.0, "x1": 610.0, "xc": 600.0},
                {"text": "50", "x0": 730.0, "x1": 755.0, "xc": 742.5},
                {"text": "150", "x0": 890.0, "x1": 920.0, "xc": 905.0},
            ]
        },
        {
            "tokens": [
                {"text": "DRUG C", "x0": 100.0, "x1": 250.0, "xc": 175.0},
                {"text": "10 TAB", "x0": 420.0, "x1": 460.0, "xc": 440.0},
                {"text": "300", "x0": 490.0, "x1": 520.0, "xc": 505.0},
                {"text": "100", "x0": 590.0, "x1": 620.0, "xc": 605.0},
                {"text": "80", "x0": 730.0, "x1": 755.0, "xc": 742.5},
                {"text": "320", "x0": 890.0, "x1": 920.0, "xc": 905.0},
            ]
        }
    ]

    col_bounds, _uncertain = table_ocr_service._build_global_column_model(hdr_lines, data_lines, rulings={"horizontal": [], "vertical": []}, img_width=1200)
    col_names = [c[0] for c in col_bounds]

    # The two numeric bands at ~505 and ~605 carry no printed label on this
    # page, so what must be recovered is the COLUMNS - the labelled ones by
    # name, the unlabelled ones as columns holding their data.
    #
    # They are deliberately NOT guessed to be "Opening" and "Receipt". That
    # guess was positional ("first inserted column is Opening, second is
    # Receipt"), true only of the layouts it came from, and it silently
    # mislabelled real data on every other one. An honest placeholder is the
    # correct output when the page shows no label.
    assert any("pack" in c.lower() for c in col_names)
    assert any("issue" in c.lower() for c in col_names)
    assert any("closing" in c.lower() for c in col_names)
    assert len(col_bounds) >= 6

    def _covers(x: float) -> bool:
        return any(x0 <= x <= x1 for _n, x0, x1 in col_bounds)

    assert _covers(505.0), "unlabelled numeric band at ~505 lost"
    assert _covers(605.0), "unlabelled numeric band at ~605 lost"


# --------------------------------------------------------------------------- #
# Vocabulary-free header detection
#
# The pipeline used to find the header row by looking for known words, so a
# document whose columns were labelled with anything unfamiliar ("Qoh", "Liq
# Days", "O.Stk") had no header found at its real position - the band anchored
# somewhere else and every column shifted. Worse, the failure mode was to add
# the new word to a list, which fixes exactly one document and no others.
#
# These tests use labels that appear in NO list anywhere in the codebase, so
# they fail if header detection ever starts depending on vocabulary again.
# --------------------------------------------------------------------------- #

def _line(tokens, y):
    """Build a line dict of (text, x0, x1) tuples at baseline y."""
    toks = [
        {"text": t, "x0": float(x0), "x1": float(x1), "xc": (x0 + x1) / 2.0,
         "y0": float(y), "y1": float(y) + 14.0, "y_center": float(y) + 7.0}
        for (t, x0, x1) in tokens
    ]
    return {
        "tokens": toks,
        "text": " ".join(t["text"] for t in toks),
        "x0": min(t["x0"] for t in toks),
        "x1": max(t["x1"] for t in toks),
        "y0": float(y),
        "y1": float(y) + 14.0,
    }


def _invented_label_document():
    """A stock table whose column labels exist in no keyword list."""
    return [
        _line([("SHRI BALAJI MEDICOS", 100, 400)], 20),
        _line([("GSTIN 09AABCU9603R1ZM", 100, 340)], 45),
        _line([("From 01/05/2026 To 29/05/2026", 100, 380)], 70),
        # The real column-label row - not one of these is a known keyword.
        _line([("Zorb", 100, 180), ("Wix", 300, 350), ("Klar", 420, 470),
               ("Vint", 540, 590), ("Frip", 660, 710)], 100),
        _line([("ALPHA TAB", 100, 220), ("10", 300, 330), ("25", 420, 450),
               ("5", 540, 560), ("30", 660, 690)], 130),
        _line([("BETA CAP", 100, 215), ("40", 300, 330), ("12", 420, 450),
               ("2", 540, 560), ("50", 660, 690)], 160),
        _line([("GAMMA SYP", 100, 225), ("7", 300, 325), ("60", 420, 450),
               ("9", 540, 560), ("58", 660, 690)], 190),
        _line([("DELTA INJ", 100, 220), ("3", 300, 325), ("14", 420, 450),
               ("1", 540, 560), ("16", 660, 690)], 220),
    ]


def test_body_start_found_without_any_keywords():
    lines = _invented_label_document()
    # The letterhead, the GSTIN and the date range all carry digits, but only
    # the product rows form a RUN of numeric-dense lines.
    assert table_ocr_service._find_body_start(lines) == 4


def test_header_row_detected_with_labels_in_no_keyword_list():
    lines = _invented_label_document()
    body_start = table_ocr_service._find_body_start(lines)
    hdr = table_ocr_service._detect_header_line(lines, body_start)
    assert hdr == 3, f"expected the 'Zorb Wix Klar' row, got {hdr}"

    # Proof the detection is genuinely structural: none of those labels is a
    # keyword, so the legacy vocabulary test would have found nothing here.
    hdr_tokens = lines[3]["tokens"]
    clean = [t["text"].lower() for t in hdr_tokens]
    assert not any(
        k in c for c in clean for k in table_ocr_service.HEADER_KEYWORDS
    ), f"test is no longer proving anything - {clean} contains a keyword"


def test_report_period_line_never_anchors_the_header():
    lines = _invented_label_document()
    body_start = table_ocr_service._find_body_start(lines)
    assert table_ocr_service._detect_header_line(lines, body_start) != 2
    assert table_ocr_service._looks_like_report_metadata("From 01/05/2026 To 29/05/2026")
    assert table_ocr_service._looks_like_report_metadata("GSTIN 09AABCU9603R1ZM")
    assert not table_ocr_service._looks_like_report_metadata("Zorb Wix Klar Vint Frip")


def test_column_anchors_measured_from_data_not_labels():
    lines = _invented_label_document()
    anchors = table_ocr_service._column_anchors(lines[4:])
    assert len(anchors) == 4, anchors
    for expected in (315.0, 435.0, 550.0, 675.0):
        assert any(abs(a - expected) < 25.0 for a in anchors), (expected, anchors)


def test_header_detection_declines_when_nothing_aligns():
    # A page with a numeric body but only prose above it: there is no header to
    # find, and inventing one is worse than reporting none.
    lines = [
        _line([("This is a paragraph of running text that wraps", 100, 700)], 20),
        _line([("across two lines and names no columns at all", 100, 690)], 45),
        _line([("ALPHA TAB", 100, 220), ("10", 300, 330), ("25", 420, 450)], 100),
        _line([("BETA CAP", 100, 215), ("40", 300, 330), ("12", 420, 450)], 130),
    ]
    body_start = table_ocr_service._find_body_start(lines)
    assert table_ocr_service._detect_header_line(lines, body_start) is None


def test_body_start_survives_interleaved_non_data_lines():
    # These ERP layouts print a customer name between figure rows. Requiring an
    # unbroken RUN of data-like lines made that first figure row look like more
    # preamble, so it was swallowed into the header band and lost.
    lines = [
        _line([("KAKKAR MEDICOS", 100, 300)], 20),
        _line([("Zorb", 100, 180), ("Wix", 300, 350), ("Klar", 420, 470)], 50),
        _line([("ALPHA TAB", 100, 220), ("20", 300, 330), ("4", 420, 445)], 80),
        _line([("GURMAIL MEDICINE CENTER", 100, 340)], 110),
        _line([("BETA CAP", 100, 215), ("21", 300, 330), ("10", 420, 450)], 140),
        _line([("GAMMA SYP", 100, 225), ("3", 300, 325), ("7", 420, 445)], 170),
    ]
    assert table_ocr_service._find_body_start(lines) == 2


def _banded_document():
    """Header tier + sparse second tier + division banner + data rows."""
    return [
        _line([("SHRI BALAJI MEDICOS", 100, 400)], 20),
        # Main label row: one label over each of the four bands.
        _line([("Zorb", 290, 340), ("Wix", 410, 460), ("Klar", 530, 580),
               ("Vint", 650, 700)], 60),
        # Second header tier: labels only, no figures, at least one standing
        # over a band. Two tokens in two different columns, like the real case
        # ("DESCRIPTION QTY." printed below the top tier) - a tier reaching
        # across columns is what marks it as a label row rather than stray text.
        _line([("Item", 100, 200), ("Frip", 650, 700)], 85),
        # Division banner: no figures AND over no band - a row, not a tier.
        _line([("SPECIAL DIVISION", 100, 260)], 110),
        _line([("ALPHA TAB", 100, 220), ("10", 300, 330), ("25", 420, 450),
               ("5", 540, 560), ("30", 660, 690)], 140),
        _line([("BETA CAP", 100, 215), ("40", 300, 330), ("12", 420, 450),
               ("2", 540, 560), ("50", 660, 690)], 170),
        _line([("GAMMA SYP", 100, 225), ("7", 300, 325), ("60", 420, 450),
               ("9", 540, 560), ("58", 660, 690)], 200),
        _line([("DELTA INJ", 100, 220), ("3", 300, 325), ("14", 420, 450),
               ("1", 540, 560), ("16", 660, 690)], 230),
    ]


def test_header_band_absorbs_sparse_tier_but_not_a_division_banner():
    # Both of these sit between the header row and the data, are digit-free,
    # and carry few tokens - so token counts cannot tell them apart. What
    # separates them is whether their labels stand over a band the data forms.
    lines = _banded_document()
    table, _non_table, col_bounds, _bboxes, _hdr = table_ocr_service._reconstruct_table_grid(
        lines, rulings={"horizontal": [], "vertical": []}, img_width=900
    )
    assert table is not None
    names = [c[0] for c in col_bounds]

    # The sparse tier was absorbed: its labels name columns.
    assert any("frip" in n.lower() for n in names), names
    assert any("item" in n.lower() for n in names), names
    # The banner was NOT absorbed: it is still a row, and its words are in no
    # column name.
    assert not any("division" in n.lower() for n in names), names
    first_cells = [(r[0] or "").lower() for r in table["rows"]]
    assert any("special division" in c for c in first_cells), table["rows"]
    # And every product row survived.
    for prod in ("alpha", "beta", "gamma", "delta"):
        assert any(prod in c for c in first_cells), (prod, first_cells)


def test_product_row_with_a_dash_placeholder_is_not_absorbed():
    # "ADABOR GEL  -" is a real product row whose values are dashes. It has no
    # parsable number, so a digit-count test alone reads it as a header tier
    # and deletes it.
    lines = _banded_document()
    lines.insert(3, _line([("ADABOR GEL", 100, 230), ("-", 655, 670)], 105))
    table, _nt, col_bounds, _b, _h = table_ocr_service._reconstruct_table_grid(
        lines, rulings={"horizontal": [], "vertical": []}, img_width=900
    )
    assert table is not None
    names = [c[0] for c in col_bounds]
    assert not any("adabor" in n.lower() for n in names), names
    first_cells = [(r[0] or "").lower() for r in table["rows"]]
    assert any("adabor" in c for c in first_cells), table["rows"]
