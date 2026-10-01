"""
Page orientation correction.

The bug this prevents, measured on a real PAN card rotated 180 degrees:

    "name":        "MANGALAPUDI VEERA VENKATA SUBBARAMI"   <- the father
    "father_name": "MANGALAPUDI ESWAR KUMAR REDDY"         <- the holder

Swapped, at confidence 1.0 with `review_required: false`. Both names really
were on the page, so grounding passed them; only the vertical order had
changed, and the model assigned the two name fields by position. A
confidently wrong identity attribution is the worst thing this system can
emit, and phone photographs of cards lying on a desk arrive upside down all
the time.

These tests use synthetic images rather than the real card, so they run in
the normal suite without a GPU or a model. The real card is covered by
tests/ground_truth/production/pan_real_upside_down.json via
tools/accuracy_harness.py.
"""

import cv2
import numpy as np
import pytest

from app.services.preprocessor import preprocessor


def _page(text_lines=("INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER",
                      "MANGALAPUDI ESWAR KUMAR", "AFAPE4843G", "30/08/2002")):
    """A plain white page with dark text - enough for the probe to score."""
    img = np.full((500, 900, 3), 255, dtype=np.uint8)
    for i, line in enumerate(text_lines):
        cv2.putText(img, line, (40, 90 + i * 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.1, (0, 0, 0), 3, cv2.LINE_AA)
    return img


def _identical(a, b) -> bool:
    return a.shape == b.shape and bool((a == b).all())


# --------------------------------------------------------------------------- #
# It must not touch pages that are already correct
# --------------------------------------------------------------------------- #

def test_an_upright_page_is_returned_untouched():
    page = _page()
    assert _identical(preprocessor.correct_orientation(page), page)


def test_a_blank_page_is_returned_untouched():
    """No text means no evidence; guessing would be worse than doing nothing."""
    blank = np.full((500, 900, 3), 255, dtype=np.uint8)
    assert _identical(preprocessor.correct_orientation(blank), blank)


def test_a_tiny_image_is_skipped_without_probing():
    """Below the size floor the probe is meaningless - and the legibility
    gate will reject the document anyway."""
    tiny = np.full((80, 120, 3), 255, dtype=np.uint8)
    assert _identical(preprocessor.correct_orientation(tiny), tiny)


def test_correction_never_raises_on_malformed_input():
    """The probe must never be the reason a document fails to process."""
    for bad in (np.zeros((10, 10, 3), dtype=np.uint8),
                np.full((300, 300, 3), 127, dtype=np.uint8)):
        assert preprocessor.correct_orientation(bad) is not None


# --------------------------------------------------------------------------- #
# It must fix the case that swapped the names
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(preprocessor._get_probe_engine() is None,
                    reason="RapidOCR not installed; orientation probe unavailable")
def test_an_upside_down_page_is_turned_the_right_way_up():
    upright = _page()
    upside_down = cv2.rotate(upright, cv2.ROTATE_180)

    corrected = preprocessor.correct_orientation(upside_down)

    assert not _identical(corrected, upside_down), "the page was left inverted"
    assert _identical(corrected, upright)


@pytest.mark.skipif(preprocessor._get_probe_engine() is None,
                    reason="RapidOCR not installed; orientation probe unavailable")
def test_the_probe_separates_upright_from_inverted():
    """
    The measurement the whole thing rests on. With the per-line angle
    classifier left ON, an inverted page scores about as well as an upright
    one (measured 175 vs 184) and nothing can be decided; with it OFF the
    recogniser genuinely fails on inverted glyphs.
    """
    page = _page()
    upright = preprocessor._orientation_score(page)
    inverted = preprocessor._orientation_score(cv2.rotate(page, cv2.ROTATE_180))

    assert upright > inverted * 2, (
        f"upright {upright:.0f} vs inverted {inverted:.0f} - the probe cannot "
        f"tell them apart, so the angle classifier is probably enabled again"
    )


def test_the_probe_engine_has_the_angle_classifier_disabled():
    """
    Pinning the detail the fix depends on. RapidOCR's default engine turns
    each detected line upright before recognising it, which is exactly what
    makes an inverted page unreadable to this probe - and readable to the
    normal pipeline, which is why the probe needs its own engine.
    """
    import inspect

    src = inspect.getsource(preprocessor.__class__._get_probe_engine)
    assert "use_cls=False" in src


# --------------------------------------------------------------------------- #
# Cost
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(preprocessor._get_probe_engine() is None,
                    reason="RapidOCR not installed; orientation probe unavailable")
def test_a_readable_page_costs_only_one_probe_pass():
    """
    The second pass only earns its cost when the first looks bad. Without
    this short-circuit every document in a bulk run pays twice for a check
    that almost always finds nothing.
    """
    calls = []
    original = preprocessor._orientation_score

    def counting(image):
        calls.append(1)
        return original(image)

    preprocessor._orientation_score = counting
    try:
        preprocessor.correct_orientation(_page())
    finally:
        preprocessor._orientation_score = original

    assert len(calls) == 1, f"probed {len(calls)} times on an obviously upright page"
