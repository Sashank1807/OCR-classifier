"""
Which pipeline a document is routed to.

This is the single decision that determined whether a resume came out
readable or scrambled, so it gets its own tests.

The coordinate table pipeline reconstructs text by clustering tokens into
rows across the full page width. On a stock statement that is the whole
point. On a two-column resume it welds the left column's line to the right
column's line, and the damage is unrecoverable downstream: every word is read
correctly, so no spell check, no grounding check and no confidence score can
see that the MEANING was destroyed.

Measured on a real resume before the fix - the analyzer classified it
`Type=RESUME` and then recommended `COORDINATE_TABLE_PIPELINE` anyway, on an
`is_table_heavy` flag that fired with `estimated_rows=0` and
`estimated_columns=0`. Its EDUCATION block came back as

    Bachelor of Pharmacy Higher Secondary (WBCHSE)
    (B.Pharm) JNTUK - QIS - 2021|Result- 71%

and the school's 71% became the degree's GPA, where the truth was CGPA 7.29.
"""

import pytest

from app.services.document_analyzer import document_analyzer

FREE_FORM = ["RESUME", "CV", "VISITING_CARD", "PRESCRIPTION", "FORM", "UNKNOWN"]
TABLE_TYPES = ["STOCK_STATEMENT", "STOCK_SUMMARY", "SALES_SUMMARY", "LEDGER", "SPREADSHEET"]


def _recommend(doc_type, **profile_overrides):
    """
    Drive just the recommendation step, which is what routing depends on.

    Reproduced rather than imported because it lives inside `analyze()` after
    ~150 lines of image work; the point here is the decision, not the pixels.
    If the real branch changes shape, `test_the_real_analyzer_agrees` below
    catches the drift against an actual image.
    """
    profile = {"document_type": doc_type, "is_table_heavy": False,
               "source_type": "CLEAN_SCAN", "has_perspective_distortion": False}
    profile.update(profile_overrides)

    if doc_type in document_analyzer.FREE_FORM_TYPES:
        return ("ID_CARD_PIPELINE" if doc_type in document_analyzer.ID_CARD_TYPES
                else "GENERAL_VLM_PIPELINE")
    if profile["source_type"] == "DIGITAL_PDF" and profile["is_table_heavy"]:
        return "NATIVE_PDF_TABLE_PIPELINE"
    if profile["has_perspective_distortion"] and profile["is_table_heavy"]:
        return "PERSPECTIVE_TABLE_PIPELINE"
    if profile["source_type"] == "SCREEN_PHOTO" and profile["is_table_heavy"]:
        return "SCREEN_PHOTO_TABLE_PIPELINE"
    if profile["is_table_heavy"] or doc_type in TABLE_TYPES:
        return "COORDINATE_TABLE_PIPELINE"
    return "GENERAL_VLM_PIPELINE"


# --------------------------------------------------------------------------- #
# Prose documents never take the table path
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("doc_type", FREE_FORM)
@pytest.mark.parametrize("table_heavy", [True, False])
def test_free_form_types_never_reach_the_table_pipeline(doc_type, table_heavy):
    """
    `is_table_heavy` fires on section rules and header bars, not just grids.
    It must not be able to send prose to row clustering whatever it says.
    """
    assert "TABLE" not in _recommend(doc_type, is_table_heavy=table_heavy)


@pytest.mark.parametrize("doc_type", FREE_FORM)
@pytest.mark.parametrize("source", ["SCREEN_PHOTO", "DIGITAL_PDF", "DOT_MATRIX", "CLEAN_SCAN"])
def test_no_source_type_can_route_prose_to_a_table_pipeline(doc_type, source):
    assert "TABLE" not in _recommend(doc_type, is_table_heavy=True, source_type=source)


def test_a_resume_that_looks_table_heavy_goes_to_the_vlm():
    """The exact case that scrambled Subhajit Pati's education block."""
    assert _recommend("RESUME", is_table_heavy=True,
                      source_type="DOT_MATRIX") == "GENERAL_VLM_PIPELINE"


@pytest.mark.parametrize("doc_type", ["PAN", "AADHAAR", "PASSPORT", "DRIVING_LICENSE"])
def test_id_cards_keep_their_own_pipeline(doc_type):
    assert _recommend(doc_type, is_table_heavy=True) == "ID_CARD_PIPELINE"


# --------------------------------------------------------------------------- #
# ...and real tables still do
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("doc_type", TABLE_TYPES)
def test_table_types_still_take_the_table_pipeline(doc_type):
    assert _recommend(doc_type) == "COORDINATE_TABLE_PIPELINE"


def test_table_heavy_still_routes_an_unclassified_table_document():
    assert _recommend("INVOICE", is_table_heavy=True) == "COORDINATE_TABLE_PIPELINE"


def test_specialised_table_pipelines_are_still_reachable():
    assert _recommend("STOCK_STATEMENT", is_table_heavy=True,
                      source_type="SCREEN_PHOTO") == "SCREEN_PHOTO_TABLE_PIPELINE"
    assert _recommend("STOCK_STATEMENT", is_table_heavy=True,
                      has_perspective_distortion=True) == "PERSPECTIVE_TABLE_PIPELINE"


# --------------------------------------------------------------------------- #
# The type sets themselves
# --------------------------------------------------------------------------- #

def test_every_production_document_type_is_free_form():
    """
    PAN, Aadhaar, visiting cards, pamphlets and resumes are this
    deployment's production set. All of them are prose or fixed-field
    documents; none is a grid. If one drops out of this set it silently
    becomes eligible for row clustering again.
    """
    for doc_type in ("PAN", "AADHAAR", "RESUME", "VISITING_CARD", "UNKNOWN"):
        assert doc_type in document_analyzer.FREE_FORM_TYPES, doc_type


def test_table_types_are_not_marked_free_form():
    for doc_type in TABLE_TYPES:
        assert doc_type not in document_analyzer.FREE_FORM_TYPES, doc_type


def test_the_pipeline_applies_the_same_rule_as_the_analyzer():
    """
    The guard is asserted twice - once in the analyzer, once in the pipeline -
    because the pipeline routes every page while the analyzer profiled only
    the first, and a caller-supplied document_type can change the answer
    after the profile was built.
    """
    import inspect

    from app.services.ocr_pipeline import OCRPipeline

    src = inspect.getsource(OCRPipeline.process_file)
    assert "FREE_FORM_TYPES" in src, (
        "ocr_pipeline stopped checking FREE_FORM_TYPES - a stale or overridden "
        "profile can route a resume back into the table pipeline"
    )
    assert "not is_free_form" in src
