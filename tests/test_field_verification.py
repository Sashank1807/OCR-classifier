"""
Field-level verification.

The failure this exists to prevent, measured on real documents before it was
written: a perfect PAN card, a perfect visiting card and a pamphlet whose
entire output was fabricated all reported confidence 0.95, while a
near-perfect Aadhaar card and resume reported 0.30. Confidence was the literal
default 0.95 plus a penalty that misfired on any dense A4 scan.

These tests pin the property that matters - the score moves with correctness,
in both directions.
"""

import pytest

from app.services.field_verification import (
    EXACT,
    FLAGGED,
    UNCHECKED,
    VERIFIED,
    assess_source_legibility,
    verify_fields,
)

PAN_OCR = ("INCOME TAX DEPARTMENT GOVT. OF INDIA SANJAY KUMAR PRAJAPATI "
           "DUKHIYA PRASHAD PRAJAPATI 02/06/1990 Permanent Account Number BJQPP5524G")
PAN_FIELDS = {
    "name": "Sanjay Kumar Prajapati",
    "father_name": "Dukhiya Prashad Prajapati",
    "dob": "02/06/1990",
    "pan_number": "BJQPP5524G",
}


def _legible(text=PAN_OCR):
    return assess_source_legibility(670, 431, text)


# --------------------------------------------------------------------------- #
# It must accept correct extractions
# --------------------------------------------------------------------------- #

def test_a_correct_extraction_scores_full_confidence():
    r = verify_fields(PAN_FIELDS, PAN_OCR, source_legibility=_legible())
    assert r["overall_confidence"] == 1.0
    assert r["needs_manual_review"] is False
    assert r["counts"][VERIFIED] == 4
    assert r["counts"][FLAGGED] == 0


def test_free_text_is_unchecked_not_flagged():
    """A rewritten summary is not fabricated just because it is not verbatim."""
    fields = {"name": "Sanjay Kumar Prajapati",
              "professional_summary": "A completely reworded paraphrase of the page."}
    r = verify_fields(fields, PAN_OCR, source_legibility=_legible())
    statuses = {f["field"]: f["status"] for f in r["fields"]}
    assert statuses["name"] == VERIFIED
    assert statuses["professional_summary"] == UNCHECKED
    assert r["needs_manual_review"] is False


def test_digital_text_sources_are_marked_exact():
    r = verify_fields(PAN_FIELDS, PAN_OCR, is_digital_source=True, source_legibility=_legible())
    assert r["counts"][EXACT] == 4
    assert r["overall_confidence"] == 1.0


# --------------------------------------------------------------------------- #
# It must reject fabrication - the case that motivated all of this
# --------------------------------------------------------------------------- #

def test_fabricated_placeholders_are_flagged():
    """
    The pamphlet result that scored 0.95 with needs_review False: nine rows of
    "[Description]" invented from a 191x148 thumbnail.
    """
    fabricated = {
        "title": "Our Services",
        "key_points": [
            {"category": "Everyday", "point": "1", "description": "[Description]"},
            {"category": "Specialty", "point": "2", "description": "[Description]"},
        ],
    }
    r = verify_fields(fabricated, "Our Services", source_legibility=_legible("Our Services"))
    assert r["needs_manual_review"] is True
    assert r["overall_confidence"] < 0.5
    assert any("[Description]" == f["value"] for f in r["flagged"])


@pytest.mark.parametrize("junk", ["N/A", "TBD", "XXXX", "Your Company", "lorem ipsum dolor",
                                  "<value>", "{name}", "placeholder"])
def test_template_filler_never_counts_as_content(junk):
    r = verify_fields({"company": junk}, "a page with real words on it",
                      source_legibility=_legible("a page with real words on it"))
    assert r["fields"][0]["status"] == FLAGGED


def test_a_value_absent_from_the_page_is_flagged():
    r = verify_fields({"name": "Someone Never Printed Here"}, PAN_OCR,
                      source_legibility=_legible())
    assert r["fields"][0]["status"] == FLAGGED
    assert "not found" in r["fields"][0]["reason"]


def test_malformed_identifiers_are_flagged():
    for field, bad in (("pan_number", "NOTAPAN123"),
                       ("email", "not-an-email"),
                       ("pin_code", "12"),
                       ("aadhaar_number", "12345")):
        r = verify_fields({field: bad}, f"page containing {bad}",
                          source_legibility=_legible(f"page containing {bad}"))
        assert r["fields"][0]["status"] == FLAGGED, f"{field}={bad} was not flagged"


def test_well_formed_identifiers_pass():
    fields = {"pan_number": "BJQPP5524G", "email": "arjun.mehta@example.com",
              "pin_code": "500033", "aadhaar_number": "XXXX XXXX 1234"}
    text = "BJQPP5524G arjun.mehta@example.com 500033 XXXX XXXX 1234"
    r = verify_fields(fields, text, source_legibility=_legible(text))
    assert r["counts"][FLAGGED] == 0, r["flagged"]


# --------------------------------------------------------------------------- #
# Legibility gate
# --------------------------------------------------------------------------- #

def test_a_thumbnail_is_reported_as_illegible():
    leg = assess_source_legibility(191, 148, "Our Services")
    assert leg["legible"] is False
    assert any("400px" in r for r in leg["reasons"])


def test_a_normal_scan_is_legible():
    assert assess_source_legibility(1024, 1024, PAN_OCR)["legible"] is True


def test_nothing_from_an_illegible_page_is_trusted():
    r = verify_fields({"name": "Rajesh Kumar"}, "Rajesh Kumar",
                      source_legibility=assess_source_legibility(148, 148, "Rajesh Kumar"))
    assert r["overall_confidence"] <= 0.1
    assert r["needs_manual_review"] is True
    assert r["fields"][0]["status"] == FLAGGED


def test_a_page_with_almost_no_text_is_illegible():
    assert assess_source_legibility(2000, 2000, "ab cd")["legible"] is False


def test_an_empty_extraction_is_not_success():
    r = verify_fields({}, PAN_OCR, source_legibility=_legible())
    assert r["needs_manual_review"] is True
    assert r["overall_confidence"] < 0.5


# --------------------------------------------------------------------------- #
# The documented limit
# --------------------------------------------------------------------------- #

def test_grounding_cannot_detect_a_value_in_the_wrong_field():
    """
    Known limit of GROUNDING, still true and still worth pinning.

    Grounding asks only whether a value appears on the page, so a value in
    the wrong field passes. Catching misplacement needs positional evidence.

    The real case this was written from - the resume whose degree carried the
    school's 71% instead of its CGPA 7.29 - has since been fixed, but NOT
    here: it was a routing bug. The resume was being run through the
    coordinate table pipeline, which clustered its two columns into single
    rows and handed the extractor "Bachelor of Pharmacy Higher Secondary
    (WBCHSE)" as one line. See tests/test_pipeline_routing.py.

    That is the useful shape of it: verification could never have caught
    this, because every value really was on the page. The fix had to be
    upstream, in how the page was read.
    """
    r = verify_fields({"gpa_or_grade": "71%"},
                      "B.Pharm CGPA- 7.29 Higher Secondary Result- 71%",
                      source_legibility=_legible("B.Pharm CGPA- 7.29 Higher Secondary Result- 71%"))
    assert r["fields"][0]["status"] == VERIFIED   # present, though misassigned
    assert r["needs_manual_review"] is False
