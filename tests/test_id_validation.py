"""
Structural validation of Indian identity numbers.

The failure this prevents is the worst one an ID OCR can produce: a
well-formed, confidently reported, WRONG identity number, written into a
permanent record under the right person's name. A shape rule cannot catch it
- twelve digits is twelve digits - and neither can grounding, because the
misread number is the one printed on the page as far as the OCR is
concerned.

Aadhaar's twelfth digit is a Verhoeff checksum over the other eleven, chosen
because it catches every single-digit substitution and every transposition
of adjacent digits: precisely the two mistakes OCR makes.

Numbers below are from the real documents this was built against.
"""

import pytest

from app.services.field_verification import assess_source_legibility, verify_fields
from app.services.id_validation import (
    PAN_HOLDER_TYPES,
    format_aadhaar,
    is_masked_aadhaar,
    validate_aadhaar,
    validate_pan,
    verhoeff_ok,
)

# A real Aadhaar from a card processed here; its check digit is correct.
REAL_AADHAAR = "639419181234"
REAL_PAN = "AFAPE4843G"
REAL_PAN_NAME = "MANGALAPUDI ESWAR KUMAR REDDY"


# --------------------------------------------------------------------------- #
# Aadhaar: the check digit
# --------------------------------------------------------------------------- #

def test_a_real_aadhaar_validates():
    assert verhoeff_ok(REAL_AADHAAR)
    result = validate_aadhaar(REAL_AADHAAR)
    assert result["valid"] is True
    assert result["status"] == "ok"


@pytest.mark.parametrize("corrupted", [
    "639419181235",   # last digit
    "539419181234",   # first digit
    "639419181334",   # a middle digit
    "649419181234",
])
def test_every_single_digit_misread_is_caught(corrupted):
    assert corrupted != REAL_AADHAAR
    result = validate_aadhaar(corrupted)
    assert result["valid"] is False
    assert result["status"] == "checksum"


@pytest.mark.parametrize("transposed", ["639419181243", "369419181234", "634919181234"])
def test_transposed_adjacent_digits_are_caught(transposed):
    """The other mistake OCR makes, and the reason Verhoeff was chosen."""
    assert validate_aadhaar(transposed)["valid"] is False


def test_an_aadhaar_never_begins_with_zero_or_one():
    """UIDAI reserves that range, so it is a misread rather than a rare number."""
    assert validate_aadhaar("139419181234")["valid"] is False
    assert validate_aadhaar("039419181234")["valid"] is False


@pytest.mark.parametrize("bad", ["63941918123", "6394191812345", "", "abcd efgh ijkl"])
def test_wrong_length_is_a_format_failure_not_a_checksum_one(bad):
    """The distinction matters: one says rescan, the other says re-read."""
    result = validate_aadhaar(bad)
    assert result["valid"] is False
    assert result["status"] == "format"


# --------------------------------------------------------------------------- #
# Masked Aadhaar is correct, not damaged
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("masked", ["XXXX XXXX 1234", "xxxxxxxx1234", "**** **** 1234"])
def test_a_masked_aadhaar_passes(masked):
    """
    UIDAI prints this itself and most systems are supposed to store it.
    Failing it for having no verifiable check digit would flag correct data.
    """
    assert is_masked_aadhaar(masked)
    result = validate_aadhaar(masked)
    assert result["valid"] is True
    assert result["status"] == "masked"


def test_a_full_number_is_not_mistaken_for_a_masked_one():
    assert not is_masked_aadhaar(REAL_AADHAAR)


# --------------------------------------------------------------------------- #
# One presentation per number
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("variant", ["639419181234", "6394 1918 1234", "6394-1918-1234"])
def test_every_spelling_normalises_to_one_form(variant):
    """
    The same card was published as '6394 1918 1234' in the markdown and
    '639419181234' in the structured field - two values for one identity.
    """
    assert format_aadhaar(variant) == "6394 1918 1234"


def test_normalisation_never_changes_a_digit():
    assert format_aadhaar(REAL_AADHAAR).replace(" ", "") == REAL_AADHAAR


def test_something_unparseable_is_returned_untouched():
    assert format_aadhaar("not a number") == "not a number"


# --------------------------------------------------------------------------- #
# PAN
# --------------------------------------------------------------------------- #

def test_a_real_pan_validates_and_reports_its_holder_type():
    result = validate_pan(REAL_PAN, REAL_PAN_NAME)
    assert result["valid"] is True
    assert result["holder_type"] == "Individual"


def test_a_fourth_letter_outside_the_holder_types_is_a_misread():
    """No issued PAN has one, so this is OCR error, not an unusual taxpayer."""
    result = validate_pan("AFAZE4843G")
    assert result["valid"] is False
    assert result["status"] == "holder_type"


@pytest.mark.parametrize("letter", sorted(PAN_HOLDER_TYPES))
def test_every_real_holder_type_is_accepted(letter):
    assert validate_pan(f"ABC{letter}E1234F")["valid"] is True


@pytest.mark.parametrize("bad", ["NOTAPAN123", "AFAPE4843", "AFAPE48431", "", "1234567890"])
def test_malformed_pans_are_rejected(bad):
    assert validate_pan(bad)["valid"] is False


def test_a_surname_mismatch_is_reported_but_never_rejected():
    """
    The Income Tax surname field regularly differs from the printed name,
    especially for South Indian names where the family name comes first.
    Rejecting on it would fail genuine cards, so it is an observation only.
    """
    result = validate_pan("AFAPZ4843G", "MANGALAPUDI ESWAR KUMAR REDDY")
    assert result["valid"] is True
    assert "not an error" in result["surname_hint"]


def test_the_real_card_reads_as_consistent():
    assert validate_pan(REAL_PAN, REAL_PAN_NAME)["surname_hint"] == "consistent"


# --------------------------------------------------------------------------- #
# Wired into verification - the part that actually protects the ERP
# --------------------------------------------------------------------------- #

def _verify(number):
    text = f"Vadlamudi Sashank {number} 19/12/2004 Male"
    return verify_fields(
        {"name": "Vadlamudi Sashank", "aadhaar_number": number},
        ocr_text=text,
        source_legibility=assess_source_legibility(536, 1280, text),
    )


def test_a_correct_aadhaar_reaches_full_confidence():
    result = _verify("6394 1918 1234")
    assert result["overall_confidence"] == 1.0
    assert result["needs_manual_review"] is False


def test_a_misread_aadhaar_is_flagged_even_though_it_is_on_the_page():
    """
    Grounding alone would pass this: the wrong number IS the text the OCR
    produced. Only the check digit knows better.
    """
    result = _verify("6394 1918 1235")
    assert result["needs_manual_review"] is True
    flagged = {f["field"] for f in result["flagged"]}
    assert "aadhaar_number" in flagged
    assert "Verhoeff" in result["flagged"][0]["reason"]


def test_a_masked_aadhaar_still_passes_verification():
    result = _verify("XXXX XXXX 1234")
    assert result["needs_manual_review"] is False


def test_a_misread_pan_is_flagged():
    text = "INCOME TAX DEPARTMENT AFAZE4843G MANGALAPUDI ESWAR KUMAR REDDY"
    result = verify_fields(
        {"pan_number": "AFAZE4843G"},
        ocr_text=text,
        source_legibility=assess_source_legibility(3038, 1932, text),
    )
    assert result["needs_manual_review"] is True
    assert result["fields"][0]["status"] == "flagged"


def test_nothing_is_ever_auto_corrected():
    """
    A failed check identifies a suspect value; it does not know the right
    one. Guessing at an identity number is how a plausible wrong one gets
    written into a permanent record - the same reasoning that stopped
    arithmetic auto-repair.
    """
    wrong = "6394 1918 1235"
    result = _verify(wrong)
    reported = [f for f in result["fields"] if f["field"] == "aadhaar_number"][0]
    assert reported["value"].replace(" ", "") == wrong.replace(" ", "")


# --------------------------------------------------------------------------- #
# Cross-field conflicts
#
# Found by running the same Aadhaar enrollment letter twice, not by reading
# the code: the letter prints its enrollment date down the left edge and the
# holder's DOB only on the detachable card, and `dob` came back as the
# enrollment date on roughly half of all runs. Grounding cannot see this -
# both dates really are printed on the page, so each passes on its own. Only
# the relationship between them is wrong.
# --------------------------------------------------------------------------- #

def _verify_fields(fields):
    text = ("Enrollment No 0704/18002/35507 05/05/2011 Vadlamudi Sashank "
            "DOB 19/12/2004 Male 6394 1918 1234")
    return verify_fields(fields, ocr_text=text,
                         source_legibility=assess_source_legibility(536, 1280, text))


def test_a_dob_copied_from_the_enrollment_date_is_flagged():
    result = _verify_fields({"name": "Vadlamudi Sashank",
                             "dob": "05/05/2011",
                             "enrollment_date": "05/05/2011"})
    assert result["needs_manual_review"] is True
    flagged = {f["field"]: f["reason"] for f in result["flagged"]}
    assert "dob" in flagged
    assert "enrollment_date" in flagged["dob"]


def test_a_genuine_dob_alongside_an_enrollment_date_passes():
    result = _verify_fields({"name": "Vadlamudi Sashank",
                             "dob": "19/12/2004",
                             "enrollment_date": "05/05/2011"})
    assert result["needs_manual_review"] is False


def test_the_check_tolerates_different_date_punctuation():
    """The two fields rarely arrive formatted identically."""
    result = _verify_fields({"dob": "05-05-2011", "enrollment_date": "05/05/2011"})
    assert result["needs_manual_review"] is True


def test_an_aadhaar_number_copied_from_the_enrollment_number_is_flagged():
    """Two different identifiers on the same letter, easily confused."""
    result = _verify_fields({"aadhaar_number": "0704 1800 2355",
                             "enrollment_number": "0704/18002/355"})
    assert result["needs_manual_review"] is True


def test_the_check_does_nothing_when_the_rival_field_is_absent():
    result = _verify_fields({"name": "Vadlamudi Sashank", "dob": "19/12/2004"})
    assert result["needs_manual_review"] is False
