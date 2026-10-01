"""
Structural validation of Indian identity numbers.

A format check ("twelve digits") only proves the OCR returned something
shaped like an Aadhaar number. It cannot tell a correct number from one with
a misread digit, and a misread digit is the single worst failure this system
can produce: a well-formed, confident, WRONG identity number written into an
ERP record, attached to the right person's name.

Both of these numbers carry structure that catches exactly that.

AADHAAR - Verhoeff check digit
    The twelfth digit is a Verhoeff checksum over the first eleven. Verhoeff
    is used precisely because it catches the two errors OCR actually makes:
    every single-digit substitution, and every transposition of adjacent
    digits. Measured on a real card here, 639419181234 validates, while
    639419181235, 539419181234 and 639419181243 - one substitution at each
    end and one transposition - all fail. That is a real misread caught for
    free, with no second model and no extra inference time.

PAN - positional grammar
    `AAAAA9999A`, where the 4th letter encodes the holder type (P for an
    individual, C a company, H a HUF, F a firm ...). A 4th letter outside
    that set means the OCR misread it, because no issued PAN has one.

    The 5th letter is the first letter of the holder's surname as the Income
    Tax record spells it. That is genuinely useful evidence, but it is NOT
    used to reject: the record's surname field regularly differs from what is
    printed on the card, especially for South Indian names where the family
    name is written first. On the real card checked here - MANGALAPUDI ESWAR
    KUMAR REDDY, PAN AFAPE4843G - the 5th letter is E, matching neither
    MANGALAPUDI nor REDDY. The card is genuine. So this is reported as an
    observation for a human, never as a failure.

Nothing here corrects a value. A failed check identifies a number as
suspect; it does not know the right one, and guessing at identity numbers is
how you write a plausible wrong one into a permanent record.
"""

import re
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Verhoeff (dihedral group D5) - multiplication, permutation, inverse tables
# --------------------------------------------------------------------------- #

_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)

_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2),
    (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0),
    (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5),
    (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)

# A masked Aadhaar is what UIDAI itself prints on e-Aadhaar and what most
# downstream systems are supposed to store. It is CORRECT, not damaged, and
# must never be failed for having no verifiable check digit.
_MASK_CHARS = set("xX*•")

_AADHAAR_SHAPE = re.compile(r"^(?:\d{4}\s?\d{4}\s?\d{4})$")
_PAN_SHAPE = re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$")

# 4th character of a PAN: the holder type. This list is fixed by the Income
# Tax Department, so anything else is a misread, not an unusual taxpayer.
PAN_HOLDER_TYPES = {
    "P": "Individual",
    "C": "Company",
    "H": "Hindu Undivided Family",
    "F": "Firm / Limited Liability Partnership",
    "A": "Association of Persons",
    "T": "Trust",
    "B": "Body of Individuals",
    "L": "Local Authority",
    "J": "Artificial Juridical Person",
    "G": "Government",
    "K": "Krish (Trust under Krishi)",
}


def digits_only(value: str) -> str:
    return re.sub(r"\D", "", value or "")


def is_masked_aadhaar(value: str) -> bool:
    """True for 'XXXX XXXX 1234' and friends - masked, not malformed."""
    if not value:
        return False
    compact = re.sub(r"[\s-]", "", str(value))
    if len(compact) != 12:
        return False
    masked = sum(1 for ch in compact if ch in _MASK_CHARS)
    return masked >= 4 and all(ch in _MASK_CHARS or ch.isdigit() for ch in compact)


def verhoeff_ok(number: str) -> bool:
    """Verhoeff check over the full 12 digits (the 12th is the check digit)."""
    digits = digits_only(number)
    if len(digits) != 12:
        return False
    checksum = 0
    for i, ch in enumerate(reversed(digits)):
        checksum = _D[checksum][_P[i % 8][int(ch)]]
    return checksum == 0


def format_aadhaar(value: str) -> str:
    """
    Canonical presentation: three groups of four.

    The same card was being published as '6394 1918 1234' in the markdown and
    '639419181234' in the structured field. A consumer keying on the number
    then sees two different values for one identity, so both are normalised
    to the spaced form UIDAI prints.
    """
    if not value:
        return value
    compact = re.sub(r"[\s-]", "", str(value).strip())
    if len(compact) == 12 and all(ch.isdigit() or ch in _MASK_CHARS for ch in compact):
        return f"{compact[0:4]} {compact[4:8]} {compact[8:12]}"
    return str(value).strip()


def validate_aadhaar(value: str) -> Dict[str, object]:
    """
    Returns {valid, status, reason, normalised}.

    status is one of:
        ok       - 12 digits and the check digit agrees
        masked   - deliberately masked, check digit not verifiable
        checksum - 12 digits but the check digit disagrees: A DIGIT IS WRONG
        format   - not twelve digits at all
    """
    raw = str(value or "").strip()
    if not raw:
        return {"valid": False, "status": "format", "reason": "empty", "normalised": raw}

    if is_masked_aadhaar(raw):
        return {"valid": True, "status": "masked", "normalised": format_aadhaar(raw),
                "reason": "masked Aadhaar; the check digit cannot be verified"}

    compact = re.sub(r"[\s-]", "", raw)
    if not _AADHAAR_SHAPE.match(compact) or len(compact) != 12:
        return {"valid": False, "status": "format", "normalised": raw,
                "reason": f"expected 12 digits, found {len(digits_only(raw))}"}

    # UIDAI never issues a number beginning 0 or 1 - that range is reserved so
    # an Aadhaar cannot be confused with other identifiers.
    if compact[0] in "01":
        return {"valid": False, "status": "checksum", "normalised": format_aadhaar(compact),
                "reason": "an Aadhaar number never begins with 0 or 1"}

    if not verhoeff_ok(compact):
        return {"valid": False, "status": "checksum", "normalised": format_aadhaar(compact),
                "reason": "the Verhoeff check digit does not match, so at least "
                          "one digit was misread"}

    return {"valid": True, "status": "ok", "normalised": format_aadhaar(compact),
            "reason": "check digit verified"}


def validate_pan(value: str, name: Optional[str] = None) -> Dict[str, object]:
    """
    Returns {valid, status, reason, normalised, holder_type, surname_hint}.

    `name`, when supplied, is used only to describe whether the 5th letter
    agrees with the printed name. It never changes `valid` - see the module
    docstring for the real card that would otherwise be rejected.
    """
    raw = str(value or "").strip().upper().replace(" ", "")
    if not raw:
        return {"valid": False, "status": "format", "reason": "empty", "normalised": raw}

    if not _PAN_SHAPE.match(raw):
        return {"valid": False, "status": "format", "normalised": raw,
                "reason": "expected five letters, four digits and a letter"}

    holder = PAN_HOLDER_TYPES.get(raw[3])
    if holder is None:
        return {"valid": False, "status": "holder_type", "normalised": raw,
                "reason": f"'{raw[3]}' is not a PAN holder-type letter, so the "
                          f"4th character was misread"}

    result = {"valid": True, "status": "ok", "normalised": raw,
              "holder_type": holder, "reason": "format and holder type valid"}

    if name:
        initials = {token[0].upper() for token in re.findall(r"[A-Za-z]+", name) if token}
        result["surname_hint"] = (
            "consistent" if raw[4] in initials else
            f"the 5th letter '{raw[4]}' matches no word in the printed name "
            f"(the Income Tax surname field often differs, so this is not an error)"
        )
    return result
