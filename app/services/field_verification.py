"""
Field-level verification for non-tabular documents.

WHY THIS EXISTS
---------------
`overall_confidence` for any document without table validation was the literal
constant 0.95 (ocr_pipeline, "audit_info.get('overall_confidence', 0.95)"), and
the only thing that moved it was a penalty for "expected a table, found none".
So the number measured nothing about the extraction. Measured on five real
documents: a perfect PAN card, a perfect visiting card and a pamphlet whose
entire output was fabricated all scored 0.95, while a near-perfect Aadhaar and
resume scored 0.30 because they were judged table-heavy and produced no table.

Nothing here asks a model how sure it is - a model that invents
`"[Description]"` will happily say it is certain. Instead each field is checked
against evidence that exists independently of the model:

  grounding  does this value actually appear in the OCR text of the page?
  format     does it match the shape its field requires (PAN, Aadhaar, email)?
  legibility was the source readable enough for any of this to mean anything?

Grounding is the one that catches fabrication, because invented text has no
counterpart on the page. It is also the cheapest.

Statuses follow the taxonomy already used in the viewer (STYLE.md section 3):
  exact     from a deterministic source (digital text, not OCR)
  verified  found on the page and correctly shaped
  unchecked recognised, but no check applies to this field
  flagged   failed a check, or the page was not legible
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

EXACT = "exact"
VERIFIED = "verified"
UNCHECKED = "unchecked"
FLAGGED = "flagged"

# Text a template emits when it has nothing to say. A model asked to read an
# illegible page tends to return the form rather than admit defeat, so these
# are treated as failures regardless of anything else.
_PLACEHOLDER_RE = re.compile(
    r"^\s*(?:\[[^\]]*\]|\{[^}]*\}|<[^>]*>|n/?a|none|null|nil|tbd|xxx+|--+|"
    r"lorem\b.*|your\s+\w+|sample\s+\w+|enter\s+\w+|description|placeholder|"
    r"text\s+here|company\s+name|full\s+name)\s*$",
    re.IGNORECASE,
)

# Per-field shape rules. Keyed by a substring of the field name so the same
# rule covers candidate_name/name, phone/mobile/contact_number and so on.
_FORMAT_RULES: List[Tuple[str, re.Pattern, str]] = [
    ("pan_number",     re.compile(r"^[A-Z]{5}[0-9]{4}[A-Z]$"),               "five letters, four digits, a letter"),
    ("aadhaar_number", re.compile(r"^(?:\d{4}\s?\d{4}\s?\d{4}|[X\*]{4}\s?[X\*]{4}\s?\d{4})$", re.I), "12 digits, or masked with the last four"),
    ("gstin",          re.compile(r"^\d{2}[A-Z]{5}\d{4}[A-Z][A-Z0-9]Z[A-Z0-9]$"), "15-character GSTIN"),
    ("ifsc",           re.compile(r"^[A-Z]{4}0[A-Z0-9]{6}$"),                "11-character IFSC"),
    ("pin_code",       re.compile(r"^\d{6}$"),                               "six digits"),
    ("email",          re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$"),       "an email address"),
    ("phone",          re.compile(r"^[+\d][\d\s\-()]{7,19}$"),               "7-20 digits"),
    ("mobile",         re.compile(r"^[+\d][\d\s\-()]{7,19}$"),               "7-20 digits"),
    ("dob",            re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"), "a date"),
    ("date_of_birth",  re.compile(r"^\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}$|^\d{4}-\d{2}-\d{2}$"), "a date"),
]

# Fields whose text is legitimately rewritten rather than copied, so grounding
# them word-for-word would flag correct output.
_NOT_GROUNDABLE = (
    "summary", "objective", "description", "responsibilities", "about",
    "profile", "state", "city", "country", "designation_normalised",
)


def _norm(text: str) -> str:
    """Fold to a comparable form: unicode-normalised, lowercase, alphanumeric."""
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _is_placeholder(value: str) -> bool:
    return bool(_PLACEHOLDER_RE.match(value or ""))


def _format_rule_for(field_path: str) -> Optional[Tuple[re.Pattern, str]]:
    low = field_path.lower()
    for key, pattern, description in _FORMAT_RULES:
        if key in low:
            return pattern, description
    return None


def _cross_field_conflict(field_path: str, value: str, fields: dict) -> Optional[str]:
    """
    Catch a value that has been copied from a neighbouring field.

    Grounding cannot see this: both values are genuinely printed on the page,
    so each passes on its own. Only the relationship between them is wrong.

    The case this was built from, found by running the same Aadhaar twice:
    the enrollment letter prints its enrollment date down the left edge and
    the holder's DOB only on the detachable card, and `dob` came back as the
    enrollment date on about half of all runs. A wrong date of birth on an
    identity record - intermittently, so single-run testing never sees it.
    """
    if not isinstance(fields, dict):
        return None
    low = field_path.lower()
    normalised = re.sub(r"\D", "", value or "")
    if not normalised:
        return None

    if low.endswith("dob") or "date_of_birth" in low:
        for other in ("enrollment_date", "issue_date", "date_of_issue", "print_date"):
            rival = fields.get(other)
            if isinstance(rival, str) and re.sub(r"\D", "", rival) == normalised:
                return (f"identical to {other}; a date of birth and "
                        f"an {other.replace('_', ' ')} are different dates")

    # The same trap in the other direction: the Aadhaar number and the
    # enrollment number are different identifiers on the same letter.
    if "aadhaar_number" in low:
        rival = fields.get("enrollment_number")
        if isinstance(rival, str) and re.sub(r"\D", "", rival) == normalised:
            return "identical to enrollment_number; these are different identifiers"

    return None


def _structural_id_check(field_path: str, value: str):
    """
    (ok, reason, normalised) for identity numbers that carry internal
    structure, or None when the field has none to check.

    Normalisation is presentation only - regrouping an Aadhaar's digits into
    'NNNN NNNN NNNN'. No digit or letter is ever changed: a failed check
    identifies a suspect value, it does not know the right one, and guessing
    at an identity number is how a plausible wrong one gets written into a
    permanent record.
    """
    from app.services.id_validation import validate_aadhaar, validate_pan

    low = field_path.lower()
    if "aadhaar" in low or "aadhar" in low or "uid" in low:
        r = validate_aadhaar(value)
        return bool(r["valid"]), str(r["reason"]), str(r["normalised"])
    if "pan_number" in low or low.endswith("pan"):
        r = validate_pan(value)
        return bool(r["valid"]), str(r["reason"]), str(r["normalised"])
    return None


def _groundable(field_path: str) -> bool:
    low = field_path.lower()
    return not any(k in low for k in _NOT_GROUNDABLE)


def _is_grounded(value: str, haystack: str) -> bool:
    """
    Does this value appear on the page?

    Long values are checked by their distinctive words rather than as one
    string, because OCR line breaks and spacing rarely survive a verbatim
    comparison of a whole paragraph.
    """
    v = _norm(value)
    if len(v) < 3:
        return True                      # too short to be evidence either way
    if v in haystack:
        return True
    words = [w for w in re.split(r"\s+", value) if len(w) > 3]
    if len(words) >= 3:
        hits = sum(1 for w in words if _norm(w) in haystack)
        return hits >= max(2, int(len(words) * 0.6))
    return False


def _walk(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    """Flatten the extracted fields into (path, scalar value) pairs."""
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_walk(v, f"{prefix}[{i}]"))
    elif obj is not None:
        s = str(obj).strip()
        if s:
            out.append((prefix, s))
    return out


def assess_source_legibility(
    width: Optional[int],
    height: Optional[int],
    ocr_text: str,
) -> Dict[str, Any]:
    """
    Was the page readable at all?

    A 148x148 pamphlet thumbnail has body text a few pixels tall. No extraction
    from it can be trusted, and the honest response is to say so rather than to
    return a confident-looking structure. This runs BEFORE the field checks so
    a fabricated result cannot be graded as merely "unverified".
    """
    reasons: List[str] = []
    min_dim = min(width, height) if width and height else None

    if min_dim is not None and min_dim < 400:
        reasons.append(
            f"source image is {width}x{height}; text below ~400px on the short "
            f"edge is not reliably legible"
        )
    words = len(re.findall(r"\w+", ocr_text or ""))
    if words < 5:
        reasons.append(f"only {words} word(s) of text were recognised on the page")

    return {"legible": not reasons, "reasons": reasons,
            "min_dimension": min_dim, "recognised_words": words}


def verify_fields(
    fields: Any,
    ocr_text: str,
    is_digital_source: bool = False,
    source_legibility: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check every extracted field against the page it came from.

    Returns per-field statuses plus a document-level verdict. Nothing is ever
    modified - this reports, exactly like the arithmetic check on the table
    path, which was reduced to detection-only for the same reason: a failed
    check identifies a suspect value, not the correct one.
    """
    haystack = _norm(ocr_text)
    legibility = source_legibility or {"legible": True, "reasons": []}
    pairs = _walk(fields)

    results: List[Dict[str, Any]] = []
    for path, value in pairs:
        entry: Dict[str, Any] = {"field": path, "value": value}

        if _is_placeholder(value):
            entry.update(status=FLAGGED, reason="value is placeholder text, not content read from the page")
            results.append(entry)
            continue

        if not legibility["legible"]:
            entry.update(status=FLAGGED, reason="source page was not legible enough to verify")
            results.append(entry)
            continue

        rule = _format_rule_for(path)
        if rule:
            # Phone patterns allow internal spacing; identifiers like PAN and
            # GSTIN are compared with spacing removed.
            spaced_ok = any(k in path.lower() for k in ("phone", "mobile", "aadhaar"))
            candidate = value if spaced_ok else value.replace(" ", "")
        if rule and not rule[0].match(candidate):
            entry.update(status=FLAGGED, reason=f"does not look like {rule[1]}")
            results.append(entry)
            continue

        # Structural checks that a regex cannot do.
        #
        # A shape rule only proves the OCR returned twelve digits. It cannot
        # distinguish a correct Aadhaar from one with a misread digit - and
        # that is the worst output this system can produce, because a
        # well-formed wrong identity number is written into a permanent
        # record under the right person's name, and every other check passes
        # it (12 digits, present on the page, confident).
        #
        # Aadhaar's 12th digit is a Verhoeff checksum over the other eleven,
        # which catches every single-digit substitution and every adjacent
        # transposition - exactly the two mistakes OCR makes.
        cross = _cross_field_conflict(path, value, fields)
        if cross:
            entry.update(status=FLAGGED, reason=cross)
            results.append(entry)
            continue

        structural = _structural_id_check(path, value)
        if structural is not None:
            ok, reason, normalised = structural
            if normalised and normalised != value:
                entry["value"] = normalised
            if not ok:
                entry.update(status=FLAGGED, reason=reason)
                results.append(entry)
                continue
            entry["structural_check"] = reason

        if is_digital_source:
            # Text came from the file's own text layer, not from pixels.
            entry.update(status=EXACT, reason="read from the document's embedded text")
            results.append(entry)
            continue

        if not _groundable(path):
            entry.update(status=UNCHECKED, reason="free text; no independent check applies")
            results.append(entry)
            continue

        if _is_grounded(value, haystack):
            entry.update(status=VERIFIED, reason="found in the page text")
        else:
            entry.update(status=FLAGGED, reason="not found anywhere in the page text")
        results.append(entry)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in (EXACT, VERIFIED, UNCHECKED, FLAGGED)}
    checked = counts[EXACT] + counts[VERIFIED] + counts[FLAGGED]

    if not legibility["legible"]:
        confidence = 0.10
    elif not results:
        confidence = 0.30            # nothing extracted is not success
    elif checked == 0:
        confidence = 0.50            # nothing could be checked either way
    else:
        confidence = round((counts[EXACT] + counts[VERIFIED]) / checked, 2)

    needs_review = bool(counts[FLAGGED]) or not legibility["legible"] or not results

    return {
        "overall_confidence": confidence,
        "needs_manual_review": needs_review,
        "field_count": len(results),
        "counts": counts,
        "source_legibility": legibility,
        "fields": results,
        "flagged": [r for r in results if r["status"] == FLAGGED],
    }
