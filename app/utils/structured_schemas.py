from datetime import datetime
from typing import Dict, Any, List
from app.utils.validator_utils import validate_field

# Strings a model writes to mean "this field is not on the document".
#
# They are not fabrication - they are an honest "absent" - but published as
# VALUES they do real damage twice over: an ERP writes the literal text "N/A"
# into a record, and field verification flags them as placeholder filler,
# which dragged a correctly extracted resume from 1.0 to 0.78 and marked it
# for review over two fields the document simply does not have.
#
# So they are normalised to absent at the envelope boundary, where empty
# values are already dropped. An absent field states "not on the document"
# exactly; "N/A" states it in a way every consumer has to special-case.
_ABSENT_MARKERS = frozenset({
    "n/a", "na", "n.a.", "none", "null", "nil", "-", "--", "---",
    "not available", "not applicable", "not provided", "not specified",
    "not mentioned", "unknown", "blank", "empty",
})


def _is_absent_marker(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _ABSENT_MARKERS


def strip_absent_markers(value: Any) -> Any:
    """Recursively turn 'N/A'-style strings into None, in place of a value."""
    if _is_absent_marker(value):
        return None
    if isinstance(value, dict):
        return {k: strip_absent_markers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_absent_markers(v) for v in value]
    return value


def _canonical_identity(key: str, value: Any) -> Any:
    """
    One presentation per identity number, whatever the model returned.

    The same Aadhaar card was published as "6394 1918 1234" in the markdown
    and "639419181234" in the structured field, so a consumer keying on the
    number saw two values for one identity. Grouping only - no digit is
    added, removed or changed.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    low = key.lower()
    if "aadhaar" in low or "aadhar" in low:
        from app.services.id_validation import format_aadhaar
        return format_aadhaar(value)
    if "pan_number" in low or low == "pan":
        return value.strip().upper().replace(" ", "")
    return value


def _has_content(value: Any) -> bool:
    """True when anything survives after the absent markers are removed."""
    if isinstance(value, dict):
        return any(_has_content(v) for v in value.values())
    if isinstance(value, list):
        return any(_has_content(v) for v in value)
    if isinstance(value, str):
        return bool(value.strip())
    return value is not None

DOCUMENT_SCHEMA_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "PAN": {
        "name": None,
        "father_name": None,
        "dob": None,
        "pan_number": None
    },
    "AADHAAR": {
        "name": None,
        # Printed as "S/O <name>" on the letter form. Without a field of its
        # own it was being swallowed into `address`, which both polluted the
        # address and hid the guardian from any consumer looking for it.
        "guardian_name": None,
        "dob": None,
        "gender": None,
        "aadhaar_number": None,
        # The enrollment letter (the tall A4 form, as opposed to the card)
        # carries these and nothing was capturing them. The enrollment number
        # is the only identifier available before an Aadhaar is issued.
        "enrollment_number": None,
        # `enrollment_date` exists to stop it squatting in `dob`.
        #
        # The letter prints its enrollment date vertically down the left
        # edge, and the holder's DOB only on the detachable card at the
        # bottom. With no field of its own, the edge date was landing in
        # `dob` on roughly half of all runs - 05/05/2011 instead of
        # 19/12/2004 on the card measured here. A wrong date of birth on an
        # identity record, intermittently, is exactly the sort of fault that
        # survives single-run testing. Same remedy as `guardian_name`: give
        # the competing value somewhere legitimate to go.
        "enrollment_date": None,
        "mobile": None,
        "address": None,
        "district": None,
        "state": None,
        "pin_code": None
    },
    "PASSPORT": {
        "passport_number": None,
        "surname": None,
        "given_name": None,
        "nationality": None,
        "dob": None,
        "gender": None,
        "place_of_birth": None,
        "date_of_issue": None,
        "date_of_expiry": None
    },
    "DRIVING_LICENSE": {
        "license_number": None,
        "name": None,
        "dob": None,
        "address": None,
        "issue_date": None,
        "expiry_date": None
    },
    "VISITING_CARD": {
        "name": None,
        "designation": None,
        "company": None,
        "phone": None,
        "email": None,
        "website": None,
        "address": None,
        "city": None,
        "state": None,
        "pin_code": None
    },
    "INVOICE": {
        "invoice_number": None,
        "invoice_date": None,
        "vendor_name": None,
        "gst_number": None,
        "buyer": None,
        "subtotal": None,
        "tax": None,
        "grand_total": None,
        "currency": None,
        "line_items": []
    },
    "STOCK_STATEMENT": {
        "distributor_name": None,
        "company_name": None,
        "statement_period": None,
        "summary_totals": {},
        "columns": [],
        "items": []
    },
    "STOCK_SUMMARY": {
        "distributor_name": None,
        "company_name": None,
        "statement_period": None,
        "summary_totals": {},
        "columns": [],
        "items": []
    },
    "STOCK_SALES_REPORT": {
        "distributor_name": None,
        "company_name": None,
        "statement_period": None,
        "summary_totals": {},
        "columns": [],
        "items": []
    },
    "SALES_SUMMARY": {
        "distributor_name": None,
        "company_name": None,
        "statement_period": None,
        "summary_totals": {},
        "columns": [],
        "items": []
    },
    "LEDGER": {
        "account_name": None,
        "statement_period": None,
        "opening_balance": None,
        "closing_balance": None, 
        "columns": [],
        "transactions": []
    },
    "BILL": {
        "invoice_number": None,
        "invoice_date": None,
        "vendor_name": None,
        "gst_number": None,
        "buyer": None,
        "subtotal": None,
        "tax": None,
        "grand_total": None,
        "currency": None,
        "line_items": []
    },
    "RECEIPT": {
        "receipt_number": None,
        "receipt_date": None,
        "received_from": None,
        "amount": None,
        "payment_mode": None,
        "line_items": []
    },
    "PRESCRIPTION": {
        "clinic_or_hospital": None,
        "date": None,
        "patient_name": None,
        "age": None,
        "medications": [],
        "advice": None,
        "doctor_signature": None
    },
    "FORM": {
        "form_title": None,
        "applicant_name": None,
        "registration_id": None,
        "date": None,
        "details": None
    },
    "RESUME": {
        "candidate_name": None,
        "contact_info": {},
        "professional_summary": None,
        "skills": {},
        "work_experience": [],
        "education": [],
        "certifications": [],
        "projects": []
    },
    "SPREADSHEET": {
        "sheet_title": None,
        "columns": [],
        "rows": []
    },
    "UNKNOWN": {
        "title": None,
        "description": None,
        "key_points": []
    }
}


def build_enterprise_envelope(
    doc_type: str,
    raw_extraction: Dict[str, Any],
    cls_confidence: float = 0.98,
    department: str = "General",
    user_id: str = "system",
    request_id: str = "req-1001"
) -> Dict[str, Any]:
    """
    Builds the standardized Enterprise Document Intelligence JSON envelope matching the exact prompt spec.
    """
    doc_type_upper = (doc_type or "UNKNOWN").strip().upper()
    valid_categories = {
        "PAN": "PAN",
        "PAN_CARD": "PAN",
        "AADHAAR": "AADHAAR",
        "AADHAAR_CARD": "AADHAAR",
        "PASSPORT": "PASSPORT",
        "DRIVING_LICENSE": "DRIVING_LICENSE",
        "VISITING_CARD": "VISITING_CARD",
        "VISITING CARD": "VISITING_CARD",
        "INVOICE": "INVOICE",
        "BILL": "BILL",
        "RECEIPT": "RECEIPT",
        "STOCK_STATEMENT": "STOCK_STATEMENT",
        "STOCK_SUMMARY": "STOCK_SUMMARY",
        "STOCK_SALES_REPORT": "STOCK_SALES_REPORT",
        "SALES_SUMMARY": "SALES_SUMMARY",
        "LEDGER": "LEDGER",
        "PRESCRIPTION": "PRESCRIPTION",
        "FORM": "FORM",
        "RESUME": "RESUME",
        "CV": "RESUME",
        "CURRICULUM_VITAE": "RESUME",
        "SPREADSHEET": "SPREADSHEET",
        "EXCEL": "SPREADSHEET",
        "TABLE": "SPREADSHEET",
        "REGISTRATION_FORM": "FORM",
        "HOSPITAL_DOCUMENTS": "FORM",
        "UNKNOWN": "UNKNOWN"
    }
    canonical_type = valid_categories.get(doc_type_upper, doc_type_upper)

    template_dict = DOCUMENT_SCHEMA_TEMPLATES.get(canonical_type, DOCUMENT_SCHEMA_TEMPLATES["UNKNOWN"]).copy()

    raw_dict = raw_extraction if isinstance(raw_extraction, dict) else {}

    # If raw_dict already has envelope format ('fields' key), extract from 'fields'
    if "fields" in raw_dict and isinstance(raw_dict["fields"], dict):
        raw_fields = raw_dict["fields"]
    else:
        raw_fields = raw_dict

    expected_keys = list(template_dict.keys())
    all_keys = list(dict.fromkeys(expected_keys + list(raw_fields.keys())))

    fields_output = {}
    missing_fields = []
    confidence_scores = []
    has_review_flag = False

    ignored_envelope_keys = {
        "document_type", "classification_confidence", "overall_document_quality",
        "overall_confidence", "needs_manual_review", "fields", "processing_summary", "metadata"
    }

    for k in all_keys:
        if k in ignored_envelope_keys:
            continue

        raw_item = raw_fields.get(k)
        val = None
        conf = 0.95

        if isinstance(raw_item, dict) and "value" in raw_item:
            val = raw_item.get("value")
            conf = float(raw_item.get("confidence", 0.95))
        elif raw_item is not None:
            val = raw_item
            conf = 0.95 if val != "" else 0.0

        # "N/A" and friends mean absent, so make them absent - including
        # inside nested objects and lists, which is where they actually turn
        # up (a resume's education entries, a card's missing website).
        val = strip_absent_markers(val)
        if isinstance(val, (dict, list)) and not _has_content(val):
            val = None
        val = _canonical_identity(k, val)

        if val is None or val == "" or val == []:
            missing_fields.append(k)
            fields_output[k] = None
        else:
            is_valid = validate_field(k, val)
            needs_review = (conf < 0.60) or (not is_valid)
            if needs_review:
                has_review_flag = True

            fields_output[k] = val
            confidence_scores.append(conf)

    # Calculate overall confidence
    if confidence_scores:
        overall_conf = round(sum(confidence_scores) / len(confidence_scores), 2)
    else:
        overall_conf = 0.50

    # Determine overall document quality
    if overall_conf >= 0.95:
        doc_quality = "Excellent"
    elif overall_conf >= 0.85:
        doc_quality = "Good"
    elif overall_conf >= 0.70:
        doc_quality = "Average"
    elif overall_conf >= 0.50:
        doc_quality = "Poor"
    else:
        doc_quality = "Unreadable"

    total_fields_cnt = len([k for k in all_keys if k not in ignored_envelope_keys])
    extracted_cnt = total_fields_cnt - len(missing_fields)
    needs_manual_review = has_review_flag or (overall_conf < 0.70)

    # Omit null/empty fields to return ONLY relevant extracted fields (preserve non-empty lists for SPREADSHEET)
    relevant_fields = {}
    for k, v in fields_output.items():
        if v is None or v == "":
            continue
        if isinstance(v, list) and len(v) == 0:
            continue
        relevant_fields[k] = v

    envelope = {
        "document_type": canonical_type,
        "classification_confidence": round(cls_confidence, 2),
        "overall_confidence": overall_conf,
        "needs_manual_review": needs_manual_review,
        "fields": relevant_fields,
        "_audit": {
            "overall_confidence": overall_conf,
            "overall_document_quality": doc_quality,
            "needs_manual_review": needs_manual_review,
            "department": department,
            "user_id": user_id,
            "request_id": request_id,
            "token_usage": {
                "prompt_tokens": 512,
                "completion_tokens": 256,
                "total_tokens": 768
            }
        }
    }

    return envelope


def normalize_structured_data(doc_type: str, data: Dict[str, Any], cls_confidence: float = 0.98) -> Dict[str, Any]:
    """Ensures extracted JSON conforms strictly to enterprise JSON envelope spec."""
    return build_enterprise_envelope(doc_type, data, cls_confidence=cls_confidence)


def create_semantic_cell(raw: Any) -> Dict[str, Any]:
    """
    Creates a semantic cell preserving raw OCR string, normalized value, and semantic flag.
    Distinguishes blank, dash (NOT_REPORTED), and zero (0).
    """
    raw_str = str(raw).strip() if raw is not None else ""
    if not raw_str or raw is None:
        return {"raw": "", "normalized": None, "semantic": "EMPTY"}
    if raw_str in ["-", "--", "---", "NA", "N/A"]:
        return {"raw": raw_str, "normalized": None, "semantic": "NOT_REPORTED"}
    if raw_str in ["0", "0.0", "0.00"]:
        return {"raw": raw_str, "normalized": 0.0, "semantic": "ZERO"}
    try:
        clean_num = float(raw_str.replace(",", ""))
        return {"raw": raw_str, "normalized": clean_num, "semantic": "NUMERIC"}
    except ValueError:
        return {"raw": raw_str, "normalized": raw_str, "semantic": "TEXT"}
