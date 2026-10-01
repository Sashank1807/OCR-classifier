"""
Validation Utility Engine for Enterprise Document Intelligence.
Performs regex pattern validation (PAN, Aadhaar, GST, Email, Phone, PIN Code, Date formats).
"""

import re
from datetime import datetime
from typing import Any


def validate_pan(pan: str) -> bool:
    if not pan or not isinstance(pan, str):
        return False
    clean_pan = pan.strip().upper()
    return bool(re.match(r"^[A-Z]{5}[0-9]{4}[A-Z]{1}$", clean_pan))


def validate_aadhaar(aadhaar: str) -> bool:
    if not aadhaar or not isinstance(aadhaar, str):
        return False
    digits = re.sub(r"\D", "", aadhaar)
    return len(digits) == 12


def validate_gst(gst: str) -> bool:
    if not gst or not isinstance(gst, str):
        return False
    clean_gst = gst.strip().upper()
    return bool(re.match(r"^\d{2}[A-Z]{5}\d{4}[A-Z]{1}[A-Z\d]{1}[Z]{1}[A-Z\d]{1}$", clean_gst))


def validate_email(email: str) -> bool:
    if not email or not isinstance(email, str):
        return False
    return bool(re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", email.strip()))


def validate_phone(phone: Any) -> bool:
    if not phone:
        return False
    phone_str = str(phone)
    digits = re.sub(r"\D", "", phone_str)
    return 10 <= len(digits) <= 12


def validate_pin_code(pin: Any) -> bool:
    if not pin:
        return False
    digits = re.sub(r"\D", "", str(pin))
    return len(digits) == 6


def validate_date(date_str: str) -> bool:
    if not date_str or not isinstance(date_str, str):
        return False
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%d/%m/%y", "%d-%m-%y"):
        try:
            datetime.strptime(date_str.strip(), fmt)
            return True
        except ValueError:
            pass
    return bool(re.search(r"\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}", date_str))


def validate_field(field_name: str, value: Any) -> bool:
    """Validates field value against domain rules."""
    if value is None:
        return True
    val_str = str(value)
    f_lower = field_name.lower()

    if "pan" in f_lower and "company" not in f_lower and "company_pan" not in f_lower:
        return validate_pan(val_str)
    elif "aadhaar" in f_lower:
        return validate_aadhaar(val_str)
    elif "gst" in f_lower:
        return validate_gst(val_str)
    elif "email" in f_lower:
        return validate_email(val_str)
    elif "phone" in f_lower or "mobile" in f_lower or "contact" in f_lower:
        return validate_phone(val_str)
    elif "pin" in f_lower or "pincode" in f_lower or "postal" in f_lower:
        return validate_pin_code(val_str)
    elif "date" in f_lower or "dob" in f_lower:
        return validate_date(val_str)

    return True
