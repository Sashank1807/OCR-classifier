import time
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import logger
from app.models.document import DocumentRecord, DocumentPage, DocumentResult
from app.services.preprocessor import preprocessor
from app.services.pdf_service import pdf_service
from app.services.excel_service import excel_service
from app.services.word_service import word_service
from app.services.rapid_ocr_service import rapid_ocr_service
from app.services.model_service import model_engine, get_token_usage, reset_token_usage
from app.services.field_verification import assess_source_legibility, verify_fields
from app.services.document_analyzer import document_analyzer
from app.services.table_ocr_service import table_ocr_service
from app.services.validation_service import validation_service
from app.prompts.ocr_prompts import (
    DOCUMENT_CLASSIFICATION_PROMPT,
    ENGLISH_TRANSLATION_RULE,
    VERBATIM_OCR_PROMPT,
    SPECIALIZED_HANDWRITING_OCR_PROMPT,
    SPECIALIZED_OCR_PROMPTS,
    DOCUMENT_GENERAL_OCR_PROMPT,
    STRUCTURED_EXTRACTION_PROMPTS
)
from app.utils.file_utils import (
    parse_json_from_llm_response,
    extract_markdown_tables,
    convert_markdown_table_to_spreadsheet_json,
    sanitize_row_wise_table,
    auto_align_billing_table_columns,
    map_invoice_column_roles,
    normalize_no_text_response,
    strip_markdown,
    strip_footer_metadata,
    html_table_to_markdown,
    clean_markdown_fence,
    sanitize_extracted_markdown
)
from app.utils.structured_schemas import normalize_structured_data


class OCRPipeline:
    """Master Pipeline orchestrating PDF/Image/Excel/Word processing, pre-processing, OCR, and Enterprise Archival."""

    def process_file(
        self,
        db: Session,
        file_path: Union[Path, str],
        original_filename: str,
        requested_doc_type: str = "AUTO",
        existing_doc_id: Optional[int] = None
    ) -> Dict[str, Any]:
        file_path = Path(file_path)
        start_time = time.time()
        file_size = file_path.stat().st_size
        ext = file_path.suffix.lower().lstrip(".")

        logger.info(f"Starting OCR Pipeline for '{original_filename}' ({ext}, {file_size} bytes)")

        # Start this document's token count. Thread-local, so concurrent
        # async jobs do not bill each other.
        reset_token_usage()

        # 1. Convert or extract page images and check for embedded digital text, Excel, or Word documents
        page_images = []
        is_digital_doc = False
        digital_results = []

        if ext in ["xlsx", "xls", "csv"]:
            ex_md, ex_plain, ex_pages = excel_service.extract_excel_content(file_path)
            is_digital_doc = True
            digital_results = ex_pages
            page_count = len(ex_pages)
            page_images = [(i + 1, file_path) for i in range(page_count)]
            logger.info(f"Native Excel/CSV spreadsheet detected for '{original_filename}'. Applying 100% accurate pandas extraction.")
        elif ext in ["docx", "doc"]:
            w_md, w_plain, w_pages = word_service.extract_word_content(file_path)
            is_digital_doc = True
            digital_results = w_pages
            page_count = len(w_pages)
            page_images = [(i + 1, file_path) for i in range(page_count)]
            logger.info(f"Native Word document detected for '{original_filename}'. Applying 100% accurate python-docx extraction.")
        elif ext in ["txt", "text", "log"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    txt_raw = f.read()
            except Exception:
                with open(file_path, "r", encoding="latin-1", errors="replace") as f:
                    txt_raw = f.read()
            formatted_lines = []
            for raw_line in txt_raw.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in re.split(r'\s{2,}', line) if p.strip()]
                if len(parts) >= 2:
                    formatted_lines.append("| " + " | ".join(parts) + " |")
                else:
                    formatted_lines.append(line)
            txt_md = "\n".join(formatted_lines)
            is_digital_doc = True
            digital_results = [{
                "page_number": 1,
                "image_path": "",
                "markdown": txt_md,
                "processing_time": 0.01
            }]
            page_count = 1
            page_images = [(1, file_path)]
            logger.info(f"Native plain text document detected for '{original_filename}'. Parsed {len(formatted_lines)} lines with pipe alignment.")
        elif ext == "pdf":
            page_images = pdf_service.convert_pdf_to_images(file_path)
            has_digital, d_md, d_results = pdf_service.extract_digital_pdf_content(file_path)
            if has_digital and len(d_md.strip()) > 50:
                is_digital_doc = True
                digital_results = d_results
                logger.info(f"Digital vector text stream detected for '{original_filename}'. Applying 100% accurate native table extraction.")
            page_count = len(page_images)
        else:
            page_images = [(1, file_path)]
            page_count = 1

        logger.info(f"Processing {page_count} page(s) for '{original_filename}' (Digital Native: {is_digital_doc})")

        per_page_results = []
        full_markdown_parts = []
        full_raw_text_parts = []

        if existing_doc_id:
            db.query(DocumentPage).filter(DocumentPage.document_id == existing_doc_id).delete()
            db.query(DocumentResult).filter(DocumentResult.document_id == existing_doc_id).delete()
            rec = db.query(DocumentRecord).filter(DocumentRecord.id == existing_doc_id).first()
            if rec:
                rec.page_count = page_count
                rec.status = f"PROCESSING (Page 1 of {page_count})"
            db.commit()

        # Determine Document Profile & Recommended Pipeline prior to Pass 1
        img_check_path = page_images[0][1] if (page_images and not is_digital_doc) else None
        doc_profile = document_analyzer.analyze(
            file_path=file_path,
            image_path=img_check_path,
            requested_type=requested_doc_type
        )

        req_type_clean = (requested_doc_type or "AUTO").strip().upper()
        detected_doc_type = "UNKNOWN"
        cls_confidence = 0.98

        fname_lower = original_filename.lower()
        if req_type_clean != "AUTO":
            detected_doc_type = req_type_clean
            cls_confidence = 1.0
        elif doc_profile.get("document_type") != "UNKNOWN":
            detected_doc_type = doc_profile["document_type"]
            cls_confidence = doc_profile.get("classification_confidence", 0.95)
        elif ext in ["xlsx", "xls", "csv", "txt", "text", "log"]:
            detected_doc_type = "SPREADSHEET"
            cls_confidence = 1.0
        elif ext in ["docx", "doc"]:
            detected_doc_type = "RESUME"
            cls_confidence = 1.0
        elif "pan" in fname_lower:
            detected_doc_type = "PAN"
            cls_confidence = 0.99
        elif "aadhaar" in fname_lower or "aadhar" in fname_lower:
            detected_doc_type = "AADHAAR"
            cls_confidence = 0.99
        elif "passport" in fname_lower:
            detected_doc_type = "PASSPORT"
            cls_confidence = 0.99
        elif "driving" in fname_lower or "license" in fname_lower or "licence" in fname_lower:
            detected_doc_type = "DRIVING_LICENSE"
            cls_confidence = 0.99
        elif "resume" in fname_lower or "cv" in fname_lower or "biodata" in fname_lower:
            detected_doc_type = "RESUME"
            cls_confidence = 0.99
        elif any(k in fname_lower for k in ["stock", "sales", "sst", "inventory", "st-", "sas", "lifecare", "medica"]):
            detected_doc_type = "STOCK_STATEMENT"
            cls_confidence = 0.99
        elif any(k in fname_lower for k in ["bill", "invoice", "receipt", "drug_agency", "voucher"]):
            detected_doc_type = "INVOICE"
            cls_confidence = 0.99
        elif any(k in fname_lower for k in ["excel", "sheet", "budget", "grid", "matrix"]):
            detected_doc_type = "SPREADSHEET"
            cls_confidence = 0.99
        elif not is_digital_doc and page_images:
            try:
                first_pre = preprocessor.process(page_images[0][1], profile=doc_profile)
                cls_raw = model_engine.predict(first_pre, DOCUMENT_CLASSIFICATION_PROMPT)
                cls_json = parse_json_from_llm_response(cls_raw)
                detected_doc_type = cls_json.get("document_type", "UNKNOWN").upper()
                cls_confidence = float(cls_json.get("classification_confidence", 0.98))
                logger.info(f"Visual classification for '{original_filename}': {detected_doc_type} (conf={cls_confidence})")
            except Exception as ce:
                logger.warning(f"Visual pre-classification skipped: {ce}")

        # A page too small to read cannot be classified either.
        #
        # Classification runs on the same pixels the extractor will use, so
        # when those pixels are unreadable the type is a guess dressed up as
        # a fact - and it is not a harmless one: the type selects the schema
        # and the prompt, so a wrong guess makes every later stage look for
        # the wrong fields. Measured, a 180x261 pamphlet thumbnail whose body
        # text is illegible even to a human was classified PRESCRIPTION off
        # the words "CLINIC" and "MEDICAL", and was then asked for a
        # patient name and a medication list it could never have.
        #
        # UNKNOWN is the honest answer, and its schema is the general one, so
        # whatever IS readable still comes through.
        if not is_digital_doc and page_images and detected_doc_type != "UNKNOWN":
            try:
                import cv2 as _cv2
                _probe = _cv2.imread(str(page_images[0][1]))
                if _probe is not None:
                    _ph, _pw = _probe.shape[:2]
                    _cls_leg = assess_source_legibility(_pw, _ph, "")
                    if not _cls_leg["legible"] and _cls_leg.get("min_dimension") is not None:
                        logger.warning(
                            f"'{original_filename}' was classified {detected_doc_type}, but the "
                            f"page is not legible ({'; '.join(_cls_leg['reasons'])}). Falling back "
                            f"to UNKNOWN rather than applying a schema chosen from unreadable text."
                        )
                        detected_doc_type = "UNKNOWN"
                        cls_confidence = min(cls_confidence, 0.30)
            except Exception as _ce:
                logger.warning(f"Legibility check on the classification skipped: {_ce}")

        # Choose the optimal specialized OCR prompt for this document type
        pass1_ocr_prompt = SPECIALIZED_OCR_PROMPTS.get(detected_doc_type, DOCUMENT_GENERAL_OCR_PROMPT)
        logger.info(f"Applying tailored Pass 1 OCR prompt for document type: {detected_doc_type}")

        # Determine processing pipeline
        rec_pipeline = doc_profile.get("recommended_pipeline", "GENERAL_VLM_PIPELINE")
        is_id_card = detected_doc_type in document_analyzer.ID_CARD_TYPES
        is_stock_doc = detected_doc_type in ["STOCK_STATEMENT", "STOCK_SUMMARY", "STOCK_SALES_REPORT", "SALES_SUMMARY"]

        # Asserted here as well as in the analyzer, deliberately. The analyzer
        # decides on the FIRST page's profile, this routes EVERY page, and the
        # classification can be overridden by the caller's document_type after
        # the profile was built. One guard in one of those places is not
        # enough: a resume that reaches the table pipeline comes out with its
        # columns interleaved, and nothing downstream can undo that.
        is_free_form = detected_doc_type in document_analyzer.FREE_FORM_TYPES
        is_table_pipeline = (
            not is_id_card and not is_free_form and (
                rec_pipeline in ["COORDINATE_TABLE_PIPELINE", "PERSPECTIVE_TABLE_PIPELINE", "SCREEN_PHOTO_TABLE_PIPELINE"] or
                is_stock_doc or
                detected_doc_type in ["SPREADSHEET", "LEDGER"] or
                doc_profile.get("is_table_heavy", False)
            )
        )
        if is_free_form and not is_id_card and rec_pipeline.endswith("TABLE_PIPELINE"):
            logger.info(
                f"'{original_filename}' is {detected_doc_type}; ignoring the "
                f"{rec_pipeline} recommendation and reading it as prose."
            )

        all_extracted_tables: List[Dict[str, Any]] = []

        if is_digital_doc:
            for page_num, img_path in page_images:
                p_text = digital_results[page_num - 1]["markdown"] if page_num <= len(digital_results) else ""
                full_markdown_parts.append(f"<!-- Page {page_num} -->\n" + p_text)
                full_raw_text_parts.append(p_text)
                per_page_results.append({
                    "page_number": page_num,
                    "image_path": str(img_path) if ext != "pdf" and ext not in ["xlsx", "xls", "csv", "txt", "text", "log"] else "",
                    "markdown": p_text,
                    "processing_time": 0.01
                })
                if existing_doc_id:
                    db.add(DocumentPage(
                        document_id=existing_doc_id,
                        page_number=page_num,
                        page_image_path=str(img_path) if ext != "pdf" and ext not in ["xlsx", "xls", "csv", "txt", "text", "log"] else "",
                        plain_text=p_text,
                        markdown=p_text,
                        processing_time=0.01
                    ))
                    rec = db.query(DocumentRecord).filter(DocumentRecord.id == existing_doc_id).first()
                    if rec:
                        rec.status = f"PROCESSING (Page {page_num} of {page_count})"
                    db.commit()
        else:
            for page_num, img_path in page_images:
                page_start = time.time()

                # Source-tailored preprocessing (perspective homography, screen moire, CLAHE)
                preprocessed_img = preprocessor.process(img_path, profile=doc_profile)

                if is_id_card:
                    # Pass 1: High-Precision Vision AI for ID Cards (100% backward compatible)
                    raw_page_md = model_engine.predict(preprocessed_img, pass1_ocr_prompt, system_prompt=ENGLISH_TRANSLATION_RULE)
                    page_md = sanitize_extracted_markdown(clean_markdown_fence(strip_footer_metadata(html_table_to_markdown(raw_page_md))))
                elif is_table_pipeline:
                    # Coordinate-Aware Table Extraction Engine
                    table_res = table_ocr_service.extract_table(preprocessed_img, profile=doc_profile)
                    if table_res and table_res.get("markdown") and table_res.get("markdown") != "No text has been found.":
                        page_md = table_res["markdown"]
                        if table_res.get("tables"):
                            all_extracted_tables.extend(table_res["tables"])
                    else:
                        # Fallback to Vision AI if coordinate OCR found no text
                        raw_page_md = model_engine.predict(preprocessed_img, pass1_ocr_prompt, system_prompt=ENGLISH_TRANSLATION_RULE)
                        page_md = sanitize_extracted_markdown(clean_markdown_fence(strip_footer_metadata(html_table_to_markdown(raw_page_md))))
                else:
                    # General Vision AI document path
                    raw_page_md = model_engine.predict(preprocessed_img, pass1_ocr_prompt, system_prompt=ENGLISH_TRANSLATION_RULE)
                    page_md = sanitize_extracted_markdown(clean_markdown_fence(strip_footer_metadata(html_table_to_markdown(raw_page_md))))

                page_elapsed = time.time() - page_start
                full_markdown_parts.append(f"<!-- Page {page_num} -->\n" + page_md)
                full_raw_text_parts.append(page_md)

                per_page_results.append({
                    "page_number": page_num,
                    "image_path": str(preprocessed_img),
                    "markdown": page_md,
                    "processing_time": round(page_elapsed, 2)
                })

                if existing_doc_id:
                    db.add(DocumentPage(
                        document_id=existing_doc_id,
                        page_number=page_num,
                        page_image_path=str(preprocessed_img),
                        plain_text=strip_markdown(page_md),
                        markdown=page_md,
                        processing_time=round(page_elapsed, 2)
                    ))
                    rec = db.query(DocumentRecord).filter(DocumentRecord.id == existing_doc_id).first()
                    if rec:
                        rec.status = f"PROCESSING (Page {page_num} of {page_count})"
                    db.commit()

        combined_markdown = normalize_no_text_response("\n\n---\n\n".join(full_markdown_parts))
        combined_raw_text = normalize_no_text_response(strip_markdown("\n\n".join(full_raw_text_parts)))

        # Extract Markdown Tables if not already captured from table OCR
        if not all_extracted_tables:
            raw_tables = extract_markdown_tables(combined_markdown) if combined_markdown != "No text has been found." else []
            for t in raw_tables:
                if isinstance(t, list) and len(t) > 0:
                    cols = t[0] if len(t) > 0 else []
                    rows = t[1:] if len(t) > 1 else []
                    all_extracted_tables.append({
                        "columns": cols,
                        "rows": rows,
                        "markdown": ""
                    })
                elif isinstance(t, dict):
                    all_extracted_tables.append(t)
        else:
            norm_tables = []
            for t in all_extracted_tables:
                if isinstance(t, list):
                    norm_tables.append({
                        "columns": t[0] if len(t) > 0 else [],
                        "rows": t[1:] if len(t) > 1 else [],
                        "markdown": ""
                    })
                elif isinstance(t, dict):
                    norm_tables.append(t)
            all_extracted_tables = norm_tables

        first_page_img = per_page_results[0]["image_path"] if per_page_results else page_images[0][1]

        # Fast-Path Classification Refinement if still UNKNOWN or INVOICE
        #
        # Gated on legibility for the same reason the first classification
        # is: these are keyword rules over OCR text, and OCR text from an
        # unreadable page is noise. Without the gate this block simply undid
        # the earlier fallback - an illegible pamphlet was forced back off
        # UNKNOWN onto FORM by the word "SERVICE", and then asked for form
        # fields it could never have.
        raw_text_upper = (combined_raw_text or "").upper()
        source_is_legible = True
        try:
            if not is_digital_doc and first_page_img:
                import cv2 as _cv2
                _ri = _cv2.imread(str(first_page_img))
                if _ri is not None:
                    _rh, _rw = _ri.shape[:2]
                    source_is_legible = assess_source_legibility(
                        _rw, _rh, combined_raw_text)["legible"]
        except Exception as _re:
            logger.warning(f"Legibility check before classification refinement skipped: {_re}")

        if source_is_legible and detected_doc_type in ["UNKNOWN", "INVOICE", "FORM"]:
            if (
                any(k in raw_text_upper for k in ["STOCK AND SALES", "STOCK & SALES", "STOCK STATEMENT", "STOCK STATMENT", "OP. QTY", "BAL. QTY", "PUR. QTY", "BAL.VAL", "DUMP QTY", "STOCK-VALUE", "SALES-VALUE"]) or
                bool(re.search(r"STOCK\s+STAT[E]?MENT", raw_text_upper)) or
                ("OPENING" in raw_text_upper and "CLOSING" in raw_text_upper and any(k in raw_text_upper for k in ["PURCHASE", "SALES", "STOCK", "RECEIPT", "ISSUE"]))
            ):
                detected_doc_type = "STOCK_STATEMENT"
                cls_confidence = 0.99
            elif "INCOME TAX" in raw_text_upper or "PERMANENT ACCOUNT NUMBER" in raw_text_upper or re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', raw_text_upper):
                detected_doc_type = "PAN"
                cls_confidence = 0.99
            elif ("UIDAI" in raw_text_upper or "UNIQUE IDENTIFICATION" in raw_text_upper or "AADHAAR" in raw_text_upper) and "INCOME TAX" not in raw_text_upper:
                detected_doc_type = "AADHAAR"
                cls_confidence = 0.99
            elif "PASSPORT" in raw_text_upper or ("REPUBLIC OF INDIA" in raw_text_upper and "PASSPORT" in raw_text_upper):
                detected_doc_type = "PASSPORT"
                cls_confidence = 0.99
            elif "DRIVING LICENCE" in raw_text_upper or "DRIVING LICENSE" in raw_text_upper:
                detected_doc_type = "DRIVING_LICENSE"
                cls_confidence = 0.99
            elif ("RESUME" in raw_text_upper or "CURRICULUM VITAE" in raw_text_upper or ("EDUCATION" in raw_text_upper and ("COURSE WORK" in raw_text_upper or "TRAINING" in raw_text_upper or "SKILLS" in raw_text_upper or "EXPERIENCE" in raw_text_upper or "PROJECTS" in raw_text_upper or "DECLARATION" in raw_text_upper or "PROFESSIONAL SUMMARY" in raw_text_upper))):
                detected_doc_type = "RESUME"
                cls_confidence = 0.99
            elif any(k in raw_text_upper for k in ["TAX INVOICE", "BILL OF SUPPLY", "RETAIL INVOICE", "CASH MEMO", "GSTIN"]):
                detected_doc_type = "INVOICE"
                cls_confidence = 0.99
            elif ("EXCEL" in raw_text_upper or "WORKBOOK" in raw_text_upper or "SHEET1" in raw_text_upper or "AUTOSAVE" in raw_text_upper or "CLUSTER" in raw_text_upper or "DATA ANALYSIS" in raw_text_upper or ("COLUMN" in raw_text_upper and "ROW" in raw_text_upper) or (len(all_extracted_tables) > 0 and "EDUCATION" not in raw_text_upper and "DECLARATION" not in raw_text_upper)):
                detected_doc_type = "SPREADSHEET"
                cls_confidence = 0.99
            elif ("MEDICAL CARE" in raw_text_upper or "HOSPITAL" in raw_text_upper or "BROCHURE" in raw_text_upper or "HEALTH SERVICES" in raw_text_upper or "FORM" in raw_text_upper or "APPLICATION" in raw_text_upper):
                detected_doc_type = "FORM"
                cls_confidence = 0.98

        detected_language = "English"
        has_handwriting_overall = doc_profile.get("has_handwriting", False) if doc_profile else False

        val_result: Optional[Dict[str, Any]] = None

        # Pass 2: Structured JSON Extraction matching document category
        if (is_digital_doc and ext in ["xlsx", "xls", "csv"]) or detected_doc_type == "SPREADSHEET":
            table_struct = convert_markdown_table_to_spreadsheet_json(combined_markdown)
            parsed_struct = table_struct or {"sheet_title": original_filename, "columns": [], "rows": []}
        elif is_digital_doc and ext in ["docx", "doc"]:
            parsed_struct = {
                "candidate_name": original_filename.replace(".docx", "").replace(".doc", "").replace("_", " "),
                "contact_info": {},
                "professional_summary": combined_raw_text[:200] if len(combined_raw_text) > 200 else combined_raw_text,
                "skills": {},
                "work_experience": [],
                "education": [],
                "certifications": [],
                "projects": []
            }
        elif detected_doc_type in ["STOCK_STATEMENT", "STOCK_SUMMARY", "STOCK_SALES_REPORT", "SALES_SUMMARY"] and all_extracted_tables:
            # Deterministic Stock Table Reconstruction with Arithmetic Validation
            primary_table = all_extracted_tables[0]
            val_result = validation_service.validate_stock_table(
                columns=primary_table.get("columns", []),
                rows=primary_table.get("rows", []),
                ocr_confidence=doc_profile.get("classification_confidence", 0.95),
                layout_confidence=0.85
            )

            # Task 9: Closed-loop targeted cell Re-OCR for suspicious cells
            if val_result.get("suspicious_cells") and first_page_img:
                repaired = self._attempt_closed_loop_reocr(
                    image_path=Path(first_page_img),
                    primary_table=primary_table,
                    suspicious_cells=val_result["suspicious_cells"]
                )
                if repaired:
                    val_result = validation_service.validate_stock_table(
                        columns=primary_table.get("columns", []),
                        rows=primary_table.get("rows", []),
                        ocr_confidence=doc_profile.get("classification_confidence", 0.95),
                        layout_confidence=0.90
                    )

            dist, comp, per = self._extract_stock_header_metadata(combined_markdown)

            stock_items = []
            for r in val_result.get("validated_rows", []):
                if r.get("is_total_row"):
                    continue
                stock_items.append({
                    "item_code": r.get("item_code"),
                    "product_name": r.get("product_name") or (r.get("raw_cells", [""])[0] if r.get("raw_cells") else ""),
                    "pack": r.get("pack"),
                    "batch": r.get("batch"),
                    "expiry": r.get("expiry"),
                    "rate": r.get("rate"),
                    "opening_qty": r.get("opening_qty"),
                    "receipt_qty": r.get("receipt_qty"),
                    "issue_qty": r.get("issue_qty"),
                    "closing_qty": r.get("closing_qty"),
                    "calculated_closing_qty": r.get("calculated_closing_qty"),
                    "discrepancy": r.get("discrepancy", 0.0),
                    "validation_status": r.get("validation_status", "UNVERIFIED"),
                    # `raw_cells` is deliberately NOT published. It was a
                    # byte-for-byte copy of the matching entry in `rows`, so
                    # every cell of the table went over the wire twice - 23% of
                    # the payload on a 16-column statement, and growing with
                    # table width. Consumers that want the cells read `rows`,
                    # which is positionally aligned with `columns`.
                    #
                    # It remains an internal field on the validated row (see
                    # validation_service and product_name above); this only
                    # stops it being serialised into the API response.
                })

            parsed_struct = {
                "distributor_name": dist,
                "company_name": comp,
                "statement_period": per,
                "summary_totals": (
                    val_result.get("totals_validation", {}).get("printed_totals", {})
                    or self._parse_summary_totals(combined_markdown)
                ),
                "columns": primary_table.get("columns", []),
                "rows": primary_table.get("rows", []),
                "items": stock_items
            }
        else:
            struct_prompt = STRUCTURED_EXTRACTION_PROMPTS.get(
                detected_doc_type, STRUCTURED_EXTRACTION_PROMPTS.get("UNKNOWN")
            )
            if combined_markdown and combined_markdown != "No text has been found.":
                struct_text_prompt = struct_prompt + f"\n\nDOCUMENT EXTRACTED TEXT REFERENCE:\n{combined_markdown[:6000]}\n"
                struct_raw = model_engine.predict_text_only(struct_text_prompt)
            else:
                struct_raw = model_engine.predict(Path(first_page_img), struct_prompt)

            parsed_struct = parse_json_from_llm_response(struct_raw)

        # Fallback / Alignment check for SPREADSHEET & INVOICE: ensure markdown table data is accurately captured row-wise
        spreadsheet_alignment_mismatches: List[int] = []
        if detected_doc_type == "SPREADSHEET":
            table_struct = convert_markdown_table_to_spreadsheet_json(combined_markdown)
            if table_struct and table_struct.get("rows"):
                if not isinstance(parsed_struct, dict):
                    parsed_struct = table_struct
                else:
                    m_rows = parsed_struct.get("rows", [])
                    t_rows = table_struct["rows"]
                    col_cnt = len(table_struct.get("columns", [])) or len(parsed_struct.get("columns", [])) or 1
                    cols = table_struct["columns"] if (not m_rows or len(m_rows) < len(t_rows)) else parsed_struct.get("columns", table_struct["columns"])
                    raw_selected_rows = t_rows if (not m_rows or not isinstance(m_rows, list) or len(m_rows) < len(t_rows)) else m_rows
                    aligned_rows, spreadsheet_alignment_mismatches = auto_align_billing_table_columns(cols, raw_selected_rows)
                    parsed_struct["columns"] = cols
                    parsed_struct["rows"] = sanitize_row_wise_table(aligned_rows, col_cnt)
                    if not parsed_struct.get("sheet_title"):
                        parsed_struct["sheet_title"] = table_struct["sheet_title"]

        elif detected_doc_type == "INVOICE" and combined_markdown:
            table_struct = convert_markdown_table_to_spreadsheet_json(combined_markdown)
            if table_struct and table_struct.get("rows") and isinstance(parsed_struct, dict):
                if not parsed_struct.get("line_items"):
                    # Map columns to roles by header keyword instead of assuming a fixed
                    # description/quantity/unit_price/.../amount position - real invoices
                    # vary in column order (e.g. "Qty, Description, Rate, Disc%, Amount"),
                    # and a fixed index silently writes values under the wrong output key
                    # whenever the actual order differs.
                    role_map = map_invoice_column_roles(table_struct.get("columns", []))
                    line_items = []
                    for idx, r in enumerate(table_struct["rows"]):
                        if any(c.strip() for c in r):
                            def _role_val(role: str) -> str:
                                r_idx = role_map.get(role)
                                return r[r_idx] if (r_idx is not None and r_idx < len(r)) else ""
                            line_items.append({
                                "row_number": idx + 1,
                                "description": _role_val("description"),
                                "quantity": _role_val("quantity"),
                                "unit_price": _role_val("unit_price"),
                                "amount": _role_val("amount")
                            })
                    if line_items:
                        parsed_struct["line_items"] = line_items

        # Build Standardized Enterprise JSON Envelope
        enterprise_envelope = normalize_structured_data(
            doc_type=detected_doc_type,
            data=parsed_struct,
            cls_confidence=cls_confidence
        )

        audit_info = enterprise_envelope.get("_audit", {})

        # If deterministic validation occurred, wire its multi-factor confidence and review flags
        if val_result:
            overall_conf = val_result["overall_confidence"]
            needs_review = val_result["needs_manual_review"]
            enterprise_envelope["overall_confidence"] = overall_conf
            enterprise_envelope["needs_manual_review"] = needs_review
            audit_info["overall_confidence"] = overall_conf
            audit_info["needs_manual_review"] = needs_review
            audit_info["confidence_breakdown"] = {
                "ocr_confidence": val_result.get("ocr_confidence"),
                "layout_confidence": val_result.get("layout_confidence"),
                "row_structure_confidence": val_result.get("row_structure_confidence"),
                "column_structure_confidence": val_result.get("col_structure_confidence"),
                "cell_accuracy_confidence": val_result.get("cell_accuracy_confidence"),
                "numeric_confidence": val_result.get("numeric_confidence"),
                "validation_confidence": val_result.get("validation_confidence")
            }
            audit_info["validation"] = {
                "arithmetic_checks": val_result["arithmetic_checks"],
                "arithmetic_matches": val_result["arithmetic_matches"],
                "arithmetic_mismatches": val_result["arithmetic_mismatches"],
                "totals_status": val_result.get("totals_validation", {}).get("status", "NO_TOTALS")
            }
        else:
            # No table validation ran, so this is an ID card, a visiting card,
            # a resume, a pamphlet. The old code took 0.95 from a default here
            # and called it confidence, which meant a fabricated extraction and
            # a perfect one reported the same number. Check the fields against
            # the page instead - see app/services/field_verification.py.
            verification = None
            try:
                page_w = page_h = None
                if first_page_img:
                    try:
                        import cv2 as _cv2
                        _probe = _cv2.imread(str(first_page_img))
                        if _probe is not None:
                            page_h, page_w = _probe.shape[:2]
                    except Exception:
                        pass

                legibility = assess_source_legibility(page_w, page_h, combined_raw_text)
                verification = verify_fields(
                    parsed_struct,
                    # Both representations, because a value may survive in one
                    # and not the other; grounding normalises whitespace away.
                    ocr_text=" ".join([combined_raw_text or "", combined_markdown or ""]),
                    is_digital_source=is_digital_doc,
                    source_legibility=legibility,
                )
                overall_conf = verification["overall_confidence"]
                needs_review = verification["needs_manual_review"]
                audit_info["field_verification"] = {
                    k: verification[k] for k in
                    ("counts", "field_count", "source_legibility", "flagged")
                }
                enterprise_envelope["overall_confidence"] = overall_conf
                enterprise_envelope["needs_manual_review"] = needs_review
                enterprise_envelope["field_verification"] = audit_info["field_verification"]
                if verification["flagged"]:
                    logger.warning(
                        f"Field verification flagged {len(verification['flagged'])} of "
                        f"{verification['field_count']} field(s) on '{original_filename}': "
                        + "; ".join(f"{f['field']} ({f['reason']})" for f in verification["flagged"][:4])
                    )
            except Exception as e:
                # Verification must never be the reason an extraction fails.
                logger.error(f"Field verification errored, falling back: {e}", exc_info=True)
                overall_conf = audit_info.get("overall_confidence", 0.95)
                needs_review = audit_info.get("needs_manual_review", False)

            # Task 10: Strictly penalize empty tables on table-heavy or spreadsheet documents.
            #
            # Only for document types that are DEFINED by having a table. The
            # is_table_heavy heuristic used to be included here, and it fires
            # on any dense A4 scan - which forced 0.30 onto correctly extracted
            # Aadhaar cards and resumes while a fabricated pamphlet kept 0.95.
            # Where field verification ran, its verdict is the better evidence.
            is_table_expected = detected_doc_type in ["STOCK_STATEMENT", "SPREADSHEET", "LEDGER"]
            if verification is not None and not is_table_expected:
                is_table_expected = False
            extracted_rows_count = len(parsed_struct.get("rows", [])) if isinstance(parsed_struct, dict) else 0
            if is_table_expected and (not all_extracted_tables or extracted_rows_count == 0):
                overall_conf = 0.30
                needs_review = True
                enterprise_envelope["overall_confidence"] = overall_conf
                enterprise_envelope["needs_manual_review"] = needs_review
                audit_info["overall_confidence"] = overall_conf
                audit_info["needs_manual_review"] = needs_review

        # Legibility gate, applied to EVERY path.
        #
        # A 148x148 pamphlet thumbnail has body text a few pixels tall. The
        # model will still return a full, confident-looking structure for it -
        # that is the worst failure mode this system has, because a reviewer
        # is actively misled rather than merely unhelped. If the page was not
        # readable, nothing extracted from it is trustworthy, whichever branch
        # produced it (a tiny pamphlet was even misclassified as a stock
        # statement and scored through the table validator).
        try:
            _probe_img = None
            if first_page_img:
                import cv2 as _cv2
                _probe_img = _cv2.imread(str(first_page_img))
            if _probe_img is not None and not is_digital_doc:
                _h, _w = _probe_img.shape[:2]
                _leg = assess_source_legibility(_w, _h, combined_raw_text)
                if not _leg["legible"]:
                    overall_conf = min(overall_conf, 0.10)
                    needs_review = True
                    enterprise_envelope["overall_confidence"] = overall_conf
                    enterprise_envelope["needs_manual_review"] = True
                    audit_info["overall_confidence"] = overall_conf
                    audit_info["needs_manual_review"] = True
                    audit_info["source_legibility"] = _leg
                    logger.warning(
                        f"Source not legible for '{original_filename}': "
                        + "; ".join(_leg["reasons"])
                    )
        except Exception as e:
            logger.error(f"Legibility gate errored (continuing): {e}")

        # Surface column-count mismatches auto_align_billing_table_columns found (rows whose
        # cell count didn't match the header before padding/truncation) as a review signal,
        # rather than letting them pass through silently padded/truncated with no trace.
        if spreadsheet_alignment_mismatches:
            audit_info["column_alignment_mismatches"] = spreadsheet_alignment_mismatches
            needs_review = True
            enterprise_envelope["needs_manual_review"] = True
            audit_info["needs_manual_review"] = True

        doc_quality = audit_info.get("overall_document_quality", "Good")
        # Real token usage, summed over every model call made for this
        # document. The envelope template's 512/256/768 is a placeholder, and
        # using it meant the analytics ledger was a constant times the
        # document count - it moved only when the document count moved. Fall
        # back to the template figure only if the counter saw no calls at
        # all (a digital PDF or spreadsheet never invokes the model).
        measured = get_token_usage()
        if measured["calls"]:
            p_tokens = measured["prompt_tokens"]
            c_tokens = measured["completion_tokens"]
            t_tokens = measured["total_tokens"]
            logger.info(
                f"Token usage for '{original_filename}': {t_tokens} "
                f"({p_tokens} prompt + {c_tokens} completion over {measured['calls']} call(s))"
            )
        else:
            p_tokens = c_tokens = t_tokens = 0

        # Build clean client JSON output with classification identifier & relevant fields
        client_envelope = {
            "document_type": enterprise_envelope.get("document_type", detected_doc_type),
            "classification_confidence": round(cls_confidence, 2),
            "overall_confidence": overall_conf,
            "needs_manual_review": needs_review,
            # Per-field evidence behind the confidence number, so a consumer
            # can route on WHICH field is suspect rather than one aggregate.
            "field_verification": audit_info.get("field_verification"),
            # Carried here because the top-level `tables` block that used to
            # hold it is no longer published (it repeated fields.rows). This is
            # a real review signal - it means a candidate column was detected
            # but rejected for lack of corroboration, so the table may be
            # MISSING a column rather than having invented one.
            "column_model_uncertain": any(
                t.get("column_model_uncertain") for t in all_extracted_tables
                if isinstance(t, dict)
            ),
            "fields": enterprise_envelope.get("fields", {})
        }

        total_elapsed = round(time.time() - start_time, 2)
        now_utc = datetime.utcnow()

        # Database Archival - Preserving existing ID if reprocessing
        if existing_doc_id is not None:
            doc_record = db.query(DocumentRecord).filter(DocumentRecord.id == existing_doc_id).first()
            if doc_record:
                doc_record.file_size = file_size
                doc_record.page_count = page_count
                doc_record.document_type = detected_doc_type
                doc_record.language = detected_language
                doc_record.has_handwriting = has_handwriting_overall
                doc_record.processing_time = total_elapsed
                doc_record.confidence = overall_conf
                doc_record.overall_confidence = overall_conf
                doc_record.needs_manual_review = needs_review
                doc_record.document_quality = doc_quality
                doc_record.prompt_tokens = p_tokens
                doc_record.completion_tokens = c_tokens
                doc_record.total_tokens = t_tokens
                doc_record.status = "COMPLETED"
                doc_record.completed_at = now_utc
            else:
                doc_record = DocumentRecord(
                    id=existing_doc_id,
                    original_filename=original_filename,
                    stored_filename=file_path.name,
                    file_path=str(file_path),
                    file_type=ext,
                    file_size=file_size,
                    page_count=page_count,
                    document_type=detected_doc_type,
                    language=detected_language,
                    has_handwriting=has_handwriting_overall,
                    processing_time=total_elapsed,
                    confidence=overall_conf,
                    overall_confidence=overall_conf,
                    needs_manual_review=needs_review,
                    document_quality=doc_quality,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    total_tokens=t_tokens,
                    status="COMPLETED",
                    completed_at=now_utc
                )
                db.add(doc_record)
        else:
            stored_fname = file_path.name
            existing_rec = db.query(DocumentRecord).filter(DocumentRecord.stored_filename == stored_fname).first()
            if existing_rec:
                stored_fname = f"{uuid.uuid4().hex[:8]}_{file_path.name}"

            doc_record = DocumentRecord(
                original_filename=original_filename,
                stored_filename=stored_fname,
                file_path=str(file_path),
                file_type=ext,
                file_size=file_size,
                page_count=page_count,
                document_type=detected_doc_type,
                language=detected_language,
                has_handwriting=has_handwriting_overall,
                processing_time=total_elapsed,
                confidence=overall_conf,
                overall_confidence=overall_conf,
                needs_manual_review=needs_review,
                document_quality=doc_quality,
                prompt_tokens=p_tokens,
                completion_tokens=c_tokens,
                total_tokens=t_tokens,
                status="COMPLETED",
                completed_at=now_utc
            )
            db.add(doc_record)

        db.flush()

        # Save Pages only if not already saved during the page loop
        existing_pages = db.query(DocumentPage).filter(DocumentPage.document_id == doc_record.id).count()
        if existing_pages == 0:
            for page_data in per_page_results:
                db_page = DocumentPage(
                    document_id=doc_record.id,
                    page_number=page_data["page_number"],
                    page_image_path=page_data["image_path"],
                    plain_text=page_data["markdown"],
                    markdown=page_data["markdown"],
                    processing_time=page_data["processing_time"]
                )
                db.add(db_page)

        # Save or Update Result (store clean client_envelope and extracted tables)
        # `cells_metadata` is INTERNAL geometry: for every cell it carries the
        # source OCR tokens with pixel bboxes, quad_slope, per-token confidence
        # and glyph dimensions. It drives the closed-loop cell re-OCR above and
        # the benchmark's debug images, and no API consumer has any use for it -
        # yet it was 93% of the table payload (30,309 bytes of 32,370 on a
        # 20-row statement), and it grows with every cell on the page.
        #
        # Stripped here, at the serialization boundary, so everything upstream
        # still has it and neither the response nor the stored row carries it.
        published_tables = [
            {k: v for k, v in t.items() if k != "cells_metadata"}
            for t in all_extracted_tables
        ]

        tables_str = json.dumps(published_tables, ensure_ascii=False) if published_tables else "[]"
        doc_result = db.query(DocumentResult).filter(DocumentResult.document_id == doc_record.id).first()
        if doc_result:
            doc_result.raw_text = combined_raw_text
            doc_result.markdown_text = combined_markdown
            doc_result.structured_json_str = json.dumps(client_envelope, ensure_ascii=False)
            doc_result.tables_json_str = tables_str
        else:
            doc_result = DocumentResult(
                document_id=doc_record.id,
                raw_text=combined_raw_text,
                markdown_text=combined_markdown,
                structured_json_str=json.dumps(client_envelope, ensure_ascii=False),
                tables_json_str=tables_str
            )
            db.add(doc_result)
        db.commit()
        db.refresh(doc_record)

        logger.info(f"Pipeline completed for ID {doc_record.id} in {total_elapsed}s")

        return {
            "id": doc_record.id,
            "filename": original_filename,
            "document_type": detected_doc_type,
            "language": detected_language,
            "has_handwriting": has_handwriting_overall,
            "pages": page_count,
            # `plain_text` is deliberately NOT published. It was the same
            # content as `markdown` with the formatting stripped, so the whole
            # document went over the wire twice. The API contract is markdown
            # plus structured JSON.
            #
            # It is still STORED (DocumentRecord/DocumentPage) because history
            # search queries it and the txt/csv exports render it.
            "markdown": combined_markdown,
            "structured_data": client_envelope,
            "tables": published_tables,
            "processing_time": total_elapsed,
            "status": doc_record.status,
            "created_at": doc_record.created_at.isoformat()
        }

    def _extract_stock_header_metadata(self, markdown_text: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Extracts distributor name, company/depot name, and statement period from pre-table markdown headers."""
        distributor = None
        company = None
        period = None

        raw_lines = (markdown_text or "").splitlines()
        lines = [line.strip().lstrip("#").strip() for line in raw_lines[:20]]
        clean_lines = [l for l in lines if l and not l.startswith("|") and not l.startswith("---")]

        # The company/division is printed as a BANNER ROW inside the grid
        # ("HETERO HEALTHCARE (GENX)"), not in the letterhead. Taking instead
        # "the line after the distributor" is a guess about layout order, and
        # on any statement whose letterhead carries a street address it returns
        # the address as the company name.
        #
        # A banner row is recognisable by shape: exactly one non-empty cell, no
        # figures anywhere in the row.
        banner_company = None
        for raw in raw_lines:
            s = raw.strip()
            if not s.startswith("|") or set(s) <= set("|- "):
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            filled = [c for c in cells if c]
            if len(filled) == 1 and len(filled[0]) > 3 and not re.search(r'\d[\d,.]*$', filled[0]):
                banner_company = filled[0]
                break

        for line in clean_lines:
            l_upper = line.upper()
            if any(k in l_upper for k in ["FROM", "PERIOD", "DATE", "TO ", "MONTH", "FOR THE"]):
                if not period:
                    period = line
            elif not distributor and len(line) > 3 and not any(k in l_upper for k in ["STATEMENT", "REPORT", "ANALYSIS", "PAGE", "STOCK"]):
                distributor = line
            elif distributor and not company and len(line) > 3 and not any(k in l_upper for k in ["STATEMENT", "REPORT", "ANALYSIS", "PAGE", "STOCK"]):
                company = line

        # Evidence beats position: a banner row was actually printed as the
        # division heading, so prefer it over the letterhead line that merely
        # happened to come second.
        if banner_company:
            company = banner_company

        return distributor, company, period

    @staticmethod
    def _parse_summary_totals(markdown_text: str) -> Dict[str, float]:
        """
        Read the printed summary block below the grid into `summary_totals`.

        These documents foot themselves ("OP.Stk Val (PTS) : 138995.86
        Pur.Val(PTS+Tax) : 39761.77 ..."), and those figures were being
        discarded while the field sat empty. Pairs are matched by shape -
        a label, a colon, a number - so no vendor's particular labels are
        assumed, and whatever a given ERP prints comes through under its own
        name.
        """
        totals: Dict[str, float] = {}
        for raw in (markdown_text or "").splitlines():
            s = raw.strip()
            if s.startswith("|"):
                continue
            # A print stamp is label-colon-number shaped too ("Printed By :
            # REKHA - COUNTER1, 01-Jun-26 9:07:57 AM"), and its clock time
            # parses as a total: {"Jun-26 9": 7.0}. It is the one line here
            # that must be skipped outright.
            if re.search(r'\bprint(?:ed)?\s*(?:by|on)\b', s, re.I):
                continue
            s = re.sub(r'\b\d{1,2}:\d{2}(:\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?', ' ', s)
            for label, value in re.findall(r'([A-Za-z][\w.\s()+/&-]{2,40}?)\s*:\s*([\d,]+\.\d{1,2}|[\d,]{2,})', s):
                key = re.sub(r'\s+', ' ', label).strip(" .-")
                try:
                    totals[key] = float(value.replace(",", ""))
                except ValueError:
                    continue
        return totals

    def _attempt_closed_loop_reocr(
        self,
        image_path: Path,
        primary_table: Dict[str, Any],
        suspicious_cells: List[Dict[str, Any]]
    ) -> bool:
        """
        Executes targeted cell crop and multi-variant second-pass OCR for arithmetic discrepancies.
        Repairs cell ONLY when re-OCR provides visual evidence matching expected arithmetic value.
        NEVER silently fabricates or overwrites values without visual verification.
        """
        import cv2
        import numpy as np

        image_path = Path(image_path)
        if not image_path.exists() or not rapid_ocr_service.engine:
            return False

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            img_h, img_w = img.shape[:2]

            repaired_any = False
            cells_meta = primary_table.get("cells_metadata", [])
            table_rows = primary_table.get("rows", [])

            for sc in suspicious_cells:
                r_idx = sc.get("row_index")
                c_idx = sc.get("column_index")
                expected_val = sc.get("expected_val")
                if r_idx is None or c_idx is None or expected_val is None:
                    continue
                if r_idx >= len(cells_meta) or c_idx >= len(cells_meta[r_idx]):
                    continue

                cell_meta = cells_meta[r_idx][c_idx]
                tokens = cell_meta.get("tokens", [])
                if not tokens:
                    continue

                x0 = min(t.get("x0", 0.0) for t in tokens)
                y0 = min(t.get("y0", 0.0) for t in tokens)
                x1 = max(t.get("x1", 0.0) for t in tokens)
                y1 = max(t.get("y1", 0.0) for t in tokens)

                pad_x = max(10, int(0.12 * (x1 - x0)))
                pad_y = max(8, int(0.18 * (y1 - y0)))
                crop_x0 = max(0, int(x0 - pad_x))
                crop_y0 = max(0, int(y0 - pad_y))
                crop_x1 = min(img_w, int(x1 + pad_x))
                crop_y1 = min(img_h, int(y1 + pad_y))

                crop = img[crop_y0:crop_y1, crop_x0:crop_x1]
                if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
                    continue

                if crop.shape[0] < 45:
                    scale = 2.0
                    crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_CUBIC)

                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(gray)
                var_a = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                var_b = cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)

                blurred = cv2.GaussianBlur(crop, (0, 0), 2.0)
                var_c = cv2.addWeighted(crop, 1.6, blurred, -0.6, 0)

                verified = False
                expected_str = str(expected_val).rstrip(".0") if str(expected_val).endswith(".0") else str(expected_val)

                for var_img in [var_a, var_b, var_c]:
                    var_res, _ = rapid_ocr_service.engine(var_img)
                    if not var_res:
                        continue
                    combined_text = " ".join(item[1] for item in var_res).replace(",", "").strip()
                    nums = re.findall(r'\d+(?:\.\d+)?', combined_text)
                    for n in nums:
                        try:
                            n_val = float(n)
                            if abs(n_val - float(expected_val)) <= 0.05:
                                verified = True
                                break
                        except ValueError:
                            pass
                    if verified:
                        break

                if verified:
                    table_rows[r_idx][c_idx] = expected_str
                    cell_meta["raw"] = expected_str
                    cell_meta["repaired"] = True
                    cell_meta["repaired_from"] = sc.get("current_val")
                    repaired_any = True
                    logger.info(
                        f"Closed-loop targeted Re-OCR verified and repaired cell (Row {r_idx+1}, Col {c_idx+1}): "
                        f"'{sc.get('current_val')}' -> '{expected_str}'"
                    )

            return repaired_any
        except Exception as e:
            logger.warning(f"Closed-loop Re-OCR failed: {e}")
            return False


pipeline = OCRPipeline()
