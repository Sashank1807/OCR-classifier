"""
Document Analyzer Service
Deterministic and visual profiling of incoming documents before OCR.
Detects source medium, document type, table heaviness, perspective distortion,
screen moire, specular glare, and recommends the optimal extraction pipeline.
"""

import re
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from app.core.logger import logger


def normalize_erp_header_string(text: str) -> str:
    """Removes all non-alphanumeric characters and converts to lowercase."""
    return re.sub(r'[^a-z0-9]', '', (text or "").lower())


class DocumentAnalyzer:
    """
    Analyzes document characteristics and recommends the optimal processing pipeline.
    Preserves specialized handling for ID cards while directing tabular documents
    to coordinate-aware extraction.
    """

    # Identity documents: fixed, well-known field sets on a small card.
    ID_CARD_TYPES = frozenset({"PAN", "AADHAAR", "PASSPORT", "DRIVING_LICENSE"})

    # Document types whose content is PROSE laid out in reading order, not a
    # grid - whatever the line-detection heuristics say about them. These must
    # never be routed to row-clustering reconstruction; see the note at the
    # recommendation step. This is the production document set for this
    # deployment, so a misroute here is a misroute of everything that matters.
    FREE_FORM_TYPES = ID_CARD_TYPES | frozenset({
        "RESUME", "CV", "CURRICULUM_VITAE",
        "VISITING_CARD",
        "PRESCRIPTION",
        "FORM", "REGISTRATION_FORM", "HOSPITAL_DOCUMENTS",
        "UNKNOWN",          # pamphlets, flyers, posters land here
    })

    # Keyword patterns for deterministic document classification
    STOCK_KEYWORDS = [
        "STOCK AND SALES", "STOCK & SALES", "STOCK STATEMENT", "STOCK STATMENT", "SALES STATEMENT",
        "STOCK REPORT", "STOCK SUMMARY", "SALES SUMMARY", "STOCK AND SALE", "STOCK-VALUE", "SALES-VALUE",
        "STOCK VALUE", "SALES VALUE", "OP. QTY", "OP.QTY", "OPENING QTY", "OPENING BAL", "OPENING STOCK",
        "SALE QTY", "CLOSING QTY", "CLOSING BAL", "CLOSING STOCK", "BAL. QTY",
        "BAL.VAL", "BAL VAL", "DUMP QTY", "PUR. QTY", "PURCHASE QTY", "RECEIPT QTY",
        "EXPIRY", "N.EXP", "BATCH NO", "BATCH NUMBER"
    ]

    COMPACT_STOCK_KEYWORDS = [
        "stocksummary", "stocksalesanalysis", "stockstatement", "stockstatment",
        "stockandsales", "stocksales", "stockclosing", "stockposition", "itemwisestock",
        "dailystock", "stockreport", "salesanalysis", "stockanalysis", "closingstock",
        "openingstock", "salesstatement", "stockledger", "itemwisestocksummary"
    ]

    INVOICE_KEYWORDS = [
        "TAX INVOICE", "INVOICE", "BILL OF SUPPLY", "RETAIL INVOICE", "CASH MEMO",
        "GSTIN", "SUBTOTAL", "GRAND TOTAL", "INVOICE NO", "INVOICE DATE", "BILL NO"
    ]

    LEDGER_KEYWORDS = [
        "LEDGER", "ACCOUNT STATEMENT", "STATEMENT OF ACCOUNT", "DEBIT", "CREDIT",
        "BALANCE B/F", "BALANCE C/F", "VOUCHER NO", "DR AMOUNT", "CR AMOUNT"
    ]

    PAN_KEYWORDS = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "P.A.N", "GOVT. OF INDIA"]
    AADHAAR_KEYWORDS = ["UNIQUE IDENTIFICATION AUTHORITY OF INDIA", "UIDAI", "AADHAAR", "GOVERNMENT OF INDIA", "HELP@UIDAI.GOV.IN"]
    PASSPORT_KEYWORDS = ["REPUBLIC OF INDIA", "PASSPORT", "PASSPORT NO", "GIVEN NAME"]
    DRIVING_KEYWORDS = ["DRIVING LICENCE", "DRIVING LICENSE", "UNION OF INDIA", "DRIVING AUTHORISATION"]

    def analyze(
        self,
        file_path: Path,
        image_path: Optional[Path] = None,
        requested_type: str = "AUTO"
    ) -> Dict[str, Any]:
        """
        Executes comprehensive document analysis and returns an execution profile.
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower().lstrip(".")
        img_target = Path(image_path) if image_path and Path(image_path).exists() else file_path

        profile: Dict[str, Any] = {
            "source_type": "UNKNOWN",
            "document_type": "UNKNOWN",
            "is_table_heavy": False,
            "has_perspective_distortion": False,
            "has_screen_moire": False,
            "has_glare": False,
            "low_contrast": False,
            "perspective_quad": None,
            "estimated_columns": 0,
            "estimated_rows": 0,
            "recommended_pipeline": "GENERAL_VLM_PIPELINE",
            "classification_confidence": 0.90
        }

        # 1. Office Documents (Excel, Word, Text)
        if ext in ["xlsx", "xls", "csv"]:
            profile["source_type"] = "OFFICE_DOC"
            profile["document_type"] = "SPREADSHEET"
            profile["is_table_heavy"] = True
            profile["recommended_pipeline"] = "OFFICE_DOC_PIPELINE"
            profile["classification_confidence"] = 1.0
            return profile

        if ext in ["docx", "doc"]:
            profile["source_type"] = "OFFICE_DOC"
            profile["document_type"] = "RESUME"
            profile["recommended_pipeline"] = "OFFICE_DOC_PIPELINE"
            profile["classification_confidence"] = 1.0
            return profile

        if ext in ["txt", "text", "log"]:
            profile["source_type"] = "OFFICE_DOC"
            profile["document_type"] = "SPREADSHEET"
            profile["is_table_heavy"] = True
            profile["recommended_pipeline"] = "OFFICE_DOC_PIPELINE"
            profile["classification_confidence"] = 1.0
            return profile

        # 2. PDF Analysis
        if ext == "pdf":
            pdf_profile = self._analyze_pdf(file_path)
            profile.update(pdf_profile)
            if profile["recommended_pipeline"] == "NATIVE_PDF_TABLE_PIPELINE":
                return profile

        # 3. Image Analysis (For images or rendered PDF pages)
        is_image_ext = ext in ["jpg", "jpeg", "png", "webp", "bmp", "tiff", "jfif", "heic"]
        if (is_image_ext or (image_path and Path(image_path).exists())):
            img_to_check = Path(image_path) if image_path and Path(image_path).exists() else file_path
            img_profile = self._analyze_image(img_to_check)
            profile.update(img_profile)

        # 4. Check Explicit Request
        req_clean = (requested_type or "AUTO").strip().upper()
        if req_clean != "AUTO":
            profile["document_type"] = req_clean
            profile["classification_confidence"] = 1.0

        # 5. Filename-based fast identification
        fname_lower = file_path.name.lower()
        fname_compact = normalize_erp_header_string(file_path.name)
        if profile["document_type"] == "UNKNOWN":
            if "pan" in fname_lower:
                profile["document_type"] = "PAN"
                profile["classification_confidence"] = 0.99
            elif "aadhaar" in fname_lower or "aadhar" in fname_lower:
                profile["document_type"] = "AADHAAR"
                profile["classification_confidence"] = 0.99
            elif "passport" in fname_lower:
                profile["document_type"] = "PASSPORT"
                profile["classification_confidence"] = 0.99
            elif "driving" in fname_lower or "license" in fname_lower or "licence" in fname_lower:
                profile["document_type"] = "DRIVING_LICENSE"
                profile["classification_confidence"] = 0.99
            elif "resume" in fname_lower or "cv" in fname_lower:
                profile["document_type"] = "RESUME"
                profile["classification_confidence"] = 0.99
            elif any(k in fname_lower for k in ["stock", "sales", "sst", "inventory", "st-", "sas", "lifecare", "medica"]) or any(k in fname_compact for k in self.COMPACT_STOCK_KEYWORDS):
                profile["document_type"] = "STOCK_STATEMENT"
                profile["classification_confidence"] = 0.95
            elif any(k in fname_lower for k in ["invoice", "bill", "receipt", "voucher"]):
                profile["document_type"] = "INVOICE"
                profile["classification_confidence"] = 0.95

        # 6. Final Pipeline Recommendation
        doc_type = profile["document_type"]

        # The coordinate table pipeline rebuilds text by clustering tokens into
        # rows ACROSS THE FULL PAGE WIDTH. On a real table that is exactly
        # right. On a two-column prose document it is exactly wrong: it welds
        # the left column's line to the right column's line.
        #
        # Measured on a real resume, which `is_table_heavy` claimed (with
        # estimated_rows=0 and estimated_columns=0 - the heuristic fires on
        # section rules and header bars, not on an actual grid). Its EDUCATION
        # block came back as
        #
        #     "Bachelor of Pharmacy Higher Secondary (WBCHSE)"
        #     "(B.Pharm) JNTUK - QIS - 2021|Result- 71%"
        #
        # and the school's 71% then became the degree's GPA. Every word was
        # read correctly; the READING ORDER destroyed the meaning, and field
        # verification cannot catch that because both values are on the page.
        #
        # So: document types that are prose by definition never take the
        # table path on the strength of a heuristic. A genuine table inside
        # one of them is still transcribed - the VLM emits markdown tables -
        # it just is not reconstructed by row clustering.
        if doc_type in self.FREE_FORM_TYPES:
            profile["recommended_pipeline"] = (
                "ID_CARD_PIPELINE" if doc_type in self.ID_CARD_TYPES else "GENERAL_VLM_PIPELINE"
            )
            if profile.get("is_table_heavy"):
                logger.info(
                    f"'{file_path.name}' looked table-heavy, but {doc_type} is a free-form "
                    f"document type - using {profile['recommended_pipeline']} so its columns "
                    f"are read in order rather than clustered into rows."
                )
                profile["is_table_heavy"] = False
        elif profile.get("source_type") == "DIGITAL_PDF" and profile.get("is_table_heavy"):
            profile["recommended_pipeline"] = "NATIVE_PDF_TABLE_PIPELINE"
        elif profile.get("has_perspective_distortion") and profile.get("is_table_heavy"):
            profile["recommended_pipeline"] = "PERSPECTIVE_TABLE_PIPELINE"
        elif profile.get("source_type") == "SCREEN_PHOTO" and profile.get("is_table_heavy"):
            profile["recommended_pipeline"] = "SCREEN_PHOTO_TABLE_PIPELINE"
        elif profile.get("is_table_heavy") or doc_type in ["STOCK_STATEMENT", "STOCK_SUMMARY", "STOCK_SALES_REPORT", "SALES_SUMMARY", "LEDGER", "SPREADSHEET"]:
            profile["recommended_pipeline"] = "COORDINATE_TABLE_PIPELINE"
        else:
            profile["recommended_pipeline"] = "GENERAL_VLM_PIPELINE"

        logger.info(
            f"DocumentAnalyzer: '{file_path.name}' -> Source={profile['source_type']}, "
            f"Type={profile['document_type']}, TableHeavy={profile['is_table_heavy']}, "
            f"Distortion={profile['has_perspective_distortion']}, Recommended={profile['recommended_pipeline']}"
        )
        return profile

    def _analyze_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Examines PDF text layer, vector commands, and font integrity."""
        res: Dict[str, Any] = {
            "source_type": "SCANNED_PDF",
            "is_table_heavy": False,
            "recommended_pipeline": "GENERAL_VLM_PIPELINE"
        }
        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            if len(doc) == 0:
                doc.close()
                return res

            page = doc.load_page(0)
            text = page.get_text("text") or ""
            words = page.get_text("words") or []
            rect = page.rect
            page_area = rect.width * rect.height

            # Check text area coverage
            total_text_area = 0.0
            for w in words:
                x0, y0, x1, y1 = w[:4]
                total_text_area += (x1 - x0) * (y1 - y0)
            area_coverage = (total_text_area / page_area) if page_area > 0 else 0.0

            # Alphanumeric, whitespace & table punctuation ratio (decimals, dashes, slashes, pipes)
            char_count = len(text.strip())
            valid_chars = sum(1 for c in text if (c.isalnum() or c.isspace() or c in ".-/|,:%$#@()[]_+=*"))
            text_valid_ratio = (valid_chars / char_count) if char_count > 0 else 0.0

            # Line drawings / table grids
            drawings = page.get_drawings()
            has_table_drawings = len(drawings) > 5

            text_upper = text.upper()
            compact_pdf = normalize_erp_header_string(text[:3000])
            has_stock_kws = (
                any(kw in text_upper for kw in self.STOCK_KEYWORDS) or
                any(k in compact_pdf for k in self.COMPACT_STOCK_KEYWORDS) or
                bool(re.search(r"STOCK\s+STAT[E]?MENT", text_upper)) or
                (("opening" in compact_pdf or "opbal" in compact_pdf) and
                 ("closing" in compact_pdf or "clbal" in compact_pdf) and
                 any(k in compact_pdf for k in ["purchase", "sales", "stock", "receipt", "issue", "qty", "val", "balance"]))
            )

            is_digital_reliable = (
                len(words) >= 20 and
                text_valid_ratio >= 0.75 and
                area_coverage >= 0.01 and
                not ("\ufffd" in text or "\\x00" in text)
            )

            if is_digital_reliable:
                res["source_type"] = "DIGITAL_PDF"
                res["is_table_heavy"] = has_table_drawings or len(words) > 80 or has_stock_kws
                if res["is_table_heavy"]:
                    res["recommended_pipeline"] = "NATIVE_PDF_TABLE_PIPELINE"
                if has_stock_kws:
                    res["document_type"] = "STOCK_STATEMENT"
            else:
                res["source_type"] = "SCANNED_PDF"

            doc.close()
        except Exception as e:
            logger.warning(f"PDF analysis failed for {pdf_path}: {e}")

        return res

    def _analyze_image(self, image_path: Path) -> Dict[str, Any]:
        """Examines visual cues: lines, perspective quadrilaterals, screen moire, glare."""
        res: Dict[str, Any] = {
            "source_type": "CLEAN_SCAN",
            "is_table_heavy": False,
            "has_perspective_distortion": False,
            "has_screen_moire": False,
            "has_glare": False,
            "low_contrast": False,
            "perspective_quad": None,
            "estimated_columns": 0,
            "estimated_rows": 0
        }

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return res

            h, w = img.shape[:2]
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Contrast Check
            std_dev = float(np.std(gray))
            if std_dev < 38.0:
                res["low_contrast"] = True

            # Glare / Specular Highlight Check
            glare_mask = gray > 242
            glare_pct = float(np.mean(glare_mask) * 100.0)
            if 0.8 < glare_pct < 18.0:
                res["has_glare"] = True

            # Screen Moire / Monitor Pattern Check (using FFT high-frequency peaks)
            res["has_screen_moire"] = self._detect_screen_moire(gray)
            if res["has_screen_moire"]:
                res["source_type"] = "SCREEN_PHOTO"

            # Perspective Distortion Check (Find dominant document quad)
            quad = self._detect_perspective_quad(gray)
            if quad is not None:
                res["has_perspective_distortion"] = True
                res["perspective_quad"] = quad

            # Table Grid Line Detection (Horizontal & Vertical kernels)
            h_lines, v_lines = self._detect_grid_lines(gray)
            num_h = len(h_lines)
            num_v = len(v_lines)
            res["estimated_rows"] = max(0, num_h - 1)
            res["estimated_columns"] = max(0, num_v - 1)

            if num_h >= 3 and num_v >= 3:
                res["is_table_heavy"] = True

            # RapidOCR fast keyword check for document type classification
            try:
                from app.services.rapid_ocr_service import rapid_ocr_service
                if rapid_ocr_service.engine:
                    rapid_res = rapid_ocr_service.extract_image_ocr(image_path)
                    top_text = (rapid_res.get("markdown", "")[:1500]).upper()
                    compact_top = normalize_erp_header_string(top_text)
                    has_compact_stock = (
                        any(k in compact_top for k in self.COMPACT_STOCK_KEYWORDS) or
                        (("opening" in compact_top or "opbal" in compact_top or "opqty" in compact_top) and
                         ("closing" in compact_top or "clbal" in compact_top or "clqty" in compact_top) and
                         any(x in compact_top for x in ["receipt", "issue", "sale", "purchase", "balance", "qty", "val", "value", "batch", "total"]))
                    )

                    # Identity Check
                    if any(k in top_text for k in self.PAN_KEYWORDS) or re.search(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', top_text):
                        res["document_type"] = "PAN"
                    elif any(k in top_text for k in self.AADHAAR_KEYWORDS) or re.search(r'\b[0-9]{4}\s+[0-9]{4}\s+[0-9]{4}\b', top_text):
                        res["document_type"] = "AADHAAR"
                    elif any(k in top_text for k in self.PASSPORT_KEYWORDS):
                        res["document_type"] = "PASSPORT"
                    elif any(k in top_text for k in self.DRIVING_KEYWORDS):
                        res["document_type"] = "DRIVING_LICENSE"
                    # Stock vs Invoice vs Ledger (Stock check takes priority if compact ERP stock tokens exist)
                    elif has_compact_stock or any(k in top_text for k in self.STOCK_KEYWORDS):
                        res["document_type"] = "STOCK_STATEMENT"
                        res["classification_confidence"] = 0.99
                        res["is_table_heavy"] = True
                    elif any(k in top_text for k in self.LEDGER_KEYWORDS):
                        res["document_type"] = "LEDGER"
                        res["is_table_heavy"] = True
                    elif any(k in top_text for k in self.INVOICE_KEYWORDS):
                        res["document_type"] = "INVOICE"
                        if num_h >= 2 or num_v >= 2 or "QTY" in top_text:
                            res["is_table_heavy"] = True

                    if rapid_res.get("lines_count", 0) > 35 or has_compact_stock:
                        res["is_table_heavy"] = True
            except Exception as re_err:
                logger.debug(f"RapidOCR fast classification skipped: {re_err}")

            # Dot-matrix printer detection (ribbon-ink tint), gated to table-heavy/stock
            # documents only and never overriding a screen-photo classification - this
            # must never fire for PAN/Aadhaar/Passport/general documents, where a
            # dot-matrix-tailored preprocessing branch would be inappropriate.
            # `.get`, not `[...]`: this branch runs before the classification
            # step has necessarily written `document_type`, and on a .webp the
            # missing key was raising KeyError inside the try, which reported
            # itself only as "Image analysis error" and silently discarded the
            # rest of the analysis for that page.
            if not res.get("has_screen_moire") and (
                res.get("is_table_heavy") or res.get("document_type") in ["STOCK_STATEMENT", "LEDGER", "INVOICE"]
            ):
                if self._detect_dot_matrix_ribbon(img):
                    res["source_type"] = "DOT_MATRIX"

        except Exception as e:
            logger.warning(f"Image analysis error on {image_path}: {e}")

        return res

    def _detect_dot_matrix_ribbon(self, img: np.ndarray) -> bool:
        """
        Detects blue/purple dot-matrix ribbon ink tint: dot-matrix ribbon ink typically
        carries elevated blue relative to red compared to black toner/laser/inkjet text.
        Purely a statistical color-channel signal, no vocabulary or filename involved.
        """
        try:
            if img is None or len(img.shape) != 3:
                return False
            b, g, r = cv2.split(img)
            ribbon_diff = float(np.mean(b.astype(np.int16) - r.astype(np.int16)))
            return ribbon_diff > 8.0
        except Exception:
            return False

    def _detect_screen_moire(self, gray: np.ndarray) -> bool:
        """Detects periodic high-frequency interference characteristic of screen/monitor photos."""
        try:
            h, w = gray.shape[:2]
            # Downsample for fast FFT
            small = cv2.resize(gray, (min(w, 512), min(h, 512)))
            f = np.fft.fft2(small)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)

            cy, cx = magnitude_spectrum.shape[0] // 2, magnitude_spectrum.shape[1] // 2
            # Zero out DC center
            magnitude_spectrum[cy-15:cy+15, cx-15:cx+15] = 0

            # Check for sharp periodic peaks in outer ring
            mean_val = np.mean(magnitude_spectrum)
            max_val = np.max(magnitude_spectrum)
            peak_ratio = (max_val / mean_val) if mean_val > 0 else 0.0

            return peak_ratio > 3.4
        except Exception:
            return False

    def _detect_perspective_quad(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects quadrilateral document boundaries with perspective tilt.
        Returns ordered 4-point numpy array or None if document is already frontal.
        """
        try:
            h, w = gray.shape[:2]
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 40, 120)

            # Dilate to connect broken edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            dilated = cv2.dilate(edges, kernel, iterations=2)

            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None

            # Sort contours by area descending
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
            img_area = float(w * h)

            for c in contours:
                area = cv2.contourArea(c)
                # Quad must occupy at least 35% of frame
                if area < 0.35 * img_area:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.025 * peri, True)

                if len(approx) == 4:
                    pts = approx.reshape(4, 2)
                    ordered = self._order_points(pts)

                    # Compute side lengths to verify trapezoidal distortion
                    top_w = np.linalg.norm(ordered[0] - ordered[1])
                    bot_w = np.linalg.norm(ordered[3] - ordered[2])
                    left_h = np.linalg.norm(ordered[0] - ordered[3])
                    right_h = np.linalg.norm(ordered[1] - ordered[2])

                    w_ratio = abs(top_w - bot_w) / max(top_w, bot_w, 1.0)
                    h_ratio = abs(left_h - right_h) / max(left_h, right_h, 1.0)

                    # If opposing sides differ by > 4%, genuine perspective distortion exists
                    if (w_ratio > 0.04 or h_ratio > 0.04) and (area < 0.96 * img_area):
                        return ordered

            return None
        except Exception:
            return None

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Orders points: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left has smallest x + y
        rect[2] = pts[np.argmax(s)]  # bottom-right has largest x + y

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right has smallest x - y
        rect[3] = pts[np.argmax(diff)]  # bottom-left has largest x - y

        return rect

    def _detect_grid_lines(self, gray: np.ndarray) -> Tuple[List[int], List[int]]:
        """Detects horizontal and vertical line positions to estimate table structure."""
        try:
            h, w = gray.shape[:2]
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Horizontal lines kernel
            h_size = max(15, w // 30)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_size, 1))
            h_morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

            # Vertical lines kernel
            v_size = max(15, h // 30)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
            v_morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

            # Find horizontal coordinates
            h_proj = np.sum(h_morphed, axis=1)
            h_lines = np.where(h_proj > (0.15 * w * 255))[0]
            clustered_h = self._cluster_coordinates(h_lines, tolerance=8)

            # Find vertical coordinates
            v_proj = np.sum(v_morphed, axis=0)
            v_lines = np.where(v_proj > (0.15 * h * 255))[0]
            clustered_v = self._cluster_coordinates(v_lines, tolerance=8)

            return clustered_h, clustered_v
        except Exception:
            return [], []

    def _cluster_coordinates(self, coords: np.ndarray, tolerance: int = 8) -> List[int]:
        """Clusters nearby coordinate lines within pixel tolerance into a single coordinate."""
        if len(coords) == 0:
            return []
        clusters = []
        curr = [coords[0]]
        for c in coords[1:]:
            if c - curr[-1] <= tolerance:
                curr.append(c)
            else:
                clusters.append(int(np.mean(curr)))
                curr = [c]
        if curr:
            clusters.append(int(np.mean(curr)))
        return clusters


document_analyzer = DocumentAnalyzer()
