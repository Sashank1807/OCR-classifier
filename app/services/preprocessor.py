# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from app.core.config import settings
from app.core.logger import logger


class ImagePreprocessor:
    """OpenCV-powered document image preprocessing engine."""

    def process(self, image_path: Path, profile: Optional[Dict[str, Any]] = None) -> Path:
        """
        Applies source-tailored preprocessing:
        - Screen photo: glare suppression, moire reduction, text edge enhancement
        - Table/Stock statement: perspective correction, illumination flattening, decimal preservation
        - Identity card (PAN/Aadhaar): existing proven high-accuracy pipeline
        - General document: standard deskew, CLAHE, and denoising
        """
        image_path = Path(image_path)
        if not settings.ENABLE_PREPROCESSING:
            return image_path

        try:
            image = cv2.imread(str(image_path))
            if image is None:
                try:
                    pil_img = Image.open(str(image_path)).convert("RGB")
                    image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
            if image is None:
                logger.warning(f"Could not load image at {image_path} with OpenCV or PIL. Skipping preprocessing.")
                return image_path

            processed = image.copy()
            source_type = profile.get("source_type", "") if profile else ""
            doc_type = profile.get("document_type", "") if profile else ""
            has_perspective = profile.get("has_perspective_distortion", False) if profile else False
            quad = profile.get("perspective_quad", None) if profile else None

            # 0a. Page orientation (90 / 180 / 270), before anything else.
            #
            # Deskew below only corrects a few degrees of tilt. Nothing
            # detected a page photographed upside down or sideways, which is
            # why the viewer grew manual "Rotate 180" and "Mirror Unflip"
            # buttons - the user was correcting it by hand, after seeing a
            # bad result.
            #
            # It is not merely an accuracy loss. Measured on a real PAN card
            # rotated 180 degrees, the extractor returned the FATHER's name
            # as `name` and the holder's name as `father_name` - swapped,
            # at confidence 1.0 with no review flag. Every value was present
            # on the page, so grounding passed it; only the layout had
            # changed, and the model assigned the two name fields by
            # position. A confidently wrong identity attribution is the worst
            # output this system can produce, so the page is turned upright
            # before the model ever sees it.
            processed = self.correct_orientation(processed)

            # 0b. Perspective Rectification if trapezoid distortion is detected
            if has_perspective and quad is not None:
                processed = self.warp_perspective(processed, quad)

            # 1. Resolution Normalization
            #
            # A continuous target-dimension scale (tried during development) helps the
            # under-served ~700-900px band, but empirical benchmarking against the real
            # core documents showed no single target is safe: this pipeline's column/row
            # geometry uses absolute-pixel thresholds throughout table_ocr_service.py
            # (numeric-cluster tolerance, column margins, etc.), and different documents
            # need very different amounts of upscaling just to detect their header/numeric
            # text at all (some needed none, one needed aggressive scaling just to detect
            # its header row, and scaling amount also visibly affects column-boundary
            # bleed between adjacent narrow columns). Getting this right needs a
            # per-document-adaptive approach (e.g. re-attempting at a higher scale when
            # initial detection is sparse), not a single global target - that is future
            # work, not something to ship without further validation. Reverted to the
            # original, previously-validated fixed rule for now.
            h, w = processed.shape[:2]
            max_dim = max(w, h)
            min_dim = min(w, h)
            if max_dim > 2560:
                scale = 2560.0 / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_AREA)
            elif min_dim < 650:
                scale = 2.0
                new_w, new_h = int(w * scale), int(h * scale)
                processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # 2. Source-Tailored Processing Branch
            if source_type == "SCREEN_PHOTO" or (profile and profile.get("has_screen_moire")):
                processed = self.process_screen_photo(processed)
            elif doc_type in ["PAN", "AADHAAR", "PASSPORT", "DRIVING_LICENSE"]:
                processed = self.process_id_card(processed)
            elif doc_type in ["STOCK_STATEMENT", "STOCK_SUMMARY", "STOCK_SALES_REPORT", "SALES_SUMMARY", "INVOICE", "SPREADSHEET", "LEDGER"] or (profile and profile.get("is_table_heavy")):
                processed = self.process_table(processed, source_type=source_type)
            else:
                processed = self.process_general(processed)

            output_path = settings.OUTPUT_DIR / f"preprocessed_{image_path.stem}.png"
            cv2.imwrite(str(output_path), processed)
            logger.info(f"Successfully preprocessed image ({source_type or 'DEFAULT'}): {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error during image preprocessing: {str(e)}", exc_info=True)
            return image_path

    def warp_perspective(self, image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """Applies 4-point perspective transform to rectify trapezoidal angle distortion with strict confidence gating."""
        try:
            # Check quadrilateral convexity
            quad_int = quad.astype(np.int32)
            if not cv2.isContourConvex(quad_int):
                logger.warning("Perspective quad rejected: contour is concave.")
                return image

            tl, tr, br, bl = quad
            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_width = max(int(width_a), int(width_b))

            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_height = max(int(height_a), int(height_b))

            if max_width < 120 or max_height < 120:
                return image

            aspect_ratio = max_width / max(1.0, float(max_height))
            if aspect_ratio < 0.25 or aspect_ratio > 4.0:
                logger.warning(f"Perspective quad rejected: unnatural aspect ratio {aspect_ratio:.2f}")
                return image

            # Save debug quad image
            try:
                debug_quad_img = image.copy()
                pts = quad_int.reshape((-1, 1, 2))
                cv2.polylines(debug_quad_img, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                debug_path = settings.OUTPUT_DIR / "debug_perspective_quad.png"
                cv2.imwrite(str(debug_path), debug_quad_img)
            except Exception:
                pass

            dst = np.array([
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1]
            ], dtype="float32")

            M = cv2.getPerspectiveTransform(quad.astype("float32"), dst)
            warped = cv2.warpPerspective(image, M, (max_width, max_height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            logger.info(f"Perspective transform rectified document to {max_width}x{max_height} frontal rectangle.")
            return warped
        except Exception as e:
            logger.warning(f"Perspective warp failed: {e}")
            return image

    def process_screen_photo(self, image: np.ndarray) -> np.ndarray:
        """Screen photograph optimization: glare reduction, moire suppression, contrast."""
        res = self.suppress_screen_glare(image)
        if settings.DESKEW_ENABLED:
            res = self.deskew(res)
        # Suppress LCD subpixel grid noise and CRT scanlines using mild morphological opening & bilateral filter
        res = cv2.bilateralFilter(res, d=5, sigmaColor=30, sigmaSpace=30)
        if settings.CLAHE_ENABLED:
            res = self.enhance_color_contrast(res)
        blurred = cv2.GaussianBlur(res, (0, 0), 1.5)
        res = cv2.addWeighted(res, 1.25, blurred, -0.25, 0)
        try:
            debug_moire_path = settings.OUTPUT_DIR / "debug_screen_moire.png"
            cv2.imwrite(str(debug_moire_path), res)
        except Exception:
            pass
        return res

    def process_table(self, image: np.ndarray, source_type: Optional[str] = None) -> np.ndarray:
        """Table & stock statement optimization: preserves thin lines, minus signs, and decimal dots."""
        res = image
        # Illumination flattening is tailored for photos/crumpled mobile captures; bypass for
        # clean flatbed scans and dot-matrix printouts to protect thin dots, pin strokes, and
        # faint ribbon-ink characters that a median-blur-based background division can erode.
        if source_type not in ["CLEAN_SCAN", "DOT_MATRIX"]:
            res = self.flatten_illumination_and_folds(image)
        if settings.DESKEW_ENABLED:
            res = self.deskew(res)
        if source_type == "DOT_MATRIX":
            # Dot-matrix ink is faint/broken by nature - apply a small-kernel, no-blur
            # contrast pass unconditionally rather than only below a contrast threshold.
            res = self._enhance_dot_matrix_contrast(res)
        else:
            # Only apply subtle contrast enhancement if image contrast is extremely low
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY) if len(res.shape) == 3 else res
            if settings.CLAHE_ENABLED and np.std(gray) < 25.0:
                res = self.enhance_color_contrast(res)
        return res

    def _enhance_dot_matrix_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Small-kernel CLAHE pass tailored to dot-matrix printouts. Deliberately does not use
        bilateral/median blur (used elsewhere for photographed/crumpled documents), since
        blurring can merge or erase individual pin-dot strokes that are already faint/broken.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def process_id_card(self, image: np.ndarray) -> np.ndarray:
        """Existing proven high-accuracy pipeline for PAN and Aadhaar identity cards."""
        res = image
        if settings.DESKEW_ENABLED:
            res = self.deskew(res)
        if settings.CLAHE_ENABLED:
            res = self.enhance_color_contrast(res)
        blurred = cv2.GaussianBlur(res, (0, 0), 2.0)
        res = cv2.addWeighted(res, 1.25, blurred, -0.25, 0)
        if settings.NOISE_REMOVAL_ENABLED:
            res = cv2.bilateralFilter(res, d=5, sigmaColor=35, sigmaSpace=35)
        return res

    def process_general(self, image: np.ndarray) -> np.ndarray:
        """Standard document preprocessing."""
        res = image
        if settings.DESKEW_ENABLED:
            res = self.deskew(res)
        if settings.CLAHE_ENABLED:
            res = self.enhance_color_contrast(res)
        if settings.NOISE_REMOVAL_ENABLED:
            res = cv2.bilateralFilter(res, d=5, sigmaColor=30, sigmaSpace=30)
        return res

    def flatten_illumination_and_folds(self, image: np.ndarray) -> np.ndarray:
        """
        Morphological background illumination division to flatten paper creases,
        folds, and harsh lighting shadows on crumpled invoices and documents.
        """
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
            bg_smooth = cv2.medianBlur(dilated, 21)
            diff = 255 - cv2.absdiff(gray, bg_smooth)
            norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            # Blend normalized flat image with original to preserve color and subtle tone
            if len(image.shape) == 3:
                norm_bgr = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
                return cv2.addWeighted(image, 0.4, norm_bgr, 0.6, 0)
            return norm
        except Exception:
            return image

    def suppress_screen_glare(self, image: np.ndarray) -> np.ndarray:
        """
        Detects bright specular glare spots on mobile/monitor screen photographs
        and recovers underlying text using localized CLAHE and unsharp masking.
        """
        try:
            if len(image.shape) != 3:
                return image
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:, :, 2]
            # Identify over-exposed glare regions
            glare_mask = v_channel > 235
            if np.sum(glare_mask) > (0.01 * image.shape[0] * image.shape[1]):
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                hsv[:, :, 2] = clahe.apply(v_channel)
                enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                # Unsharp mask to sharpen faint edges under reflection
                blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
                sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
                return sharpened
            return image
        except Exception:
            return image

    def auto_crop_content_region(self, image: np.ndarray) -> np.ndarray:
        """
        Detects bright screen/document region in dark/shadowed phone photos
        and crops out excess dark background padding to maximize text resolution.
        Ensures full document height and margins are safely preserved.
        """
        try:
            h, w = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

            # 1. Letterbox Row/Col mean check (fast & precise for vertical/horizontal black bars)
            row_means = np.mean(gray, axis=1)
            col_means = np.mean(gray, axis=0)
            valid_rows = np.where(row_means > 15)[0]
            valid_cols = np.where(col_means > 15)[0]

            if len(valid_rows) > 0 and len(valid_cols) > 0:
                y1 = max(0, valid_rows[0] - 5)
                y2 = min(h, valid_rows[-1] + 5)
                x1 = max(0, valid_cols[0] - 5)
                x2 = min(w, valid_cols[-1] + 5)
                if (y2 - y1) < (0.95 * h) or (x2 - x1) < (0.95 * w):
                    logger.info(f"Auto-cropped black letterbox padding: x=[{x1}:{x2}], y=[{y1}:{y2}] (Original: {w}x{h})")
                    return image[y1:y2, x1:x2]

            # 2. Contour check for document boundary in screen photos
            _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                x, y, cw, ch = cv2.boundingRect(c)
                has_top_bottom = (y > 0.05 * h) and ((y + ch) < 0.95 * h)
                has_left_right = (x > 0.05 * w) and ((x + cw) < 0.95 * w)
                if (cw * ch) > (0.3 * w * h) and (has_top_bottom or has_left_right):
                    px = max(0, x - 10)
                    py = max(0, y - 10) if has_top_bottom else 0
                    pw = min(w - px, cw + 20)
                    ph = min(h - py, ch + 20) if has_top_bottom else (h - py)
                    return image[py:py+ph, px:px+pw]

            return image
        except Exception as e:
            logger.warning(f"Auto-crop content region failed: {str(e)}")
            return image

    def enhance_color_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhances contrast on Lightness channel of LAB color space, preserving full color."""
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            limg = cv2.merge((cl, a_channel, b_channel))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        except Exception:
            return image

    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Detects text skew angle using horizontal text lines and straightens slightly tilted scans."""
        try:
            (h, w) = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            min_len = max(40, w // 10)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=15)

            if lines is None or len(lines) < 3:
                return image

            angles = []
            for line in lines:
                pts = line.reshape(-1)
                if len(pts) != 4:
                    continue
                x1, y1, x2, y2 = pts
                if x2 - x1 == 0:
                    continue
                deg = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # Only consider slight tilt angles between -15 and +15 degrees
                if -15.0 <= deg <= 15.0:
                    angles.append(deg)

            if not angles or len(angles) < 5:
                return image

            angles_arr = np.array(angles)
            near_zero_pct = float(np.mean(np.abs(angles_arr) <= 1.0) * 100.0)
            std_angle = float(np.std(angles_arr))
            median_angle = float(np.median(angles_arr))

            # Safety Guards:
            # 1. If >= 20% of lines are already horizontal within 1 degree, do NOT rotate (screen text or table lines).
            # 2. If std_angle > 2.5 degrees, there is no uniform document skew (e.g. tilted monitor bezel vs straight text).
            # 3. Only rotate if there is tight consensus on a true document page tilt.
            if near_zero_pct >= 20.0 or std_angle > 2.5 or abs(median_angle) < 0.75 or abs(median_angle) > 15.0:
                logger.debug(f"Deskew bypassed: near_zero={near_zero_pct:.1f}%, std={std_angle:.2f}, median={median_angle:.2f}")
                return image

            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            logger.info(f"Deskew applied gentle angle correction: {median_angle:.2f} degrees (std={std_angle:.2f})")
            return rotated

        except Exception as e:
            logger.warning(f"Deskew failed: {str(e)}")
            return image

    # Below this, an image is too small for the orientation probe to be
    # meaningful - and a page that small has bigger problems (the legibility
    # gate will catch it).
    _ORIENT_MIN_DIM = 200
    # The probe runs OCR four times, so it runs on a downscaled copy. 1000px
    # on the long edge keeps body text recognisable while costing a fraction
    # of a full-resolution pass.
    _ORIENT_PROBE_DIM = 640
    # How much better a rotation must score before the page is turned. An
    # upright page usually wins by a wide margin; requiring a clear margin
    # stops a near-tie from flipping a page that was already correct.
    _ORIENT_MARGIN = 3.0
    # A first-pass score at or above this means the page reads well as
    # received, so it cannot be upside down and the second pass is skipped.
    # Inverted pages measured 5-9 against ~200 upright, so this sits far from
    # both: well above the inverted range, well below a normal page.
    _ORIENT_CONFIDENT_SCORE = 40.0

    _probe_engine = None
    _probe_engine_tried = False

    def _get_probe_engine(self):
        """
        A recogniser with the ANGLE CLASSIFIER TURNED OFF, built once.

        This detail is the whole trick. The normal engine runs a per-line
        angle classifier that silently turns each detected line the right way
        up, so it reads an upside-down page almost as well as an upright one
        - measured scores 184 vs 175, useless for deciding anything. With the
        classifier off the recogniser genuinely fails on inverted glyphs and
        the same comparison becomes 200 vs 6.
        """
        if self._probe_engine_tried:
            return self._probe_engine
        ImagePreprocessor._probe_engine_tried = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            ImagePreprocessor._probe_engine = RapidOCR(
                use_cls=False,
                det_box_thresh=settings.RAPIDOCR_DET_BOX_THRESH,
                text_score=settings.RAPIDOCR_TEXT_SCORE,
            )
        except Exception as exc:
            logger.warning(f"Orientation probe unavailable, pages used as received: {exc}")
            ImagePreprocessor._probe_engine = None
        return ImagePreprocessor._probe_engine

    def _orientation_score(self, image) -> float:
        """
        How much confident, word-like text this image yields.

        Sum of (confidence x alphanumeric length) over recognised tokens.
        Length-weighting stops a handful of confident single characters from
        outvoting real words.
        """
        engine = self._get_probe_engine()
        if engine is None:
            return 0.0
        try:
            result, _ = engine(image)
        except Exception:
            return 0.0
        score = 0.0
        for item in result or []:
            try:
                text, confidence = str(item[1]).strip(), float(item[2])
            except Exception:
                continue
            letters = sum(ch.isalnum() for ch in text)
            if letters >= 2:
                score += confidence * letters
        return score

    def correct_orientation(self, image):
        """
        Turn an upside-down page the right way up.

        Scope is deliberately 180 degrees only, because that is what the
        signal actually supports. Measured on a real PAN card, the four
        rotations score {0: 200, 90: 195, 180: 6, 270: 4}: inverted glyphs
        defeat the recogniser, but a quarter turn does not, because the
        detector finds oriented boxes and crops each line in its own reading
        direction. So 0 and 90 are not distinguishable this way - and they do
        not need to be, since a quarter-turned card already extracts
        correctly. An upside-down one does not: it returned the father's name
        as the holder's, at full confidence.

        Two probe passes, not four, and only on a downscaled copy.
        """
        try:
            h, w = image.shape[:2]
            if min(h, w) < self._ORIENT_MIN_DIM:
                return image
            if self._get_probe_engine() is None:
                return image

            scale = min(1.0, self._ORIENT_PROBE_DIM / float(max(h, w)))
            probe = (cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA)
                     if scale < 1.0 else image)

            upright = self._orientation_score(probe)

            # Short-circuit: the second pass only earns its cost when the
            # first one looks bad. Measured, an upright page scores around
            # 200 and an inverted one 5-9, so a page already reading well is
            # not upside down and there is nothing to compare against. This
            # keeps the common case at ONE probe pass; only a page that
            # reads poorly - sparse, blurred, or actually inverted - pays for
            # the second.
            if upright >= self._ORIENT_CONFIDENT_SCORE:
                return image

            inverted = self._orientation_score(cv2.rotate(probe, cv2.ROTATE_180))

            # Needs a decisive win, not a nudge: on a genuinely upright page
            # the gap is roughly thirty-fold, so a close call means the probe
            # does not know and the page is left exactly as it arrived.
            if inverted > max(upright, 1.0) * self._ORIENT_MARGIN:
                logger.info(
                    f"Page is upside down (legibility {upright:.0f} upright vs "
                    f"{inverted:.0f} inverted); turning it 180 degrees before OCR."
                )
                return cv2.rotate(image, cv2.ROTATE_180)
            return image
        except Exception as exc:
            # Never let the probe be the reason a document fails to process.
            logger.warning(f"Orientation probe failed, using the page as received: {exc}")
            return image

    def rotate_or_flip(self, image_path: Path, rotate_deg: int = 0, flip_horizontal: bool = False) -> Path:
        """
        Rotates an image by specified degrees (90, 180, 270) or flips horizontally (mirror fix).
        Overwrites/saves corrected image to output path.
        """
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return image_path

            modified = image.copy()

            if rotate_deg == 90:
                modified = cv2.rotate(modified, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_deg == 180:
                modified = cv2.rotate(modified, cv2.ROTATE_180)
            elif rotate_deg == 270:
                modified = cv2.rotate(modified, cv2.ROTATE_90_COUNTERCLOCKWISE)

            if flip_horizontal:
                modified = cv2.flip(modified, 1)  # 1 = horizontal flip

            output_path = settings.OUTPUT_DIR / f"transformed_{image_path.name}"
            cv2.imwrite(str(output_path), modified)
            logger.info(f"Image transformed (rotate={rotate_deg}, flip={flip_horizontal}): {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to transform image: {str(e)}", exc_info=True)
            return image_path


preprocessor = ImagePreprocessor()

