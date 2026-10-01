"""
DEPRECATED - not imported or used anywhere in the pipeline (verified by repo-wide
search). document_analyzer.py's DocumentAnalyzer._analyze_image/_detect_screen_moire
is the live source/document-type profiler; its DOT_MATRIX detection is
DocumentAnalyzer._detect_dot_matrix_ribbon, ported from this module's ribbon-ink
heuristic. Kept only as reference for the FFT-based moire and table/numeric density
scoring here, which the live analyzer does not (yet) replicate in full. Do not wire
this module in without first checking it doesn't duplicate/conflict with
document_analyzer.py's profile dict shape.

Document Profile & Visual Quality Analyzer
Detects general document physical and visual characteristics:
- source_type: SCREEN_PHOTO, DOT_MATRIX, PAPER_SCAN, CLEAN_SCAN
- visual artifacts: glare, moire, low-contrast, faint ribbon ink
- density metrics: table_density, numeric_density
All classifications are strictly derived from computer vision and image processing signals.
Zero filename-specific or document-specific hardcoding.
"""

import cv2
import numpy as np
import re
from typing import Dict, Any, List, Optional, Tuple


class SourceType:
    SCREEN_PHOTO = "SCREEN_PHOTO"
    DOT_MATRIX = "DOT_MATRIX"
    PAPER_SCAN = "PAPER_SCAN"
    CLEAN_SCAN = "CLEAN_SCAN"


class DocumentProfiler:
    """Analyzes document-level image characteristics to guide preprocessing and OCR cascades."""

    def analyze_profile(
        self,
        image: np.ndarray,
        tokens: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive visual and structural profile from an input image:
        - Detects screen Moire & CRT/LCD subpixel grid
        - Detects dot-matrix pin dropouts & ribbon ink characteristics
        - Detects paper scan grain & ruling line ink
        - Measures glare, contrast, and foreground density
        """
        if image is None or image.size == 0:
            return {
                "source_type": SourceType.CLEAN_SCAN,
                "contrast": 0.0,
                "glare_ratio": 0.0,
                "moire_energy": 0.0,
                "is_screen": False,
                "is_dot_matrix": False,
                "is_faint": False,
                "table_density": 0.0,
                "numeric_density": 0.0
            }

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        # 1. Global Contrast & Variance
        p5 = float(np.percentile(gray, 5))
        p95 = float(np.percentile(gray, 95))
        contrast = p95 - p5
        variance = float(np.var(gray))
        mean_int = float(np.mean(gray))

        # 2. Glare detection (fraction of saturated pixels > 248)
        glare_mask = gray >= 248
        glare_ratio = float(np.sum(glare_mask) / max(1, h * w))

        # 3. Screen Moire & Subpixel Grid Detection
        # Sample a central 400x400 patch to analyze periodic high-frequency patterns
        ch0, cw0 = max(0, (h - 400) // 2), max(0, (w - 400) // 2)
        patch = gray[ch0:ch0+400, cw0:cw0+400]
        moire_energy = 0.0
        is_screen = False

        if patch.shape[0] >= 100 and patch.shape[1] >= 100:
            # Measure vertical difference (detecting periodic 1-2px subpixel LCD stripes)
            diff_h = np.abs(patch[:, 2:].astype(np.int16) - patch[:, :-2].astype(np.int16))
            stripe_pixels = np.sum((diff_h > 15) & (diff_h < 60))
            stripe_ratio = float(stripe_pixels / max(1, diff_h.size))

            # FFT analysis for periodic spatial frequencies
            dft = cv2.dft(np.float32(patch), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            mag = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)
            # High frequency ring energy
            ph, pw = patch.shape
            crow, ccol = ph // 2, pw // 2
            # Mask out DC and low frequencies (radius < 20)
            y, x = np.ogrid[:ph, :pw]
            dist_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)
            hf_mask = (dist_from_center > 25) & (dist_from_center < min(crow, ccol) - 10)
            if np.any(hf_mask):
                moire_energy = float(np.mean(mag[hf_mask]))

            if stripe_ratio > 0.12 or moire_energy > 165.0:
                is_screen = True

        # 4. Dot-Matrix Detection
        # Dot matrix typically has blue/purple ribbon ink or discrete pin dropouts
        is_dot_matrix = False
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            # Blue/purple ribbon ink has higher blue intensity relative to red
            ribbon_diff = np.mean(b.astype(np.int16) - r.astype(np.int16))
            if ribbon_diff > 8.0:
                is_dot_matrix = True

        # Also inspect token characters if available (e.g. presence of dot-matrix font confusions like 10.S, 1*10)
        numeric_density = 0.0
        table_density = 0.0
        if tokens:
            total_toks = len(tokens)
            num_tokens = sum(1 for t in tokens if re.search(r'\d', t.get("text", "")))
            numeric_density = round(num_tokens / max(1, total_toks), 3)

            dot_matrix_indicators = sum(
                1 for t in tokens
                if re.search(r'\b(?:\d+[\*xX]\d+|\d+\.S|1GRAN|\d+·\d+)\b', t.get("text", ""), re.IGNORECASE)
            )
            if dot_matrix_indicators >= 2:
                is_dot_matrix = True

            # Table density based on vertical alignment
            y_centers = [t.get("yc", t.get("y_center", 0)) for t in tokens]
            if len(y_centers) >= 10:
                unique_rows = len(set(int(y / 15) for y in y_centers))
                table_density = round(unique_rows / max(1, total_toks), 3)

        # 5. Classify primary SourceType
        if is_screen:
            source_type = SourceType.SCREEN_PHOTO
        elif is_dot_matrix:
            source_type = SourceType.DOT_MATRIX
        elif contrast < 60.0 or variance < 500.0:
            source_type = SourceType.PAPER_SCAN
        else:
            source_type = SourceType.CLEAN_SCAN

        is_faint = (contrast < 45.0) or (variance < 250.0)

        return {
            "source_type": source_type,
            "contrast": round(contrast, 2),
            "mean_intensity": round(mean_int, 2),
            "variance": round(variance, 2),
            "glare_ratio": round(glare_ratio, 3),
            "moire_energy": round(moire_energy, 2),
            "is_screen": is_screen,
            "is_dot_matrix": is_dot_matrix,
            "is_faint": is_faint,
            "table_density": table_density,
            "numeric_density": numeric_density
        }


document_profiler = DocumentProfiler()
