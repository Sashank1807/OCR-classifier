import time, logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from app.core.config import settings
from app.core.logger import logger

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False
    logger.warning("rapidocr-onnxruntime not installed. Operating in Qwen-only mode.")


class RapidOCRService:
    """
    Ultra-fast local OCR service using RapidOCR / PaddleOCR ONNX models.
    Executes text & table line detection in ~0.3 - 3.5s per image.
    Calculates exact page confidence and reconstructs spatial Markdown tables.
    """

    def __init__(self):
        self.engine = RapidOCR(
            det_box_thresh=settings.RAPIDOCR_DET_BOX_THRESH,
            det_unclip_ratio=settings.RAPIDOCR_DET_UNCLIP_RATIO,
            text_score=settings.RAPIDOCR_TEXT_SCORE,
        ) if RAPID_OCR_AVAILABLE else None

    def extract_image_ocr(self, image_path: Path) -> Dict[str, Any]:
        """
        Runs RapidOCR on an image file path.
        Reconstructs table grid using spatial Y & X coordinate clustering.
        Returns extracted markdown, confidence score, and fallback status.
        """
        if not self.engine or not image_path.exists():
            return {
                "markdown": "",
                "confidence": 0.0,
                "lines_count": 0,
                "should_fallback": True,
                "elapsed": 0.0
            }

        t0 = time.time()
        try:
            result, elapse = self.engine(str(image_path))
            t1 = time.time()
            elapsed = round(t1 - t0, 3)

            if not result:
                return {
                    "markdown": "No text has been found.",
                    "confidence": 0.0,
                    "lines_count": 0,
                    "should_fallback": True,
                    "elapsed": elapsed
                }

            # Parse lines and bounding boxes
            scores = []
            boxes = []

            for item in result:
                box, text, score = item[0], item[1].strip(), float(item[2])
                if text:
                    scores.append(score)
                    y_center = sum([p[1] for p in box]) / len(box)
                    x_min = min([p[0] for p in box])
                    boxes.append({'y': y_center, 'x': x_min, 'text': text, 'score': score})

            avg_confidence = float(sum(scores) / len(scores)) if scores else 0.0

            # Spatial Row Clustering (Threshold: 14px vertical tolerance)
            boxes.sort(key=lambda b: b['y'])
            rows = []
            for b in boxes:
                matched = False
                for r in rows:
                    if abs(r['y_avg'] - b['y']) < 14:
                        r['items'].append(b)
                        r['y_avg'] = sum([item['y'] for item in r['items']]) / len(r['items'])
                        matched = True
                        break
                if not matched:
                    rows.append({'y_avg': b['y'], 'items': [b]})

            # Build Markdown output
            markdown_lines = []
            table_row_count = 0

            for r in rows:
                r['items'].sort(key=lambda item: item['x'])
                row_texts = [item['text'] for item in r['items']]

                if len(row_texts) > 2:
                    # Multi-column row -> format as Markdown table row
                    table_row = "| " + " | ".join(row_texts) + " |"
                    markdown_lines.append(table_row)
                    table_row_count += 1
                elif len(row_texts) == 2:
                    markdown_lines.append(f"**{row_texts[0]}**: {row_texts[1]}")
                else:
                    markdown_lines.append(row_texts[0])

            markdown_content = "\n".join(markdown_lines)

            # Trigger Qwen fallback if average page confidence < 95% or fewer than 3 lines found
            should_fallback = avg_confidence < 0.95 or len(boxes) < 3

            return {
                "markdown": markdown_content,
                "confidence": round(avg_confidence, 4),
                "lines_count": len(boxes),
                "should_fallback": should_fallback,
                "elapsed": elapsed
            }

        except Exception as e:
            logger.error(f"RapidOCR extraction failed for '{image_path.name}': {str(e)}")
            return {
                "markdown": "",
                "confidence": 0.0,
                "lines_count": 0,
                "should_fallback": True,
                "elapsed": 0.0
            }


rapid_ocr_service = RapidOCRService()
