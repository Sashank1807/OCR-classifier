"""
Compare OCR engines on this deployment's own documents.

The question this answers is "would a different OCR engine read our documents
more accurately", and it is deliberately measured BELOW the table
reconstruction layer. Feeding two engines through the full pipeline would mix
engine accuracy with how well each one happens to suit the existing column
clustering, alignment scoring and cell cascade - all of which were tuned
against RapidOCR's output. What is measured here is the raw read.

Three metrics, because they fail differently:

  * cell recall    - what fraction of ground-truth cell values appear in the
                     engine's output at all. Catches text the DETECTOR missed
                     entirely, which no amount of downstream repair recovers.
  * numeric recall - the same, restricted to values that are numbers. Stock
                     and ERP data is mostly numeric, and a digit misread is
                     both the most likely error and the most costly one.
  * exact / CER    - for values the engine did find, how close the characters
                     are. Separates "missed it" from "read it wrong".

Recall is computed against a normalised concatenation of the whole page rather
than per cell, so an engine is not punished for splitting or merging boxes
differently - only for not reading the characters.

    python tools/compare_ocr_engines.py --engines rapidocr
    python tools/compare_ocr_engines.py --engines rapidocr,paddleocr --out outputs/ocr_engine_comparison.json

PaddleOCR is imported lazily and may live in a different interpreter (it pulls
in paddlepaddle, which conflicts with this project's pinned numpy/opencv), in
which case run it with --engines paddleocr from that interpreter and merge the
two report files with --merge.
"""

import argparse
import json
import re
import statistics
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

GT_DIR = ROOT / "tests" / "ground_truth"

# Ground-truth stem -> the image it was transcribed from. The fixture names
# were normalised at some point and the image names were not.
GT_IMAGES = {
    "1000411295": "test_data_june/1000411295.jpg",
    "1000411296": "test_data_june/1000411296.jpg",
    "1000517666": "test_data_june/1000517666.jpg",
    "1000517796": "test_data_june/1000517796.jpg",
    "agarwal_jaipur": "test_data_june/Agarwal Jaipur.jpeg",
    "bansal_barelly": "test_data_june/Bansal barelly.jpeg",
}

_NUM = re.compile(r"^-?[\d,]*\.?\d+%?$")


def _norm(text: str) -> str:
    """Fold away everything an engine can legitimately differ on."""
    return re.sub(r"[^a-z0-9.]", "", (text or "").lower())


def _is_numeric(value: str) -> bool:
    return bool(_NUM.match((value or "").strip()))


def _cer(expected: str, got: str) -> float:
    """Character error rate, 0.0 = identical."""
    if not expected:
        return 0.0
    ratio = SequenceMatcher(None, expected, got).ratio()
    return round(1.0 - ratio, 4)


# --------------------------------------------------------------------------- #
# Engines
#
# Each returns a list of (text, confidence). Boxes are not compared: the point
# is what characters came back, not where each engine chose to draw a box.
# --------------------------------------------------------------------------- #

def engine_rapidocr(**overrides) -> Tuple[str, Callable]:
    """The engine this deployment runs today: PP-OCRv4 mobile, ONNX runtime."""
    from rapidocr_onnxruntime import RapidOCR

    from app.core.config import settings

    kwargs = dict(
        det_box_thresh=settings.RAPIDOCR_DET_BOX_THRESH,
        det_unclip_ratio=settings.RAPIDOCR_DET_UNCLIP_RATIO,
        text_score=settings.RAPIDOCR_TEXT_SCORE,
    )
    kwargs.update({k: v for k, v in overrides.items() if v is not None})
    eng = RapidOCR(**kwargs)

    def run(image_path: str):
        result, _ = eng(image_path)
        return [(item[1], float(item[2])) for item in (result or [])]

    label = "rapidocr(" + ",".join(f"{k.replace('det_', '')}={v}" for k, v in sorted(kwargs.items())) + ")"
    return label, run


def engine_paddleocr(lang="en", **_) -> Tuple[str, Callable]:
    """
    PaddleOCR proper. 3.7 pulls PP-OCRv6 medium det+rec.

    oneDNN is disabled deliberately. With it on, every predict() on this
    machine raises `ConvertPirAttribute2RuntimeAttribute not support
    [pir::ArrayAttribute<pir::DoubleAttribute>]` from the oneDNN instruction
    path - a paddle/oneDNN incompatibility, not a model problem. Worth knowing
    before anyone plans a migration: the framework needed a workaround to run
    at all here, where the ONNX runtime needed none.
    """
    from paddleocr import PaddleOCR

    eng = PaddleOCR(lang=lang, use_textline_orientation=False, enable_mkldnn=False)

    def run(image_path: str):
        out = eng.predict(image_path)
        texts: List[Tuple[str, float]] = []
        for page in out or []:
            # 3.x returns dict-like results; 2.x returned nested lists. Accept
            # both rather than pinning a version this comparison then depends on.
            if isinstance(page, dict) or hasattr(page, "get"):
                rec = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                texts += [(t, float(s)) for t, s in zip(rec, scores)]
            else:
                for line in page or []:
                    try:
                        texts.append((line[1][0], float(line[1][1])))
                    except Exception:
                        pass
        return texts

    return f"paddleocr(lang={lang})", run


def engine_rapidocr_v6(**_) -> Tuple[str, Callable]:
    """
    Modern RapidOCR (3.x, the renamed `rapidocr` package) on PP-OCRv6 small.

    This is the cheap upgrade path and the one worth measuring before any
    migration: the same ONNX runtime the app already uses, the same integration
    code, but the current model generation instead of PP-OCRv4. If a newer
    MODEL is what closes the gap, nothing about the framework needs to change.
    """
    from rapidocr import RapidOCR as RapidOCRv3

    eng = RapidOCRv3()

    def run(image_path: str):
        out = eng(image_path)
        return list(zip(out.txts or [], [float(s) for s in (out.scores or [])]))

    return "rapidocr-3.x(PP-OCRv6 small)", run


ENGINES = {
    "rapidocr": engine_rapidocr,
    "rapidocr-v6": engine_rapidocr_v6,
    "paddleocr": engine_paddleocr,
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def _gt_values(fixture: Dict[str, Any]) -> List[str]:
    """
    Every cell value in a ground-truth fixture, plus its column headers.

    The fixture shapes differ and the difference is easy to get wrong:
    `columns` is a list of {name, type, x_order} objects, and `rows[].cells`
    is a DICT keyed by column name - so iterating it naively yields the column
    names once per row instead of the values, which silently scores an engine
    on the header text alone.
    """
    values: List[str] = []
    for column in fixture.get("columns") or []:
        name = column.get("name") if isinstance(column, dict) else column
        if name and str(name).strip():
            values.append(str(name))

    for row in fixture.get("rows") or []:
        cells = row.get("cells") if isinstance(row, dict) else row
        if isinstance(cells, dict):
            cells = list(cells.values())
        for cell in cells or []:
            if cell is not None and str(cell).strip():
                values.append(str(cell))
    return values


def score_page(gt_values: List[str], recognised: List[Tuple[str, float]]) -> Dict[str, Any]:
    haystack = _norm(" ".join(t for t, _ in recognised))
    per_line = [_norm(t) for t, _ in recognised if _norm(t)]

    found = missed = 0
    num_found = num_missed = 0
    missed_cers: List[float] = []
    near_miss = absent = 0
    missed_values: List[Dict[str, Any]] = []

    for value in gt_values:
        needle = _norm(value)
        if not needle:
            continue
        present = needle in haystack
        if present:
            found += 1
        else:
            missed += 1
            # For a value that is NOT present, how close did the nearest read
            # get? This is the split that matters: a near miss means the
            # detector found the text and the recogniser got characters wrong
            # (fixable by a better rec model, or by the cell cascade). An
            # absent value means the detector never saw it, and nothing
            # downstream can recover it.
            best = min((_cer(needle, line) for line in per_line), default=1.0)
            missed_cers.append(best)
            if best <= 0.34:
                near_miss += 1
            else:
                absent += 1
            if len(missed_values) < 25:
                closest = min(per_line, key=lambda l: _cer(needle, l)) if per_line else ""
                missed_values.append({"expected": value, "closest_read": closest,
                                      "cer": best})
        if _is_numeric(value):
            if present:
                num_found += 1
            else:
                num_missed += 1

    total = found + missed
    num_total = num_found + num_missed
    scores = [s for _, s in recognised]
    return {
        "gt_values": total,
        "recall": round(found / total, 4) if total else 0.0,
        "numeric_gt_values": num_total,
        "numeric_recall": round(num_found / num_total, 4) if num_total else 0.0,
        # Of what was missed, how it was missed.
        "missed_near": near_miss,
        "missed_absent": absent,
        "mean_cer_on_missed": round(statistics.mean(missed_cers), 4) if missed_cers else 0.0,
        "boxes_detected": len(recognised),
        "mean_engine_confidence": round(statistics.mean(scores), 4) if scores else 0.0,
        "missed_examples": missed_values,
    }


def run_engine(label: str, run: Callable, docs: List[Tuple[str, Path, List[str]]]) -> Dict[str, Any]:
    pages: Dict[str, Any] = {}
    timings: List[float] = []

    for stem, image, gt_values in docs:
        t0 = time.time()
        try:
            recognised = run(str(image))
            error = None
        except Exception as exc:
            recognised, error = [], f"{type(exc).__name__}: {exc}"
        elapsed = round(time.time() - t0, 3)
        timings.append(elapsed)

        page = score_page(gt_values, recognised)
        page["seconds"] = elapsed
        if error:
            page["error"] = error
        pages[stem] = page
        sys.stderr.write(
            f"  {stem:<18} recall {page['recall']:>6.1%}  numeric {page['numeric_recall']:>6.1%}  "
            f"boxes {page['boxes_detected']:>4}  near/absent {page['missed_near']:>3}/{page['missed_absent']:<3} "
            f"{elapsed:>6.2f}s"
            + (f"  ERROR {error}" if error else "") + "\n"
        )

    def agg(key: str, weight: str) -> float:
        """Weight by ground-truth value count, so a 4-row page does not
        count the same as a 29-row one."""
        num = sum(p[key] * p[weight] for p in pages.values())
        den = sum(p[weight] for p in pages.values())
        return round(num / den, 4) if den else 0.0

    return {
        "engine": label,
        "documents": len(pages),
        "recall": agg("recall", "gt_values"),
        "numeric_recall": agg("numeric_recall", "numeric_gt_values"),
        "missed_near": sum(p["missed_near"] for p in pages.values()),
        "missed_absent": sum(p["missed_absent"] for p in pages.values()),
        "total_boxes": sum(p["boxes_detected"] for p in pages.values()),
        "median_seconds": round(statistics.median(timings), 3) if timings else 0.0,
        "total_seconds": round(sum(timings), 2),
        "pages": pages,
    }


def load_docs(only: Optional[str]) -> List[Tuple[str, Path, List[str]]]:
    docs = []
    for stem, rel in sorted(GT_IMAGES.items()):
        if only and only not in stem:
            continue
        fixture = GT_DIR / f"{stem}.json"
        image = ROOT / rel
        if not fixture.exists() or not image.exists():
            sys.stderr.write(f"  skipping {stem}: missing fixture or image\n")
            continue
        values = _gt_values(json.loads(fixture.read_text(encoding="utf-8")))
        if not values:
            sys.stderr.write(f"  skipping {stem}: fixture has no cell values\n")
            continue
        docs.append((stem, image, values))
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="rapidocr",
                    help="comma separated: " + ",".join(ENGINES))
    ap.add_argument("--only", help="run a single ground-truth document")
    ap.add_argument("--lang", default="en", help="paddleocr language pack")
    ap.add_argument("--det-box-thresh", type=float)
    ap.add_argument("--det-unclip-ratio", type=float)
    ap.add_argument("--text-score", type=float)
    ap.add_argument("--out", default="outputs/ocr_engine_comparison.json")
    ap.add_argument("--merge", nargs="*", default=[],
                    help="existing report files to fold into this one")
    args = ap.parse_args()

    docs = load_docs(args.only)
    if not docs:
        raise SystemExit("No scorable ground-truth documents found.")

    sys.stderr.write(f"\n{len(docs)} document(s), "
                     f"{sum(len(v) for _, _, v in docs)} ground-truth cell values\n")

    results = []
    for name in [e.strip() for e in args.engines.split(",") if e.strip()]:
        if name not in ENGINES:
            raise SystemExit(f"Unknown engine '{name}'. Known: {', '.join(ENGINES)}")
        sys.stderr.write(f"\n--- {name}\n")
        try:
            label, run = ENGINES[name](
                lang=args.lang,
                det_box_thresh=args.det_box_thresh,
                det_unclip_ratio=args.det_unclip_ratio,
                text_score=args.text_score,
            )
        except ImportError as exc:
            sys.stderr.write(f"  {name} is not installed here: {exc}\n")
            continue
        results.append(run_engine(label, run, docs))

    for path in args.merge:
        prior = json.loads(Path(path).read_text(encoding="utf-8"))
        results = prior.get("results", []) + results

    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    sys.stderr.write("\n" + "=" * 104 + "\n")
    sys.stderr.write(f"  {'engine':<46} {'recall':>8} {'numeric':>8} {'near':>6} {'absent':>7} "
                     f"{'boxes':>7} {'median s':>9}\n")
    for r in results:
        sys.stderr.write(f"  {r['engine']:<46} {r['recall']:>7.1%} {r['numeric_recall']:>8.1%} "
                         f"{r['missed_near']:>6} {r['missed_absent']:>7} {r['total_boxes']:>7} "
                         f"{r['median_seconds']:>9.2f}\n")
    sys.stderr.write("=" * 104 + "\n")
    sys.stderr.write("  near   = value missed but nearest read is close (recogniser error, recoverable)\n")
    sys.stderr.write("  absent = value not read at all (detector miss, nothing downstream recovers it)\n")
    sys.stderr.write(f"\n  -> {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
