"""Local OCR via RapidOCR (ONNX, models ship inside the pip package: no downloads).

The engine is created lazily on first use and calls are serialised with a lock
(ingestion is already one-at-a-time, the lock is a safety net).
"""
from __future__ import annotations

import logging
import statistics
import threading

log = logging.getLogger(__name__)

_ENGINE = None
_LOCK = threading.Lock()
MIN_CONFIDENCE = 0.5


class OCRUnavailableError(RuntimeError):
    """The OCR engine is not installed."""


def _engine():
    global _ENGINE
    if _ENGINE is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover
            raise OCRUnavailableError(
                "OCR engine not installed (pip install rapidocr-onnxruntime)."
            ) from exc
        log.info("Loading OCR engine")
        _ENGINE = RapidOCR()
    return _ENGINE


def _lines_to_text(result) -> str:
    """Turn RapidOCR output [[box, text, score], ...] into reading-order text."""
    items = []
    for box, text, score in result or []:
        if not text or float(score) < MIN_CONFIDENCE:
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"y": sum(ys) / 4, "x": min(xs), "h": max(ys) - min(ys), "t": text})
    if not items:
        return ""
    tol = 0.6 * statistics.median(i["h"] for i in items)
    items.sort(key=lambda i: i["y"])
    lines: list[list[dict]] = []
    for it in items:
        if lines and abs(it["y"] - lines[-1][0]["y"]) <= tol:
            lines[-1].append(it)
        else:
            lines.append([it])
    return "\n".join(
        " ".join(i["t"] for i in sorted(line, key=lambda i: i["x"])) for line in lines
    )


def ocr_image(image) -> str:
    """OCR a path, bytes or numpy array. Returns '' when nothing readable is found."""
    engine = _engine()
    if isinstance(image, (bytes, bytearray)):
        import io

        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(image)) as im:
            image = np.array(im.convert("RGB"))
    with _LOCK:
        result, _ = engine(image)
    return _lines_to_text(result)
