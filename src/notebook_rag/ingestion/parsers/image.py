"""Image parser: OCR the text in PNG/JPG/WEBP/BMP/TIFF/GIF files."""
from __future__ import annotations

from pathlib import Path

from notebook_rag.config import get_settings
from notebook_rag.ingestion.parsers.ocr import ocr_image
from notebook_rag.models import ParsedPage

MAX_FRAMES = 20


class ImageParser:
    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        import numpy as np
        from PIL import Image, ImageSequence

        if not get_settings().ocr_enabled:
            raise ValueError("OCR is disabled (OCR_ENABLED=false), so images can't be read.")

        pages: list[ParsedPage] = []
        with Image.open(path) as img:
            frames = list(ImageSequence.Iterator(img))[:MAX_FRAMES]
            for number, frame in enumerate(frames, start=1):
                text = ocr_image(np.array(frame.convert("RGB")))
                if text.strip():
                    pages.append(
                        ParsedPage(
                            doc_id=doc_id,
                            page=number if len(frames) > 1 else None,
                            section=path.stem,
                            text=text,
                        )
                    )
        return pages
