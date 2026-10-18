"""PyMuPDF-based PDF parser.

Extracts text at the page-block level preserving reading order.
Tables are serialised to Markdown via fitz's built-in table finder.
Pages with (almost) no text layer - scans, photos of documents - are rendered
and sent through OCR, up to `max_pdf_ocr_pages` pages per file.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF

from notebook_rag.config import get_settings
from notebook_rag.ingestion import progress
from notebook_rag.ingestion.parsers.ocr import OCRUnavailableError, ocr_image
from notebook_rag.models import ParsedPage

log = logging.getLogger(__name__)


class PDFParser:
    """Parse a PDF file into one ParsedPage per PDF page."""

    def parse(self, path: Path, doc_id: str) -> list[ParsedPage]:
        cfg = get_settings()
        pages: list[ParsedPage] = []
        ocr_done = 0
        skipped_pages = 0
        ocr_missing = False
        with fitz.open(str(path)) as doc:
            total = doc.page_count
            for page_num, page in enumerate(doc, start=1):
                progress.update(doc_id, f"Reading page {page_num}/{total}…")
                text_parts: list[str] = []

                # Extract tables first, collect their bboxes to skip in text
                table_bboxes: list[fitz.Rect] = []
                try:
                    tabs = page.find_tables()
                    for tab in tabs.tables:
                        df = tab.to_markdown()
                        text_parts.append(df)
                        table_bboxes.append(tab.bbox)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Table extraction failed on page %d: %s", page_num, exc)

                # Extract blocks; skip those covered by a table bbox
                blocks = page.get_text("blocks", sort=True)  # (x0,y0,x1,y1,text,block_no,block_type)
                for block in blocks:
                    bx0, by0, bx1, by1, block_text, *_ = block
                    block_rect = fitz.Rect(bx0, by0, bx1, by1)
                    covered = any(block_rect.intersects(tb) for tb in table_bboxes)
                    if not covered and block_text.strip():
                        text_parts.append(block_text.strip())

                combined = "\n\n".join(text_parts).strip()

                # Scanned / image-only page: fall back to OCR
                if len(combined) < cfg.ocr_min_chars and cfg.ocr_enabled and not ocr_missing:
                    if ocr_done >= cfg.max_pdf_ocr_pages:
                        skipped_pages += 1
                    else:
                        try:
                            progress.update(doc_id, f"OCR on page {page_num}/{total} (slow)…")
                            pix = page.get_pixmap(dpi=cfg.ocr_dpi)
                            ocr_text = ocr_image(pix.tobytes("png"))
                            ocr_done += 1
                            combined = (combined + "\n\n" + ocr_text).strip()
                        except OCRUnavailableError as exc:
                            log.warning("%s", exc)
                            ocr_missing = True
                            skipped_pages += 1

                if not combined:
                    continue

                pages.append(
                    ParsedPage(
                        doc_id=doc_id,
                        page=page_num,
                        section=None,
                        text=combined,
                    )
                )

        warning = None
        if ocr_missing:
            warning = "Some pages have no text layer and the OCR engine is not installed, so they were skipped."
        elif skipped_pages:
            warning = (
                f"{skipped_pages} scanned page(s) were skipped: OCR is limited to the first "
                f"{cfg.max_pdf_ocr_pages} scanned pages per file."
            )
        if warning and pages:
            pages[0].metadata["warning"] = warning
        return pages
