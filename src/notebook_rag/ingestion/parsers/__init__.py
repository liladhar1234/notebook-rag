"""Parser registry: pick a DocumentParser from a file's extension.

Parsers are created lazily so an optional library that isn't installed only
breaks the file types that need it, not the whole app.
"""
from __future__ import annotations

from pathlib import Path

from notebook_rag.models import ParsedPage  # noqa: F401  (re-export for convenience)

# extension (no dot) -> parser key
_EXT_TO_KEY: dict[str, str] = {
    "pdf": "pdf",
    "md": "markdown", "markdown": "markdown", "txt": "markdown",
    "docx": "word",
    "xlsx": "sheet", "xlsm": "sheet", "csv": "sheet", "tsv": "sheet",
    "pptx": "slides",
    "html": "web", "htm": "web",
    **{e: "image" for e in ("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff", "gif")},
    **{
        e: "text"
        for e in (
            "json", "jsonl", "xml", "yaml", "yml", "toml", "ini", "cfg", "conf", "log",
            "rst", "tex", "srt", "vtt", "py", "js", "ts", "tsx", "jsx", "java", "c", "h",
            "cpp", "hpp", "cs", "go", "rs", "rb", "php", "sh", "sql", "css", "swift", "kt",
        )
    },
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset("." + e for e in _EXT_TO_KEY)

# Formats we recognise but cannot read: tell the user what to do instead.
UNSUPPORTED_HINTS: dict[str, str] = {
    ".doc": "Old Word files (.doc) are not supported. Save it as .docx and upload again.",
    ".xls": "Old Excel files (.xls) are not supported. Save it as .xlsx or .csv and upload again.",
    ".ppt": "Old PowerPoint files (.ppt) are not supported. Save it as .pptx and upload again.",
    ".rtf": "RTF is not supported. Save it as .docx or .txt and upload again.",
    ".odt": "OpenDocument files are not supported. Save it as .docx and upload again.",
    ".zip": "Archives are not supported. Extract the files and upload them one by one.",
    ".mp3": "Audio is not supported.", ".wav": "Audio is not supported.",
    ".mp4": "Video is not supported.", ".mov": "Video is not supported.",
}

_CACHE: dict[str, object] = {}


def _build(key: str):
    if key == "pdf":
        from notebook_rag.ingestion.parsers.pdf import PDFParser
        return PDFParser()
    if key == "markdown":
        from notebook_rag.ingestion.parsers.markdown import MarkdownParser
        return MarkdownParser()
    if key == "word":
        from notebook_rag.ingestion.parsers.word import WordParser
        return WordParser()
    if key == "sheet":
        from notebook_rag.ingestion.parsers.sheet import SheetParser
        return SheetParser()
    if key == "slides":
        from notebook_rag.ingestion.parsers.slides import SlidesParser
        return SlidesParser()
    if key == "web":
        from notebook_rag.ingestion.parsers.web import WebParser
        return WebParser()
    if key == "image":
        from notebook_rag.ingestion.parsers.image import ImageParser
        return ImageParser()
    from notebook_rag.ingestion.parsers.text import TextParser
    return TextParser()


def looks_like_text(path: Path, sample_bytes: int = 8192) -> bool:
    """Heuristic: is this (unknown-extension) file readable text?"""
    with path.open("rb") as fh:
        sample = fh.read(sample_bytes)
    if not sample or b"\x00" in sample:
        return False
    text = sample.decode("utf-8", errors="replace")
    bad = text.count("�") + sum(1 for ch in text if ord(ch) < 9 or 13 < ord(ch) < 32)
    return bad / max(len(text), 1) < 0.05


def get_parser(path: Path):
    """Return the parser for `path`, falling back to plain text for unknown text files."""
    ext = path.suffix.lower().lstrip(".")
    key = _EXT_TO_KEY.get(ext)
    if key is None:
        if not looks_like_text(path):
            raise ValueError(f"Unsupported file type: '{path.suffix or 'no extension'}'")
        key = "text"
    if key not in _CACHE:
        _CACHE[key] = _build(key)
    return _CACHE[key]
