"""Shared helpers for parsers that produce Markdown tables."""
from __future__ import annotations

from typing import Iterable


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).replace("\r", " ").replace("\n", " ").replace("|", "\\|")
    return " ".join(text.split())


def rows_to_markdown(rows: Iterable[Iterable[object]]) -> str:
    """Render rows (first row = header) as a Markdown table.

    Rows are padded to the widest row; fully empty rows are dropped. The output is
    exactly what the chunker treats as an atomic table block.
    """
    clean = [[_cell(c) for c in row] for row in rows]
    clean = [r for r in clean if any(r)]
    if not clean:
        return ""
    width = max(len(r) for r in clean)
    # drop columns that are empty in every row
    keep = [i for i in range(width) if any(i < len(r) and r[i] for r in clean)]
    grid = [[(r[i] if i < len(r) else "") for i in keep] for r in clean]
    header, body = grid[0], grid[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)
