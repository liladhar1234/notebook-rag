"""Parser tests for the non-PDF formats.

Formats that need an optional library are skipped when it isn't installed.
"""
from __future__ import annotations

import json

import pytest

from notebook_rag.ingestion.parsers import SUPPORTED_EXTENSIONS, get_parser, looks_like_text
from notebook_rag.ingestion.parsers._tables import rows_to_markdown


def test_rows_to_markdown_pads_escapes_and_drops_empty():
    md = rows_to_markdown([["a", "b|c", None], ["1", "2"], [None, None, None], [3.0, "x", ""]])
    lines = md.splitlines()
    assert lines[0] == "| a | b\\|c |"
    assert lines[1] == "|---|---|"
    assert "| 3 | x |" in lines          # 3.0 rendered as 3, empty column dropped
    assert len(lines) == 4               # header, separator, 2 data rows


def test_registry_maps_extensions():
    for ext in (".pdf", ".docx", ".xlsx", ".csv", ".pptx", ".html", ".png", ".json", ".md", ".txt"):
        assert ext in SUPPORTED_EXTENSIONS


def test_looks_like_text(tmp_path):
    t = tmp_path / "a.xyz"; t.write_text("hello world\nsecond line")
    b = tmp_path / "b.xyz"; b.write_bytes(bytes(range(256)) * 4)
    assert looks_like_text(t) and not looks_like_text(b)


def test_unknown_binary_rejected_by_registry(tmp_path):
    b = tmp_path / "b.xyz"; b.write_bytes(b"\x00\x01\x02" * 100)
    with pytest.raises(ValueError):
        get_parser(b)


def test_csv_becomes_one_table(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name;qty\nbolt;5\nnut;7\n", encoding="utf-8")
    pages = get_parser(f).parse(f, "d")
    assert len(pages) == 1
    assert pages[0].text.splitlines()[0] == "| name | qty |"
    assert "| nut | 7 |" in pages[0].text


def test_csv_row_cap_sets_warning(tmp_path, monkeypatch):
    from notebook_rag.config import get_settings
    monkeypatch.setattr(get_settings(), "max_sheet_rows", 10)
    f = tmp_path / "big.csv"
    f.write_text("id,v\n" + "\n".join(f"{i},{i}" for i in range(50)))
    pages = get_parser(f).parse(f, "d")
    assert "first 10 rows" in pages[0].metadata["warning"]


def test_json_is_pretty_printed(tmp_path):
    f = tmp_path / "x.json"
    f.write_text(json.dumps({"a": {"b": [1, 2, 3]}}))
    pages = get_parser(f).parse(f, "d")
    assert "\n" in pages[0].text and '"b"' in pages[0].text


def test_html_sections_tables_and_noise(tmp_path):
    f = tmp_path / "p.html"
    f.write_text(
        "<html><head><title>x</title><style>p{}</style></head><body>"
        "<script>var a=1;</script>"
        "<h1>Guide</h1><p>Intro text.</p>"
        "<h2>Pricing</h2><p>See table.</p>"
        "<table><tr><th>Plan</th><th>Cost</th></tr><tr><td>Pro</td><td>20</td></tr></table>"
        "</body></html>"
    )
    pages = get_parser(f).parse(f, "d")
    text = "\n".join(p.text for p in pages)
    assert "var a=1" not in text and "p{}" not in text
    assert any(p.section == "Guide > Pricing" for p in pages)
    assert "| Plan | Cost |" in text and "| Pro | 20 |" in text


def test_docx_headings_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    d = docx.Document()
    d.add_heading("Report", level=1)
    d.add_paragraph("Opening paragraph.")
    d.add_heading("Costs", level=2)
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Item", "Price"
    t.cell(1, 0).text, t.cell(1, 1).text = "Widget", "9"
    f = tmp_path / "r.docx"; d.save(str(f))
    pages = get_parser(f).parse(f, "d")
    assert [p.section for p in pages] == ["Report", "Report > Costs"]
    assert "| Widget | 9 |" in pages[1].text


def test_xlsx_one_page_per_sheet(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    wb = openpyxl.Workbook()
    ws = wb.active; ws.title = "Q1"
    ws.append(["month", "sales"]); ws.append(["Jan", 100])
    ws2 = wb.create_sheet("Q2"); ws2.append(["month", "sales"]); ws2.append(["Apr", 250])
    f = tmp_path / "s.xlsx"; wb.save(str(f))
    pages = get_parser(f).parse(f, "d")
    assert [p.section for p in pages] == ["Q1", "Q2"]
    assert "| Apr | 250 |" in pages[1].text


def test_pptx_slide_text_and_notes(tmp_path):
    pptx = pytest.importorskip("pptx")
    prs = pptx.Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = "Roadmap"
    slide.placeholders[1].text = "Ship stage two"
    slide.notes_slide.notes_text_frame.text = "Mention the deadline"
    f = tmp_path / "d.pptx"; prs.save(str(f))
    pages = get_parser(f).parse(f, "d")
    assert pages[0].page == 1
    assert "Ship stage two" in pages[0].text and "Mention the deadline" in pages[0].text


def test_image_ocr_reads_text(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    PIL = pytest.importorskip("PIL.Image")
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(img).text((30, 50), "INVOICE TOTAL 4200", fill="black", font=ImageFont.load_default(size=60))
    f = tmp_path / "scan.png"; img.save(f)
    pages = get_parser(f).parse(f, "d")
    assert pages and "4200" in pages[0].text.replace(" ", "")
