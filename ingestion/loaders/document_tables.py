from pathlib import Path
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from utils.table_utils import iter_block_items, clean_text, is_heading, get_heading_level, update_section_path, extract_table_rows, find_preceding_text, find_table_caption
import re
from pathlib import Path
from ingestion.loaders.pdf_loader import _extract_page_tables_pymupdf,_extract_page_tables_pdfplumber, _table_fingerprint

import pymupdf  # modern import name
import pdfplumber


def extract_docx_tables(path: Path) -> list[dict]:
    doc = Document(path)
    tables_out = []

    current_headings = []
    recent_paragraphs = []
    table_idx = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            text = clean_text(block.text)
            if not text:
                continue

            if is_heading(block):
                level = get_heading_level(block)
                current_headings = update_section_path(current_headings, level, text)

            recent_paragraphs.append({
                "text": text,
                "style": block.style.name if block.style else None,
            })
            recent_paragraphs = recent_paragraphs[-5:]

        elif isinstance(block, Table):
            rows = extract_table_rows(block)

            caption = find_table_caption(recent_paragraphs)
            preceding_text = find_preceding_text(recent_paragraphs, caption)

            tables_out.append({
                "source_type": "docx",
                "source_file": str(path),
                "table_index": table_idx,
                "section_path": list(current_headings),
                "caption": caption,
                "preceding_text": preceding_text,
                "rows": rows,
            })
            table_idx += 1

    return tables_out

def extract_pdf_tables(path: Path) -> list[dict]:
    """
    Extract raw table objects from a PDF.

    Returns a list of dicts intended for downstream normalization, similar to
    the raw table stage in a DOCX pipeline.

    Strategy:
    1. Use PyMuPDF page.find_tables() as primary extraction.
    2. Attach nearest caption above the table when possible.
    3. Normalize headers / rows into a consistent raw structure.
    4. Optionally fall back to pdfplumber for pages where PyMuPDF found nothing.

    Notes:
    - Coordinates use PDF page coordinate space.
    - Page numbers are returned as 1-based.
    - This assumes machine-generated PDFs, not scanned image-only PDFs.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    raw_tables: list[dict] = []
    seen_fingerprints: set[tuple] = set()

    doc = pymupdf.open(path.as_posix())
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_number = page_idx + 1

            page_tables = _extract_page_tables_pymupdf(
                page=page,
                path=path,
                page_number=page_number,
            )

            if not page_tables and pdfplumber is not None:
                page_tables = _extract_page_tables_pdfplumber(
                    path=path,
                    page_number=page_number,
                )

            for t in page_tables:
                fp = _table_fingerprint(t)
                if fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                raw_tables.append(t)
    finally:
        doc.close()

    return raw_tables

def extract_tables_for_source(path, source_type):
    if source_type == "pdf":
        return extract_pdf_tables(path)
    if source_type == "docx":
        return extract_docx_tables(path)
    return []