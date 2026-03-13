from docx.document import Document as _Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph
from utils.hash_utils import short_hash
from pathlib import Path
from typing import Any
import re

def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()

def iter_block_items(parent):
    parent_elm = parent.element.body

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def is_heading(paragraph: Paragraph) -> bool:
    style_name = paragraph.style.name if paragraph.style else ""
    return style_name.startswith("Heading")

def get_heading_level(paragraph: Paragraph) -> int:
    style_name = paragraph.style.name if paragraph.style else ""
    m = re.search(r"Heading\s+(\d+)", style_name)
    return int(m.group(1)) if m else 1

def update_section_path(current_headings: list[str], level: int, text: str) -> list[str]:
    while len(current_headings) >= level:
        current_headings.pop()
    current_headings.append(text)
    return current_headings

def extract_table_rows(table: Table) -> list[list[str]]:
    rows = []
    for row in table.rows:
        row_cells = [clean_text(cell.text) for cell in row.cells]
        rows.append(row_cells)
    return rows

def find_table_caption(recent_paragraphs: list[dict]) -> str | None:
    for p in reversed(recent_paragraphs[-3:]):
        text = p["text"]
        if re.match(r"^(Table|TABLE)\b", text):
            return text
    return None

def find_preceding_text(recent_paragraphs: list[dict], caption: str | None) -> str | None:
    texts = []
    for p in recent_paragraphs[-3:]:
        if caption and p["text"] == caption:
            continue
        texts.append(p["text"])
    return " ".join(texts[-2:]) if texts else None

def is_repeated_title_row(row: list[str]) -> bool:
    vals = [x.strip() for x in row if x and x.strip()]
    return len(vals) > 1 and len(set(vals)) == 1

## Creates RAG chunks from table summary used in build corpus
def build_table_summary_chunks(
    table_records: list[dict[str, Any]],
    *,
    source: str,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []

    for tbl in table_records:
        source_path = tbl.get("source_file")
        doc_id = Path(source_path).stem if source_path else "doc"
        table_id = tbl.get("table_id")
        table_index = tbl.get("table_index", 0)
        section_path = tbl.get("section_path", []) or []
        caption = (tbl.get("caption") or "").strip()
        preceding_text = (tbl.get("preceding_text") or "").strip()
        header_terms = tbl.get("header_terms", []) or []
        rows = tbl.get("rows", []) or []

        example_labels = []
        for row in rows[:2]:
            if row and str(row[0]).strip():
                example_labels.append(str(row[0]).strip())

        parts = []
        if caption:
            parts.append(caption)
        if preceding_text:
            parts.append(preceding_text)
        if header_terms:
            parts.append("Columns: " + ", ".join(str(h).strip() for h in header_terms if str(h).strip()))
        if example_labels:
            parts.append("Example categories: " + "; ".join(example_labels))

        text = " ".join(parts).strip()
        if not text:
            continue

        chunks.append({
            "id": f"{table_id}__summary__c0__{short_hash(text)}",
            "doc_id": doc_id,
            "source_path": source_path,
            "section_path": section_path,
            "section_index": -1,
            "block_type": "table_summary",
            "block_index": table_index,
            "text": text,
            "extra": {
                "source": source,
                "table_id": table_id,
                "table_index": table_index,
                "caption": caption or None,
                "header_terms": header_terms,
            },
        })

    return chunks


