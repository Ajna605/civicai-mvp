from pathlib import Path
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from utils.table_utils import iter_block_items, clean_text, is_heading, get_heading_level, update_section_path, extract_table_rows, find_preceding_text, find_table_caption

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

def extract_pdf_tables(path):
    ...

def extract_tables_for_source(path, source_type):
    # if source_type == "pdf":
    #     return extract_pdf_tables(path)
    if source_type == "docx":
        return extract_docx_tables(path)
    return []