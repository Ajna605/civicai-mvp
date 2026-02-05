
import os
from typing import List, Tuple, Optional, Dict
from ingestion.loaders.pdf_loader import pages_to_sections, extract_tables_from_page_layout, extract_pdf_pages, _sha256_file
from datetime import datetime
from ingestion.schema.document_schema import NormalizedDoc, SourceInfo, Section, Table
import fitz  # PyMuPDF

def normalize_pdf(pdf_path: str, doc_id: str | None = None) -> NormalizedDoc:
    file_name = os.path.basename(pdf_path)
    doc_id = doc_id or os.path.splitext(file_name)[0]

    # existing: cleaned page text for sectioning
    pages = extract_pdf_pages(pdf_path)

    # NEW: layout-aware tables from the actual PDF pages
    page_tables: Dict[int, List[Table]] = {}
    table_counter = 0

    doc = fitz.open(pdf_path)
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            tcs = extract_tables_from_page_layout(page)
            for tc in tcs:
                table_counter += 1
                t = Table(
                    table_id=f"t{table_counter:04d}",
                    page=tc["page"],
                    caption=tc.get("caption"),
                    raw_text=tc.get("raw_text"),
                    rows=None,
                )
                page_tables.setdefault(tc["page"], []).append(t)
    finally:
        doc.close()

    secs = pages_to_sections(pages)

    out_sections: list[Section] = []
    for s in secs:
        tables_for_section: List[Table] = []
        for pg in range(s.page_start, s.page_end + 1):
            tables_for_section.extend(page_tables.get(pg, []))

        out_sections.append(
            Section(
                section_id=s.section_id,
                heading_path=s.heading_path,
                page_start=s.page_start,
                page_end=s.page_end,
                text=s.text,
                tables=tables_for_section,
            )
        )

    src = SourceInfo(
        file_name=file_name,
        file_type="pdf",
        sha256=_sha256_file(pdf_path),
        ingested_at=datetime.now().isoformat(),
    )

    all_tables = []
    for s in out_sections:
        all_tables.extend(s.tables)

    return NormalizedDoc(doc_id=doc_id, source=src, sections=out_sections, tables=all_tables)

