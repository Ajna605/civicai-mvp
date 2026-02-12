from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import csv
from ingestion.schema.document_schema import Table, Section, SourceInfo, NormalizedDoc
from utils.hash_utils import _sha256_file
from datetime import datetime

# import your existing types
# from .types import NormalizedDoc, Section, Table, SourceInfo

def _markdown_table(headers: List[str], rows: List[List[str]], max_rows: int = 25) -> str:
    """
    Render a small markdown table for retrieval.
    Keeps chunks bounded while still useful for table_lookup.
    """
    preview = rows[:max_rows]
    # Escape pipes to avoid broken markdown rendering
    esc = lambda s: (s or "").replace("|", "\\|")

    md = []
    md.append("| " + " | ".join(esc(h) for h in headers) + " |")
    md.append("| " + " | ".join("---" for _ in headers) + " |")
    for r in preview:
        r = (r + [""] * len(headers))[:len(headers)]
        md.append("| " + " | ".join(esc(str(x)) for x in r) + " |")

    if len(rows) > max_rows:
        md.append(f"\n(Previewing first {max_rows} of {len(rows)} rows.)")

    return "\n".join(md)

def normalize_csv(csv_path: str, doc_id: str | None = None) -> NormalizedDoc:
    file_name = os.path.basename(csv_path)
    doc_id = doc_id or os.path.splitext(file_name)[0]

    path = Path(csv_path)

    # Basic CSV read (no pandas needed)
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)

    if not all_rows:
        headers, data_rows = [], []
    else:
        headers = all_rows[0]
        data_rows = all_rows[1:]

    raw_text = _markdown_table(headers, data_rows, max_rows=25)

    # Stable, deterministic ID
    table_id = f"{doc_id}::csv::{path.name}::main"

    tbl = Table(
        table_id=table_id,
        page=-1,  # IMPORTANT: sentinel for non-PDF
        caption=f"CSV: {path.name}",
        raw_text=raw_text,
        rows=None,  # optional: set later if you want structured analysis
    )

    # Synthetic single section for the file
    sec = Section(
        section_id=f"{doc_id}::section::root",
        heading_path=[path.name],
        page_start=-1,
        page_end=-1,
        text="",  # CSV files don’t have narrative section text
        tables=[tbl],
    )

    norm_doc = NormalizedDoc(
        doc_id=doc_id,
        source=SourceInfo(file_name=file_name, file_type="csv", sha256 = _sha256_file(path), ingested_at=datetime.now().isoformat(),
                          ),  # adapt to your SourceInfo fields
        sections=[sec],
        tables=[tbl],
    )
    return norm_doc
