# ingestion/load_docs.py
from __future__ import annotations
import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple
from pathlib import Path
from ingestion.normalize.normalize_pdf import normalize_pdf

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


# ----------------------------
# Arguments and paths
# ----------------------------
NORMALIZED_DIR = Path("data/normalized")
RAW_DIR = Path("data/raw")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pdf", "docx"], required=True)
    p.add_argument("--input", required=True, help="Path to the document")
    p.add_argument("--out-blocks", default=None, help="blocks.jsonl output path")
    return p.parse_args()

def default_out_for(source: str) -> Path:
    return NORMALIZED_DIR / source / "blocks.jsonl"

# ----------------------------
# Text cleanup
# ----------------------------
def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ----------------------------
# Iterate blocks in doc order
# ----------------------------
def iter_block_items(doc: Document) -> Iterator[Tuple[str, Any]]:
    parent = doc.element.body
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield "p", Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield "tbl", Table(child, doc)


# ----------------------------
# Lists / headings
# ----------------------------
def is_list_paragraph(p: Paragraph) -> bool:
    style_name = (p.style.name or "").lower() if p.style else ""
    return ("list" in style_name) or ("bullet" in style_name) or ("number" in style_name)


def get_list_level(p: Paragraph) -> int:
    try:
        numPr = p._p.pPr.numPr  # type: ignore[attr-defined]
        if numPr is None or numPr.ilvl is None:
            return 0
        return int(numPr.ilvl.val)
    except Exception:
        return 0


def is_heading_paragraph(p: Paragraph) -> bool:
    style_name = (p.style.name or "") if p.style else ""
    return style_name.lower().startswith("heading")


def get_heading_level(p: Paragraph) -> Optional[int]:
    if not is_heading_paragraph(p):
        return None
    style = p.style.name if p.style else ""
    m = re.search(r"(\d+)", style or "")
    return int(m.group(1)) if m else 1


# ----------------------------
# Tables -> structured JSON
# ----------------------------
def table_to_struct(tbl: Table) -> Dict[str, Any]:
    """
    Returns:
      {
        "header": [..],
        "rows": [[..], [..], ...],
        "n_rows": int,
        "n_cols": int
      }
    """
    rows: List[List[str]] = []
    for row in tbl.rows:
        cells = [clean_text(cell.text) for cell in row.cells]
        rows.append(cells)

    # drop empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return {"header": [], "rows": [], "n_rows": 0, "n_cols": 0}

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    first = rows[0]
    headerish = all(c.strip() for c in first) and sum(len(c) for c in first) <= 240
    if headerish:
        header = first
        body = rows[1:]
    else:
        header = [f"col_{i+1}" for i in range(n_cols)]
        body = rows

    return {
        "header": header,
        "rows": body,
        "n_rows": len(body),
        "n_cols": n_cols,
    }


# ----------------------------
# Output schema
# ----------------------------
@dataclass
class SectionRecord:
    doc_id: str
    source_path: str
    section_index: int

    path: List[str]
    path_text: str
    title: str
    heading_level: Optional[int]

    # Instead of one big string, store structured blocks
    blocks: List[Dict[str, Any]]  # each: {"type": "...", ...}


def find_docx_files() -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw folder at: {RAW_DIR}")
    return sorted(RAW_DIR.rglob("*.docx"))


# ----------------------------
# Extraction (CONDENSED sections w/ blocks)
# ----------------------------
def extract_sections_from_docx(docx_path: Path) -> List[SectionRecord]:
    doc = Document(str(docx_path))
    doc_id = docx_path.stem

    heading_stack: List[Tuple[int, str]] = []

    # list grouping
    current_list: List[Tuple[int, str]] = []

    # current section
    current_section_title: Optional[str] = None
    current_section_level: Optional[int] = None
    current_section_path: List[str] = []
    blocks: List[Dict[str, Any]] = []

    sections: List[SectionRecord] = []
    section_index = 0

    def flush_list_into_blocks() -> None:
        nonlocal current_list, blocks
        if not current_list:
            return
        items = [{"level": lvl, "text": t} for (lvl, t) in current_list]
        blocks.append({"type": "list", "items": items})
        current_list = []

    def flush_section() -> None:
        nonlocal section_index, blocks, current_section_title, current_section_path, current_section_level
        flush_list_into_blocks()

        # only write if there is a heading + at least one block
        if current_section_title and blocks:
            path_text = " > ".join(current_section_path) if current_section_path else current_section_title
            sections.append(
                SectionRecord(
                    doc_id=doc_id,
                    source_path=str(docx_path),
                    section_index=section_index,
                    path=current_section_path.copy(),
                    path_text=path_text,
                    title=current_section_title,
                    heading_level=current_section_level,
                    blocks=blocks,
                )
            )
            section_index += 1

        blocks = []

    for kind, obj in iter_block_items(doc):
        if kind == "p":
            p: Paragraph = obj
            txt = clean_text(p.text)
            if not txt:
                continue

            lvl = get_heading_level(p)
            if lvl is not None:
                flush_section()

                # correct sibling handling
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, txt))

                current_section_title = txt
                current_section_level = lvl
                current_section_path = [t for _, t in heading_stack]
                continue

            if is_list_paragraph(p):
                current_list.append((get_list_level(p), txt))
                continue

            flush_list_into_blocks()
            blocks.append({"type": "paragraph", "text": txt})

        elif kind == "tbl":
            flush_list_into_blocks()
            tbl: Table = obj
            struct = table_to_struct(tbl)
            if struct["n_rows"] == 0 and struct["n_cols"] == 0:
                continue
            blocks.append({"type": "table", "table": struct})

    flush_section()
    return sections



def extract_sections_from_pdf(pdf_path: Path) -> List[SectionRecord]:
    """
    Adapter: PDF → NormalizedDoc → blocks
    """
    norm_doc = normalize_pdf(str(pdf_path))  # if normalize_pdf expects str; otherwise keep as pdf_path

    blocks: List[SectionRecord] = []
    block_index = 0

    seen_table_ids = set()

    for sec in norm_doc.sections:
        # 1) Emit the SECTION block
        blocks.append(
            SectionRecord(
                doc_id=norm_doc.doc_id,
                source_path=str(pdf_path),
                block_type="section",
                block_index=block_index,
                section_path=sec.heading_path,
                text=sec.text or "",
                extra={
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "section_id": getattr(sec, "section_id", None),
                },
            )
        )
        block_index += 1

        # 2) Emit TABLE blocks under that section (dedup by table_id)
        for tbl in getattr(sec, "tables", []) or []:
            if tbl.table_id in seen_table_ids:
                continue
            seen_table_ids.add(tbl.table_id)

            blocks.append(
                SectionRecord(
                    doc_id=norm_doc.doc_id,
                    source_path=str(pdf_path),
                    block_type="table",
                    block_index=block_index,
                    section_path=sec.heading_path,  # inherit section path
                    text=tbl.raw_text or "",
                    extra={
                        "caption": tbl.caption,
                        "raw_text": tbl.raw_text,
                        "page": tbl.page,
                        "table_id": tbl.table_id,
                    },
                )
            )
            block_index += 1

    return blocks


    # Tables
    for tbl in norm_doc.tables:
        blocks.append(
            Block(
                doc_id=norm_doc.doc_id,
                source_path=str(pdf_path),
                block_type="table",
                block_index=block_index,
                section_path=tbl.section_path if hasattr(tbl, "section_path") else [],
                text=tbl.raw_text,
                extra={
                    "caption": tbl.caption,
                    "raw_text": tbl.raw_text,
                    "page": tbl.page,
                    "table_id": tbl.table_id,
                },
            )
        )
        block_index += 1

    return blocks



def find_docx_files() -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data folder at: {RAW_DIR}")
    return sorted(RAW_DIR.rglob("*.docx"))

def find_pdf_files() -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data folder at: {RAW_DIR}")
    return sorted(RAW_DIR.rglob("*.pdf"))


def main() -> None:
    args = parse_args()

    out_path = default_out_for(args.source)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose inputs
    if args.input:
        files = [Path(args.input)]
        # Optional: validate extension matches source
        if args.source == "docx" and files[0].suffix.lower() != ".docx":
            raise ValueError(f"--source docx requires a .docx input, got: {files[0]}")
        if args.source == "pdf" and files[0].suffix.lower() != ".pdf":
            raise ValueError(f"--source pdf requires a .pdf input, got: {files[0]}")
    else:
        if args.source == "docx":
            files = find_docx_files()
        else:
            files = find_pdf_files()

    if not files:
        raise FileNotFoundError(
            f"No {args.source.upper()} files found (input={args.input!r})."
        )

    total_blocks = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in files:
            if args.source == "docx":
                blocks = extract_sections_from_docx(path)
            else:
                blocks = extract_sections_from_pdf(path)

            for b in blocks:
                f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
            total_blocks += len(blocks)

    print(f"[ingest_documents] Source={args.source} Files={len(files)}")
    print(f"[ingest_documents] Wrote {total_blocks} blocks → {out_path}")

if __name__ == "__main__":
    main()
