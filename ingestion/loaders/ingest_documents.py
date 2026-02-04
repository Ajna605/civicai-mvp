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
    s = s.replace("\u00a0", " ")   # non-breaking space
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


# ----------------------------
# Iterate blocks in doc order
# ----------------------------
def iter_block_items(doc: Document) -> Iterator[Tuple[str, Any]]:
    """
    Yield ("p", Paragraph) or ("tbl", Table) in the order they appear.
    """
    parent = doc.element.body
    for child in parent.iterchildren():
        if isinstance(child, CT_P):
            yield "p", Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield "tbl", Table(child, doc)


# ----------------------------
# List detection + nesting
# ----------------------------
def is_list_paragraph(p: Paragraph) -> bool:
    style_name = (p.style.name or "").lower() if p.style else ""
    return ("list" in style_name) or ("bullet" in style_name) or ("number" in style_name)


def get_list_level(p: Paragraph) -> int:
    """
    Best-effort nesting level via numbering properties; returns 0 if unknown.
    """
    try:
        numPr = p._p.pPr.numPr  # type: ignore[attr-defined]
        if numPr is None or numPr.ilvl is None:
            return 0
        return int(numPr.ilvl.val)
    except Exception:
        return 0


def paragraph_kind(p: Paragraph) -> str:
    """
    heading | list | paragraph
    """
    style_name = (p.style.name or "") if p.style else ""
    if style_name.lower().startswith("heading"):
        return "heading"
    if is_list_paragraph(p):
        return "list"
    return "paragraph"


# ----------------------------
# Table formatting
# ----------------------------
def table_to_markdown(tbl: Table) -> str:
    """
    Convert a Word table to a markdown-ish table for RAG.
    Uses first row as header if it looks header-like; otherwise creates col_1..col_N.
    """
    rows: List[List[str]] = []
    for row in tbl.rows:
        rows.append([clean_text(cell.text) for cell in row.cells])

    # Drop empty rows
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ""

    n_cols = max(len(r) for r in rows)
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    first = rows[0]
    headerish = all(c.strip() for c in first) and sum(len(c) for c in first) <= 200

    if headerish:
        header = first
        body = rows[1:]
    else:
        header = [f"col_{i+1}" for i in range(n_cols)]
        body = rows

    md: List[str] = []
    md.append("| " + " | ".join(header) + " |")
    md.append("| " + " | ".join(["---"] * n_cols) + " |")
    for r in body:
        md.append("| " + " | ".join((c if c else " ") for c in r) + " |")
    return "\n".join(md)


# ----------------------------
# Output schema
# ----------------------------
@dataclass
class Block:
    doc_id: str
    source_path: str
    block_index: int
    block_type: str  # heading | paragraph | list | table
    section_path: List[str]
    text: str
    extra: Dict[str, Any]


# ----------------------------
# Extraction
# ----------------------------
def extract_blocks_from_docx(docx_path: Path) -> List[Block]:
    doc = Document(str(docx_path))
    doc_id = docx_path.stem

    section_path: List[str] = []
    blocks: List[Block] = []
    block_index = 0

    # Group consecutive list paragraphs into one block
    current_list: List[Tuple[int, str]] = []
    current_list_section: Optional[List[str]] = None

    def flush_list() -> None:
        nonlocal block_index, current_list, current_list_section
        if not current_list:
            return
        lines: List[str] = []
        for lvl, t in current_list:
            indent = "  " * max(lvl, 0)
            lines.append(f"{indent}- {t}")

        blocks.append(
            Block(
                doc_id=doc_id,
                source_path=str(docx_path),
                block_index=block_index,
                block_type="list",
                section_path=(current_list_section or section_path.copy()),
                text=clean_text("\n".join(lines)),
                extra={"num_items": len(current_list)},
            )
        )
        block_index += 1
        current_list = []
        current_list_section = None

    for kind, obj in iter_block_items(doc):
        if kind == "p":
            p: Paragraph = obj
            txt = clean_text(p.text)
            if not txt:
                continue

            k = paragraph_kind(p)

            # Headings update section path and are emitted as blocks too
            if k == "heading":
                flush_list()
                style = p.style.name if p.style else ""
                m = re.search(r"(\d+)", style or "")
                lvl = int(m.group(1)) if m else 1

                section_path = section_path[: max(lvl - 1, 0)]
                section_path.append(txt)

                blocks.append(
                    Block(
                        doc_id=doc_id,
                        source_path=str(docx_path),
                        block_index=block_index,
                        block_type="heading",
                        section_path=section_path.copy(),
                        text=txt,
                        extra={"heading_level": lvl, "style": style},
                    )
                )
                block_index += 1
                continue

            # Lists: accumulate
            if k == "list":
                lvl = get_list_level(p)
                if current_list_section is None:
                    current_list_section = section_path.copy()
                current_list.append((lvl, txt))
                continue

            # Normal paragraph
            flush_list()
            blocks.append(
                Block(
                    doc_id=doc_id,
                    source_path=str(docx_path),
                    block_index=block_index,
                    block_type="paragraph",
                    section_path=section_path.copy(),
                    text=txt,
                    extra={"style": p.style.name if p.style else None},
                )
            )
            block_index += 1

        elif kind == "tbl":
            flush_list()
            tbl: Table = obj
            md = table_to_markdown(tbl)
            if not md.strip():
                continue

            blocks.append(
                Block(
                    doc_id=doc_id,
                    source_path=str(docx_path),
                    block_index=block_index,
                    block_type="table",
                    section_path=section_path.copy(),
                    text=md,
                    extra={
                        "rows": len(tbl.rows),
                        "cols": len(tbl.columns),
                        "format": "markdown_table",
                    },
                )
            )
            block_index += 1

    flush_list()
    return blocks

def extract_blocks_from_pdf(pdf_path: Path) -> List[Block]:
    """
    Adapter: PDF → NormalizedDoc → blocks
    """
    norm_doc = normalize_pdf(str(pdf_path))  # if normalize_pdf expects str; otherwise keep as pdf_path

    blocks: List[Block] = []
    block_index = 0

    seen_table_ids = set()

    for sec in norm_doc.sections:
        # 1) Emit the SECTION block
        blocks.append(
            Block(
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
                Block(
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
                blocks = extract_blocks_from_docx(path)
            else:
                blocks = extract_blocks_from_pdf(path)

            for b in blocks:
                f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
            total_blocks += len(blocks)

    print(f"[ingest_documents] Source={args.source} Files={len(files)}")
    print(f"[ingest_documents] Wrote {total_blocks} blocks → {out_path}")

if __name__ == "__main__":
    main()
