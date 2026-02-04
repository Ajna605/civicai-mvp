# ingestion/load_docs.py
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


# ----------------------------
# Paths (repo-aware)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_OUT = PROCESSED_DIR / "sections.jsonl"


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


def main() -> None:
    docx_files = find_docx_files()
    if not docx_files:
        raise FileNotFoundError(f"No .docx found under: {RAW_DIR}")

    total_sections = 0
    with open(DEFAULT_OUT, "w", encoding="utf-8") as f:
        for docx_path in docx_files:
            secs = extract_sections_from_docx(docx_path)
            for s in secs:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            total_sections += len(secs)

    print(f"[load_docs] Found {len(docx_files)} docx file(s)")
    print(f"[load_docs] Wrote {total_sections} sections → {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
