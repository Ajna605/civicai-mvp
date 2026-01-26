# ingestion/load_docs.py
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

# Condensed output: one record per heading section (Goal/Objective/Policy/etc.)
DEFAULT_OUT = PROCESSED_DIR / "sections.jsonl"


# ----------------------------
# Text cleanup
# ----------------------------
def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")  # non-breaking space
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


def is_heading_paragraph(p: Paragraph) -> bool:
    style_name = (p.style.name or "") if p.style else ""
    return style_name.lower().startswith("heading")


def get_heading_level(p: Paragraph) -> Optional[int]:
    """
    Parse 'Heading 2' -> 2, returns None if not heading.
    """
    if not is_heading_paragraph(p):
        return None
    style = p.style.name if p.style else ""
    m = re.search(r"(\d+)", style or "")
    return int(m.group(1)) if m else 1


# ----------------------------
# Table formatting
# ----------------------------
def table_to_markdown(tbl: Table) -> str:
    rows: List[List[str]] = []
    for row in tbl.rows:
        rows.append([clean_text(cell.text) for cell in row.cells])

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
# Condensed "section" schema
# ----------------------------
@dataclass
class SectionRecord:
    doc_id: str
    source_path: str
    section_index: int

    # Heading stack info
    path: List[str]          # ["Administration Element", "Goal ADM-1.", "Objective ADM-1.1.", "Policy ADM-1.1.2."]
    path_text: str           # "Administration Element > Goal ADM-1. > Objective ADM-1.1. > Policy ADM-1.1.2."
    title: str               # "Policy ADM-1.1.2."
    heading_level: Optional[int]

    # Content under that heading
    content: str             # paragraphs/lists/tables combined
    extra: Dict[str, Any]


def find_docx_files() -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data folder at: {RAW_DIR}")
    return sorted(RAW_DIR.rglob("*.docx"))


# ----------------------------
# Extraction (CONDENSED)
# ----------------------------
def extract_sections_from_docx(docx_path: Path) -> List[SectionRecord]:
    doc = Document(str(docx_path))
    doc_id = docx_path.stem

    # Heading stack: list of (level:int, text:str)
    heading_stack: List[Tuple[int, str]] = []

    # Accumulate list items before flushing into current section content
    current_list: List[Tuple[int, str]] = []

    # Current section accumulator
    current_section_title: Optional[str] = None
    current_section_level: Optional[int] = None
    current_section_path: List[str] = []
    content_parts: List[str] = []

    sections: List[SectionRecord] = []
    section_index = 0

    def flush_list_into_content() -> None:
        nonlocal current_list, content_parts
        if not current_list:
            return
        lines: List[str] = []
        for lvl, t in current_list:
            indent = "  " * max(lvl, 0)
            lines.append(f"{indent}- {t}")
        content_parts.append(clean_text("\n".join(lines)))
        current_list = []

    def flush_section() -> None:
        nonlocal section_index, content_parts, current_section_title, current_section_path, current_section_level
        flush_list_into_content()
        content = clean_text("\n\n".join([p for p in content_parts if p.strip()]))

        # Only write a section if it has a heading AND some content
        if current_section_title and content:
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
                    content=content,
                    extra={},
                )
            )
            section_index += 1

        # reset content accumulator (heading info set when next heading arrives)
        content_parts = []

    for kind, obj in iter_block_items(doc):
        if kind == "p":
            p: Paragraph = obj
            txt = clean_text(p.text)
            if not txt:
                continue

            lvl = get_heading_level(p)

            # --- Heading encountered ---
            if lvl is not None:
                # finish prior section (if any)
                flush_section()

                # Maintain correct hierarchy:
                # pop while last_level >= current_level (siblings replace each other)
                while heading_stack and heading_stack[-1][0] >= lvl:
                    heading_stack.pop()
                heading_stack.append((lvl, txt))

                current_section_title = txt
                current_section_level = lvl
                current_section_path = [t for _, t in heading_stack]

                continue

            # --- List paragraph ---
            if is_list_paragraph(p):
                current_list.append((get_list_level(p), txt))
                continue

            # --- Normal paragraph ---
            flush_list_into_content()
            content_parts.append(txt)

        elif kind == "tbl":
            # table belongs to current section
            flush_list_into_content()
            tbl: Table = obj
            md = table_to_markdown(tbl)
            if md.strip():
                content_parts.append(md)

    # flush last section at EOF
    flush_section()
    return sections


def main() -> None:
    docx_files = find_docx_files()
    if not docx_files:
        raise FileNotFoundError(f"No .docx files found under: {RAW_DIR}")

    total_sections = 0
    with open(DEFAULT_OUT, "w", encoding="utf-8") as f:
        for docx_path in docx_files:
            secs = extract_sections_from_docx(docx_path)
            for s in secs:
                f.write(json.dumps(asdict(s), ensure_ascii=False) + "\n")
            total_sections += len(secs)

    print(f"[load_docs] Found {len(docx_files)} .docx file(s) in {RAW_DIR}")
    print(f"[load_docs] Wrote {total_sections} condensed sections → {DEFAULT_OUT}")


if __name__ == "__main__":
    main()
