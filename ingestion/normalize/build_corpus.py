# Creates rag_chunks.jsonl and sections.jsonl
# from blocks.jsonl
# ingestion/preprocess.py
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple, Iterable
from ingestion.normalize import normalize_pdf

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

# ----------------------------
# Paths (repo-aware)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "normalized"
BLOCKS_FILE = PROCESSED_DIR / "blocks.jsonl"



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build retrieval corpus chunks from blocks.jsonl")
    p.add_argument("--source", choices=["pdf", "docx"], default=None,
                   help="Use standard folder layout under data/normalized/<source>/")
    p.add_argument("--base", type=str, default=str(PROCESSED_DIR),
                   help="Base normalized dir (default: data/normalized)")
    p.add_argument("--blocks", type=str, default=None,
                   help="Explicit path to blocks.jsonl (overrides --source)")
    p.add_argument("--out", type=str, default=None,
                   help="Explicit output path for rag_chunks.jsonl (overrides --source)")
    return p.parse_args()

def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    base = Path(args.base)
    if args.blocks:
        blocks = Path(args.blocks)
    else:
        if not args.source:
            raise ValueError("Provide either --source (pdf/docx) or --blocks path.")
        blocks = base / args.source / "blocks.jsonl"

    if args.out:
        out = Path(args.out)
    else:
        if not args.source and not args.blocks:
            raise ValueError("Provide either --source (pdf/docx) or --out path.")
        # If blocks path was explicit but source omitted, default output to sibling rag_chunks.jsonl
        if args.source:
            out = base / args.source / "rag_chunks.jsonl"
        else:
            out = blocks.parent / "rag_chunks.jsonl"

    return blocks, out

def find_pdf_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {input_path}")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Expected input at: {input_path}")
    return sorted(input_path.rglob("*.pdf"))


def find_docx_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".docx":
            raise ValueError(f"Expected a .docx file, got: {input_path}")
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Expected input at: {input_path}")
    return sorted(input_path.rglob("*.docx"))
# ----------------------------
# Table detection
# ----------------------------
def table_dims(raw_text: str) -> tuple[int, int]:
    lines = [ln for ln in (raw_text or "").splitlines() if ln.strip()]
    rows = len(lines)
    cols = max((len(ln.split(" | ")) for ln in lines), default=0)
    return rows, cols

def looks_like_table(raw: str) -> bool:
    if not raw or raw.strip() == "":
        return False
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    # must have at least 2 lines with pipes
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) < 2:
        return False
    # reject obvious narrative lists like "evaluation of the following"
    # (caption check can also be added, but keep it here purely text-based)
    bulletish = sum(1 for ln in lines[:6] if ln.startswith(("•", "-", "*")))
    # allow bullet-ish tables only if they look like 2+ columns consistently
    if bulletish >= 2:
        # require that most pipe lines have at least 2 separators (=> 3 cells),
        # otherwise it's probably just "• | item"
        rich = sum(1 for ln in pipe_lines if ln.count("|") >= 2)
        if rich == 0:
            return False
    return True



def table_tier(caption: str | None, raw_text: str) -> str:
    rows, cols = table_dims(raw_text)
    cap_ok = bool(caption) and ("table" in (caption or "").lower())
    if cap_ok and ((rows >= 6) or (cols >= 3)):
        return "A"
    return "B"

# ----------------------------
# Cleanup / hashing
# ----------------------------
def clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = s.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]



def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 200) -> List[str]:
    """
    Chunk by paragraph breaks when possible; fallback to hard split for huge paragraphs.
    """
    # policy_span = extract_policy_spans(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    cur = ""

    def push(c: str) -> None:
        c = c.strip()
        if c:
            chunks.append(c)

    for p in parts:
        if not cur:
            cur = p
        elif len(cur) + 2 + len(p) <= max_chars:
            cur += "\n\n" + p
        else:
            push(cur)
            tail = cur[-overlap_chars:] if overlap_chars > 0 else ""
            cur = (tail + "\n\n" + p).strip()

    push(cur)

    # Safety: hard-split any chunk that still exceeds max_chars
    final: List[str] = []
    for c in chunks:
        if len(c) <= max_chars:
            final.append(c)
        else:
            start = 0
            while start < len(c):
                end = min(start + max_chars, len(c))
                final.append(c[start:end].strip())
                if end >= len(c):
                    break
                start = max(end - overlap_chars, end)
    return [c for c in final if c]


# ----------------------------
# IO
# ----------------------------
def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

# ----------------------------
# ID detection
# ----------------------------

def extract_code_from_title(title: str | None) -> str | None:
    if not title:
        return None
    m = re.search(r"\b([A-Z]{2,8}-\d+(?:\.\d+)*)\b", title)
    return m.group(1) if m else None


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

## Add title to content
def compose_embed_text(sec: dict) -> str:
    title = sec.get("title", "").strip()
    path_text = sec.get("path_text", "").strip()
    content = sec.get("content", "").strip()
    return "\n".join([x for x in [title, path_text, "", content] if x != ""]).strip()

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
            content = clean_text(f"{current_section_title}\n\n{content}")
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

def extract_sections_from_pdf(pdf_path: Path) -> List[SectionRecord]:
    norm = normalize_pdf(str(pdf_path))
    doc_id = norm.doc_id

    sections: List[SectionRecord] = []
    section_index = 0

    for sec in norm.sections:
        content_parts = []

        body = clean_text(sec.text or "")
        if body:
            content_parts.append(body)

        # include tables (optional)
        if getattr(sec, "tables", None):
            for t in sec.tables:
                if t.raw_text:
                    caption = (t.caption or "").strip()
                    table_text = t.raw_text.strip()
                    if caption:
                        content_parts.append(f"{caption}\n{table_text}")
                    else:
                        content_parts.append(table_text)

        content = clean_text("\n\n".join([p for p in content_parts if p.strip()]))
        
        # only write if we have a heading path and some content
        if not sec.heading_path or not content:
            continue
        title = sec.heading_path[-1]
        path = sec.heading_path
        path_text = " > ".join(path)

        code = extract_code_from_title(title)
        id_line = f"ID: {code}" if code else ""
        retrieval_text = "\n".join([
            title,
            id_line,
            path_text,
            "",
            content,
        ]).strip()
        retrieval_text = f"{title}\nID: {id_line}\n{path_text}\n\n{content}"

        sections.append(
            SectionRecord(
                doc_id=doc_id,
                source_path=str(pdf_path),
                section_index=section_index,
                path=path,
                path_text=path_text,
                title=title,
                heading_level=len(path),   # proxy
                content=retrieval_text,           # clean body (for humans)      # 👈 what the retriever embeds
                extra={
                    "page_start": sec.page_start,
                    "page_end": sec.page_end,
                    "section_id": getattr(sec, "section_id", None),
                    "code": code,           # optional but very useful
                    "body":content
                },
            )
        )
        
        section_index += 1

    return sections


# ----------------------------
# Main preprocess
# ----------------------------
from pathlib import Path
import json
from dataclasses import asdict

NORMALIZED_BASE = Path("data/normalized")

RAW_DIR = Path("data/raw")

def main() -> None:
    args = parse_args()  # args.source in {"pdf","docx"}
    DEFAULT_OUT = NORMALIZED_BASE/ args.source / "sections.jsonl"
    if args.source == "docx":
        print(RAW_DIR)
        docx_files = find_docx_files(RAW_DIR)
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
        return

    # ----------------------------
    # PDF blocks-based corpora build
    # ----------------------------
    base_dir = NORMALIZED_BASE / "pdf"
    blocks_file = base_dir / "blocks.jsonl"
    sections_file = base_dir / "sections.jsonl"
    chunks_file = base_dir / "rag_chunks.jsonl"

    print("blocks:", blocks_file)
    print("sections:", sections_file)
    print("chunks:", chunks_file)

    if not blocks_file.exists():
        raise FileNotFoundError(
            f"Missing {blocks_file}. Run ingestion first to create blocks.jsonl."
        )

    base_dir.mkdir(parents=True, exist_ok=True)

    max_chars = 1400
    overlap_chars = 200
    dedupe = True

    seen: set[str] = set()
    total_blocks = 0
    total_sections = 0
    total_chunks = 0
    section_index = 0

    with open(sections_file, "w", encoding="utf-8") as sections_f, \
         open(chunks_file, "w", encoding="utf-8") as chunks_f:

        for block in read_jsonl(blocks_file):
            total_blocks += 1

            block_type = block.get("block_type", "unknown")
            section_path = block.get("section_path", []) or []
            doc_id = block.get("doc_id", "doc")
            block_index = block.get("block_index", 0)
            source_path = block.get("source_path")
            extra = dict(block.get("extra", {}) or {})
            text = block.get("text", "") or ""

            # 1) sections.jsonl (from section blocks)
            if block_type == "section":
                title = clean_text(section_path[-1]) if section_path else None
                body = clean_text(text)

                if title and body:
                    sec_rec = {
                        "doc_id": doc_id,
                        "source_path": source_path,
                        "section_index": section_index,
                        "path": section_path,
                        "path_text": " > ".join([clean_text(x) for x in section_path]) if section_path else title,
                        "title": title,
                        "heading_level": len(section_path) if section_path else None,
                        # KEY: keep it short + specific
                        "content": clean_text(f"{title}\n\n{body}"),
                        "extra": {**extra, "block_index": block_index},
                    }
                    sections_f.write(json.dumps(sec_rec, ensure_ascii=False) + "\n")
                    total_sections += 1
                    section_index += 1

            # 2) rag_chunks.jsonl
            if block_type == "table":
                caption = extra.get("caption") or extra.get("table_caption")
                raw_table = extra.get("raw_text") or extra.get("table_raw_text") or text

                if not looks_like_table(raw_table):
                    continue

                tier = table_tier(caption, raw_table)
                rows, cols = table_dims(raw_table)
                extra["table_tier"] = tier
                extra["table_rows"] = rows
                extra["table_cols"] = cols

                if tier != "A":
                    continue

                text = f"{caption}\n\n{raw_table}" if caption else raw_table

            if block_type == "section" and section_path:
                title = clean_text(section_path[-1])
                text = f"{title}\n\n{text}"

            text = clean_text(text)
            if not text:
                continue

            for i, ch in enumerate(chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)):
                item = {
                    "id": f"{doc_id}__b{block_index}__c{i}__{stable_hash(ch)}",
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "section_path": section_path,
                    "block_type": block_type,
                    "block_index": block_index,
                    "text": ch,
                    "extra": extra,
                }

                if dedupe:
                    key = (doc_id or "") + "|" + " > ".join(section_path or []) + "|" + (block_type or "") + "|" + ch
                    h = stable_hash(key)
                    if h in seen:
                        continue
                    seen.add(h)

                chunks_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[build_pdf] Read {total_blocks} blocks from {blocks_file}")
    print(f"[build_pdf] Wrote {total_sections} sections → {sections_file}")
    print(f"[build_pdf] Wrote {total_chunks} chunks → {chunks_file}")
    print(f"[build_pdf] Params: max_chars={max_chars}, overlap_chars={overlap_chars}, dedupe={dedupe}")


if __name__ == "__main__":
    main()
