# ingestion/load_docs.py
import argparse
import json
import re
from dataclasses import asdict
from typing import Any, Dict, Iterator, List, Optional, Tuple, Set
from pathlib import Path
from ingestion.normalize.normalize_pdf import normalize_pdf
from ingestion.schema.document_schema import NormalizedDoc, SectionRecord
from ingestion.loaders.document_tables import extract_tables_for_source
from ingestion.normalize.process_tables import normalize_extracted_tables, normalize_all_table_rows

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
    p.add_argument("--source", choices=["pdf", "docx", "csv"], required=True)
    p.add_argument("--input", help="Path to the document", required= False)
    p.add_argument("--out-blocks", default=None, help="sections.jsonl output path")
    p.add_argument(
        "--vision",
        action="store_true",
        help=(
            "Extract and interpret images using the OpenAI Vision API. "
            "Requires OPENAI_API_KEY to be set. Supported for --source pdf and docx."
        ),
    )
    return p.parse_args()

def default_out_for(source: str) -> Path:
    return NORMALIZED_DIR / source / "sections.jsonl"




# ----------------------------
# File type functions
# ----------------------------

SOURCE_EXTENSIONS = {
    "pdf": [".pdf"],
    "docx": [".docx"]
}

SOURCE_EXTRACTORS = {
    "pdf": lambda p: extract_sections_from_pdf(p),
    "docx": lambda p: extract_sections_from_docx(p),
}

def find_files_for_source(source: str) -> List[Path]:
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Expected raw data folder at: {RAW_DIR}")

    exts = SOURCE_EXTENSIONS.get(source)
    if not exts:
        raise ValueError(f"Unknown source type: {source}")

    files = []
    for ext in exts:
        files.extend(RAW_DIR.rglob(f"*{ext}"))

    return sorted(files)

def _validate_input_matches_source(source: str, path: Path) -> None:
    allowed = SOURCE_EXTENSIONS[source]
    if path.suffix.lower() not in allowed:
        raise ValueError(f"--source {source} requires {', '.join(allowed)} input, got: {path}")

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


def slugify(text: str, max_len: int = 40) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:max_len]

## Create a deterministic table_id across formats.
def stable_table_id(
    *,
    doc_id: str,
    section_path: List[str],
    table_ordinal: int,
    source_type: str,
) -> str:
    section_name = section_path[-1] if section_path else "root"
    slug = slugify(section_name)
    return f"{doc_id}::{source_type}::{slug}::tbl::{table_ordinal}"


# ----------------------------
# Extraction (CONDENSED sections w/ sections)
# ----------------------------



def normalized_doc_to_records(norm_doc: NormalizedDoc, source_path: str) -> List[SectionRecord]:
    records: List[SectionRecord] = []
    seen_table_ids: set[str] = set()

    for i, sec in enumerate(norm_doc.sections):
        blocks: List[Dict[str, Any]] = []

        # Text block
        if sec.text:
            blocks.append({
                "type": "text",
                "text": sec.text,
                "page_start": sec.page_start,
                "page_end": sec.page_end,
                "section_id": sec.section_id,
            })

        # Table blocks
        for tbl in sec.tables or []:
            if tbl.table_id in seen_table_ids:
                continue
            seen_table_ids.add(tbl.table_id)

            blocks.append({
                "type": "table",
                "table_id": tbl.table_id,
                "caption": tbl.caption,
                "raw_text": tbl.raw_text,
                "page": tbl.page,  # -1 for csv/xlsx, real page for pdf
            })

        path = sec.heading_path or []
        records.append(
            SectionRecord(
                doc_id=norm_doc.doc_id,
                source_path=source_path,
                section_index=i,
                path=path,
                path_text=" > ".join(path),
                title=path[-1] if path else "",
                heading_level=len(path) if path else None,
                blocks=blocks,
            )
        )

    return records


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
            continue

    flush_section()
    return sections

def extract_sections_from_pdf(pdf_path: Path) -> List[SectionRecord]:
    norm_doc = normalize_pdf(str(pdf_path))
    return normalized_doc_to_records(norm_doc, source_path=str(pdf_path))

def main() -> None:
    args = parse_args()
    out_path = default_out_for(args.source)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ## Paths
    table_out_path = NORMALIZED_DIR /args.source / "doc_tables" / "raw_tables.jsonl"
    row_out_path = NORMALIZED_DIR / args.source /"doc_tables" / "table_rows.jsonl"
    table_out_path.parent.mkdir(parents=True, exist_ok=True)
    row_out_path.parent.mkdir(parents=True, exist_ok=True)

    # Choose inputs
    if args.input:
        files = [Path(args.input)]
        _validate_input_matches_source(args.source, files[0])
    else:
        files = find_files_for_source(args.source)

    if not files:
        raise FileNotFoundError(
            f"No {args.source.upper()} files found (input={args.input!r})."
        )

    extractor = SOURCE_EXTRACTORS[args.source]

    total_blocks = 0
    total_table_facts = 0
    total_rows = 0
    with open(out_path, "w", encoding="utf-8") as doc_f, \
        open(table_out_path, "w", encoding="utf-8") as table_f, \
        open(row_out_path, "w", encoding="utf-8") as row_f:
        for path in files:
            blocks = extractor(path)
            #Table detection
            raw_tables = extract_tables_for_source(path, args.source)
            table_outputs = normalize_extracted_tables(raw_tables=raw_tables, path=path, source_type=args.source)
            row_outputs = normalize_all_table_rows(table_outputs)

            for b in blocks:
                doc_f.write(json.dumps(asdict(b), ensure_ascii=False) + "\n")
            # NEW: write table summary chunks into same doc index
            for tb in table_outputs:
                table_f.write(json.dumps(tb, ensure_ascii=False) + "\n")
            
            for row in row_outputs:
                row_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_blocks += len(blocks)
            total_table_facts += len(table_outputs)
            total_rows += len(row_outputs)

    print(f"[ingest_documents] Source={args.source} Files={len(files)}")
    print(f"[ingest_documents] Wrote {total_blocks} sections → {out_path}")
    print(f"[ingest_documents] Wrote {total_table_facts} table facts → {table_out_path}")
    print(f"[ingest_documents] Wrote {total_rows} table facts → {row_out_path}")

    # Optional: extract and interpret images from PDF and DOCX files
    if args.vision:
        if args.source not in ("pdf", "docx"):
            print(
                f"[ingest_documents] --vision is not supported for --source {args.source}; skipping."
            )
        else:
            from vision.pipeline import run_vision_pipeline_to_jsonl

            img_out_path = NORMALIZED_DIR / args.source / "image_descriptions.jsonl"
            total_images = 0
            for i, path in enumerate(files):
                n = run_vision_pipeline_to_jsonl(
                    str(path), str(img_out_path), append=(i > 0)
                )
                total_images += n
            print(
                f"[ingest_documents] Wrote {total_images} image description chunks"
                f" → {img_out_path}"
            )

if __name__ == "__main__":
    main()
