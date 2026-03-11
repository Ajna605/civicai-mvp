# Creates rag_chunks.jsonl and sections.jsonl
# from blocks.jsonl
# ingestion/preprocess.py
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Iterable
from utils.hash_utils import short_hash
from utils.text_utils import clean_text
from utils. table_utils import build_table_summary_chunks

# ----------------------------
# Paths (repo-aware)
# ----------------------------
NORMALIZED_BASE = Path("data/normalized")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, choices=["pdf", "docx", "csv"])
    p.add_argument("--max_chars", type=int, default=1400)
    p.add_argument("--overlap_chars", type=int, default=200)
    p.add_argument("--no_dedupe", action="store_true", help="Disable deduplication")
    return p.parse_args()

# ----------------------------
# Chunking
# ----------------------------

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
# Iterate blocks in doc order
# ----------------------------
def block_to_text(block: Dict[str, Any]) -> str:
    t = block.get("type")
    if t in ("text", "paragraph"):
        return block.get("text", "") or ""
    if t == "list":
        items = block.get("items", []) or []
        lines = []
        for it in items:
            txt = (it.get("text") or "").strip()
            if txt:
                lines.append(f"- {txt}")
        return "\n".join(lines)
    return ""


def build_chunks_for_section(
    sec: Dict[str, Any],
    *,
    source: str,
    max_chars: int,
    overlap_chars: int,
    dedupe: bool,
    seen: set[str],
    chunks_f
) -> int:
    """Returns number of chunks written for this section."""
    doc_id = sec.get("doc_id", "doc")
    source_path = sec.get("source_path")
    section_index = sec.get("section_index", 0)
    section_path = sec.get("path", []) or []
    title = clean_text(sec.get("title") or (section_path[-1] if section_path else ""))

    blocks: List[Dict[str, Any]] = sec.get("blocks", []) or []

    written = 0
    buf: List[str] = []

    def flush_text(block_ordinal: int) -> int:
        nonlocal buf, written
        text = clean_text("\n\n".join(x for x in buf if x).strip())
        buf = []
        if not text:
            return 0

        if title:
            text = clean_text(f"{title}\n\n{text}")

        n = 0
        for i, ch in enumerate(chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)):
            ch = clean_text(ch)
            if not ch:
                continue

            item = {
                "id": f"{doc_id}__s{section_index}__b{block_ordinal}__c{i}__{short_hash(ch)}",
                "doc_id": doc_id,
                "source_path": source_path,
                "section_path": section_path,
                "section_index": section_index,
                "block_type": "section_text",
                "block_index": block_ordinal,
                "text": ch,
                "extra": {"source": source},
            }

            if dedupe:
                key = f"{doc_id}|{' > '.join(section_path)}|section_text|{ch}"
                h = short_hash(key)
                if h in seen:
                    continue
                seen.add(h)

            chunks_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
        written += n
        return n

    for block_ordinal, blk in enumerate(blocks):
        btype = blk.get("type")

        # ---- NARRATIVE: accumulate and chunk as section text ----
        txt = block_to_text(blk)
        if txt and txt.strip():
            buf.append(txt)

    flush_text(block_ordinal=len(blocks))
    return written

def main() -> None:
    args = parse_args()
    base_dir = NORMALIZED_BASE / args.source
    sections_file = base_dir / "sections.jsonl"
    chunks_file = base_dir / "rag_chunks.jsonl"
    tables_file = NORMALIZED_BASE / "doc_tables" / "raw_tables.jsonl"


    if not sections_file.exists():
        raise FileNotFoundError(f"Missing {sections_file}. Run ingest_documents first.")

    base_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    total_sections = 0
    total_chunks = 0
    total_table_chunks = 0

    with open(chunks_file, "w", encoding="utf-8") as chunks_f:
        for sec in read_jsonl(sections_file):
            total_sections += 1
            total_chunks += build_chunks_for_section(
                sec,
                source=args.source,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                dedupe=not args.no_dedupe,
                seen=seen,
                chunks_f=chunks_f,
            )
        print(tables_file.exists())
        if tables_file.exists():
            table_records = list(read_jsonl(tables_file))
            table_summary_chunks = build_table_summary_chunks(
                table_records=table_records,
                source=args.source,
            )

            for chunk in table_summary_chunks:
                if not args.no_dedupe:
                    key = f"{chunk['doc_id']}|{' > '.join(chunk.get('section_path', []))}|table_summary|{chunk['text']}"
                    h = short_hash(key)
                    if h in seen:
                        continue
                    seen.add(h)

                chunks_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                total_table_chunks += 1

    print(f"[build_corpus] Source={args.source}")
    print(f"[build_corpus] Read {total_sections} sections from {sections_file}")
    print(f"[build_corpus] Wrote {total_chunks + total_table_chunks} chunks → {chunks_file}")
    print(f"[build_corpus] Included {total_table_chunks} table summary chunks from {tables_file}")
    print(f"[build_corpus] Params: max_chars={args.max_chars}, overlap_chars={args.overlap_chars}, "
        f"no_dedupe={args.no_dedupe}"
    )
if __name__ == "__main__":
    main()
