# Clean text, preserve headings, attach metadata,
# output standardized markdown to data/processed
# ingestion/preprocess.py
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ----------------------------
# Paths (repo-aware)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "normalized"
# BLOCKS_FILE = PROCESSED_DIR / "blocks.jsonl"


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

    # corpus parameters (optional but useful)
    p.add_argument("--max-chars", type=int, default=1400)
    p.add_argument("--overlap-chars", type=int, default=200)
    p.add_argument("--dedupe", action="store_true", default=True)
    p.add_argument("--no-dedupe", action="store_false", dest="dedupe")

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


# ------------------------------------
# Chunking + Logic for splitting by policy
# ------------------------------------
ITEM_RE = re.compile(
    r"(?mi)^\s*(?:[•\-\u2022]\s*)?(?:policy|objective|goal|program)?\s*"
    r"([A-Z]{2,8}-\d+(?:\.\d+)+)\.?\s*(.*)$"
)
def split_into_policy_items(text: str):
    """
    Split a section into (code, item_text) chunks.
    """
    matches = list(ITEM_RE.finditer(text))
    out = []

    for i, m in enumerate(matches):
        code = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        item_text = text[start:end].strip()
        item_text = f"{code}\n\n{item_text}"
        out.append((code, item_text))

    return out


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
# Main preprocess
# ----------------------------
def main() -> None:
    args = parse_args()
    BLOCKS_FILE, OUT_FILE = resolve_paths(args)
    print(BLOCKS_FILE, OUT_FILE)
    if not BLOCKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BLOCKS_FILE}. Run ingestion/load_docs.py first to create blocks.jsonl."
        )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    max_chars = 1400
    overlap_chars = 200
    dedupe = True

    seen: set[str] = set()
    total_blocks = 0
    total_chunks = 0

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for block in read_jsonl(BLOCKS_FILE):
            total_blocks += 1

            block_type = block.get("block_type", "unknown")
            section_path = block.get("section_path", [])
            doc_id = block.get("doc_id", "doc")
            block_index = block.get("block_index", 0)
            source_path = block.get("source_path")

            block_type = block.get("block_type", "unknown")
            # IMPORTANT: use a mutable copy so we can add fields
            extra = dict(block.get("extra", {}) or {})

            # default text
            text = block.get("text", "")

            # If this is a table block, prepend caption and compute table metadata
            if block_type == "table":
                caption = extra.get("caption") or extra.get("table_caption")
                raw_table = extra.get("raw_text") or extra.get("table_raw_text") or text

                # NEW: guard against bullet-lists pretending to be tables
                if not looks_like_table(raw_table):
                    continue

                tier = table_tier(caption, raw_table)
                rows, cols = table_dims(raw_table)

                extra["table_tier"] = tier
                extra["table_rows"] = rows
                extra["table_cols"] = cols

                # optional: only index Tier A tables (recommended)
                if tier != "A":
                    continue

                # make sure caption is attached to content for retrieval
                if caption:
                    text = f"{caption}\n\n{raw_table}"
                else:
                    text = raw_table

            if block_type == "section":
                for code, item_text in split_into_policy_items(text):
                    chunk_obj = {
                        "doc_id": doc_id,
                        "block_type": "policy",
                        "block_index": block_index,
                        "section_path": section_path,
                        "source_path": source_path,
                        "text": item_text,
                        "extra": {
                            **extra,
                            "chunk_kind": "policy_item",
                            "policy_code": code,
                        },
                    }

                    out_f.write(json.dumps(chunk_obj, ensure_ascii=False) + "\n")
                    total_chunks += 1

            text = clean_text(text)

            if not text:
                continue

            # Optional: skip extremely short headings
            if block_type == "heading" and len(text) < 3:
                continue

            
            chunks = chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)
            for i, ch in enumerate(chunks):
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
                    # Dedupe by section+type+text (good default for repeated headers/footers)
                    key = (
                        (doc_id or "")
                        + "|"
                        + " > ".join(section_path or [])
                        + "|"
                        + (block_type or "")
                        + "|"
                        + ch
                    )
                    h = stable_hash(key)
                    if h in seen:
                        continue
                    seen.add(h)

                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"[preprocess] Read {total_blocks} blocks from {BLOCKS_FILE}")
    print(f"[preprocess] Wrote {total_chunks} chunks → {OUT_FILE}")
    print(f"[preprocess] Params: max_chars={max_chars}, overlap_chars={overlap_chars}, dedupe={dedupe}")


if __name__ == "__main__":
    main()
