# Clean text, preserve headings, attach metadata,
# output standardized markdown to data/processed
# ingestion/preprocess.py
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List


# ----------------------------
# Paths (repo-aware)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
BLOCKS_FILE = PROCESSED_DIR / "blocks.jsonl"
OUT_FILE = PROCESSED_DIR / "rag_chunks.jsonl"


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


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 200) -> List[str]:
    """
    Chunk by paragraph breaks when possible; fallback to hard split for huge paragraphs.
    """
    text = clean_text(text)
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
    if not BLOCKS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {BLOCKS_FILE}. Run ingestion/load_docs.py first to create blocks.jsonl."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

            text = clean_text(block.get("text", ""))
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
                    "extra": block.get("extra", {}),
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
