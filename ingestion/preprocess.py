# ingestion/preprocess.py
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


# ----------------------------
# Paths (repo-aware)
# ----------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SECTIONS_FILE = PROCESSED_DIR / "sections.jsonl"       # produced by ingestion/load_docs.py
OUT_FILE = PROCESSED_DIR / "rag_chunks.jsonl"          # final for embedding
REPORT_FILE = PROCESSED_DIR / "ingestion_report.json"  # stats report


# ----------------------------
# Cleaning steps (tracked)
# ----------------------------
CLEANING_STEPS = [
    "replace_non_breaking_space",
    "normalize_newlines",
    "collapse_spaces_tabs",
    "collapse_excess_blank_lines",
    "strip_edges",
]

# Rough token estimate without extra deps
USE_SIMPLE_TOKEN_ESTIMATE = True


# ----------------------------
# Chunking parameters
# ----------------------------
MAX_CHARS = 1600
OVERLAP_CHARS = 200
DEDUPE = True

# Handle rare huge sections better
GIANT_SECTION_CHAR_THRESHOLD = 4000
GIANT_MAX_CHARS = 900
GIANT_OVERLAP_CHARS = 120


# ----------------------------
# Cleanup / hashing
# ----------------------------
def clean_text(s: str) -> str:
    # Keep in sync with CLEANING_STEPS
    s = s.replace("\u00a0", " ")          # replace_non_breaking_space
    s = s.replace("\r\n", "\n")           # normalize_newlines
    s = re.sub(r"[ \t]+", " ", s)         # collapse_spaces_tabs
    s = re.sub(r"\n{3,}", "\n\n", s)      # collapse_excess_blank_lines
    return s.strip()                      # strip_edges


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def estimate_tokens(text: str) -> int:
    """
    Lightweight estimate: 1 token ~ 4 chars (rough).
    Good enough for monitoring distributions.
    """
    text = text.strip()
    if not text:
        return 0
    if USE_SIMPLE_TOKEN_ESTIMATE:
        return max(1, len(text) // 4)
    return 0


def percentile(sorted_vals: List[int], p: float) -> int:
    """
    p in [0, 100]. Uses nearest-rank.
    """
    if not sorted_vals:
        return 0
    if p <= 0:
        return sorted_vals[0]
    if p >= 100:
        return sorted_vals[-1]
    k = int(round((p / 100) * (len(sorted_vals) - 1)))
    return sorted_vals[k]


# ----------------------------
# Chunking
# ----------------------------
def chunk_text(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Chunk by paragraph breaks; fallback to hard split for huge paragraphs/tables.
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

    # Safety: hard split any chunk still > max_chars
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


def write_report(report: Dict[str, Any], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


# ----------------------------
# Main preprocess + report
# ----------------------------
def main() -> None:
    if not SECTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {SECTIONS_FILE}. Run ingestion/load_docs.py first."
        )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()

    # Stats accumulators
    total_sections = 0
    kept_sections = 0
    total_chunks = 0

    section_char_counts: List[int] = []
    section_token_counts: List[int] = []
    chunk_char_counts: List[int] = []
    chunk_token_counts: List[int] = []

    per_doc_sections = Counter()
    per_doc_chunks = Counter()

    # To help debug giant outliers
    # store (char_len, doc_id, path_text, title)
    largest_sections: List[Tuple[int, str, str, str]] = []

    with open(OUT_FILE, "w", encoding="utf-8") as out_f:
        for sec in read_jsonl(SECTIONS_FILE):
            total_sections += 1

            doc_id = sec.get("doc_id", "doc")
            source_path = sec.get("source_path")
            section_index = sec.get("section_index", 0)

            title = clean_text(sec.get("title", ""))
            path_text = clean_text(sec.get("path_text", title))
            content = clean_text(sec.get("content", ""))

            if not content:
                continue

            kept_sections += 1
            per_doc_sections[doc_id] += 1

            # Breadcrumb once (helps orientation without repeating per line)
            full_text = clean_text(f"{path_text}\n\n{content}")

            # Section-level stats
            sec_chars = len(full_text)
            section_char_counts.append(sec_chars)
            section_token_counts.append(estimate_tokens(full_text))
            largest_sections.append((sec_chars, doc_id, path_text, title))

            # Adaptive chunking for huge sections
            if sec_chars > GIANT_SECTION_CHAR_THRESHOLD:
                chunks = chunk_text(full_text, max_chars=GIANT_MAX_CHARS, overlap_chars=GIANT_OVERLAP_CHARS)
            else:
                chunks = chunk_text(full_text, max_chars=MAX_CHARS, overlap_chars=OVERLAP_CHARS)

            for i, ch in enumerate(chunks):
                item = {
                    "id": f"{doc_id}__sec{section_index}__c{i}__{stable_hash(ch)}",
                    "doc_id": doc_id,
                    "source_path": source_path,
                    "section_index": section_index,
                    "title": title,
                    "path_text": path_text,
                    "text": ch,
                    "meta": {
                        "is_giant_section": sec_chars > GIANT_SECTION_CHAR_THRESHOLD,
                        "section_chars": sec_chars,
                        "chunk_chars": len(ch),
                        "chunk_tokens_est": estimate_tokens(ch),
                    },
                }

                if DEDUPE:
                    h = stable_hash((doc_id or "") + "|" + (path_text or "") + "|" + ch)
                    if h in seen:
                        continue
                    seen.add(h)

                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_chunks += 1
                per_doc_chunks[doc_id] += 1

                chunk_char_counts.append(len(ch))
                chunk_token_counts.append(estimate_tokens(ch))

    # Sort for percentiles and top-N
    section_char_counts_sorted = sorted(section_char_counts)
    section_token_counts_sorted = sorted(section_token_counts)
    chunk_char_counts_sorted = sorted(chunk_char_counts)
    chunk_token_counts_sorted = sorted(chunk_token_counts)

    largest_sections.sort(key=lambda x: x[0], reverse=True)
    top10 = largest_sections[:10]

    report: Dict[str, Any] = {
        "inputs": {
            "sections_file": str(SECTIONS_FILE),
        },
        "outputs": {
            "chunks_file": str(OUT_FILE),
            "report_file": str(REPORT_FILE),
        },
        "preprocessing_stats": {
            "number_of_sections_total": total_sections,
            "number_of_sections_kept": kept_sections,
            "number_of_chunks_written": total_chunks,
            "dedupe_enabled": DEDUPE,
            "chunking": {
                "default": {"max_chars": MAX_CHARS, "overlap_chars": OVERLAP_CHARS},
                "giant_section_rule": {
                    "threshold_chars": GIANT_SECTION_CHAR_THRESHOLD,
                    "max_chars": GIANT_MAX_CHARS,
                    "overlap_chars": GIANT_OVERLAP_CHARS,
                },
            },
        },
        "character_counts": {
            "sections": {
                "total": sum(section_char_counts),
                "mean": (sum(section_char_counts) / len(section_char_counts)) if section_char_counts else 0,
                "p50": percentile(section_char_counts_sorted, 50),
                "p90": percentile(section_char_counts_sorted, 90),
                "p95": percentile(section_char_counts_sorted, 95),
                "max": section_char_counts_sorted[-1] if section_char_counts_sorted else 0,
            },
            "chunks": {
                "total": sum(chunk_char_counts),
                "mean": (sum(chunk_char_counts) / len(chunk_char_counts)) if chunk_char_counts else 0,
                "p50": percentile(chunk_char_counts_sorted, 50),
                "p90": percentile(chunk_char_counts_sorted, 90),
                "p95": percentile(chunk_char_counts_sorted, 95),
                "max": chunk_char_counts_sorted[-1] if chunk_char_counts_sorted else 0,
            },
        },
        "token_counts_estimate": {
            "note": "Token counts are rough estimates (chars//4). Use tiktoken for exact counts per model.",
            "sections": {
                "total": sum(section_token_counts),
                "mean": (sum(section_token_counts) / len(section_token_counts)) if section_token_counts else 0,
                "p50": percentile(section_token_counts_sorted, 50),
                "p90": percentile(section_token_counts_sorted, 90),
                "p95": percentile(section_token_counts_sorted, 95),
                "max": section_token_counts_sorted[-1] if section_token_counts_sorted else 0,
            },
            "chunks": {
                "total": sum(chunk_token_counts),
                "mean": (sum(chunk_token_counts) / len(chunk_token_counts)) if chunk_token_counts else 0,
                "p50": percentile(chunk_token_counts_sorted, 50),
                "p90": percentile(chunk_token_counts_sorted, 90),
                "p95": percentile(chunk_token_counts_sorted, 95),
                "max": chunk_token_counts_sorted[-1] if chunk_token_counts_sorted else 0,
            },
        },
        "cleaning_steps_applied": CLEANING_STEPS,
        "largest_sections_top10": [
            {"chars": c, "doc_id": d, "path_text": p, "title": t}
            for c, d, p, t in top10
        ],
        "by_doc": {
            "sections_kept": dict(per_doc_sections),
            "chunks_written": dict(per_doc_chunks),
        },
    }

    write_report(report, REPORT_FILE)

    print(f"[preprocess] Read {total_sections} sections from {SECTIONS_FILE}")
    print(f"[preprocess] Kept {kept_sections} sections (non-empty content)")
    print(f"[preprocess] Wrote {total_chunks} chunks → {OUT_FILE}")
    print(f"[preprocess] Wrote report → {REPORT_FILE}")


if __name__ == "__main__":
    main()
