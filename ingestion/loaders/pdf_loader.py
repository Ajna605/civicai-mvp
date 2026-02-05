from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import Counter
from typing import Tuple
from dataclasses import dataclass
from typing import Optional


# ---------- output structure ----------
@dataclass
class SectionText:
    section_id: str
    heading: Optional[str]
    heading_path: List[str]   # NEW
    page_start: int
    page_end: int
    text: str

@dataclass
class PageText:
    page: int
    text: str         # cleaned text for sectioning
    raw_text: str     # NEW: raw text for table detection


# ---------- check type of pdf - scanned or text ----------

def pdf_has_text_layer(pdf_path: str, sample_pages: int = 3) -> bool:
    """
    Returns True if the PDF likely has a selectable text layer.
    Checks a few early pages for any text spans.
    """
    try:
        import fitz
    except Exception as e:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf") from e

    doc = fitz.open(pdf_path)
    n = min(sample_pages, doc.page_count)

    for i in range(n):
        page = doc.load_page(i)
        d = page.get_text("dict") or {}
        blocks = d.get("blocks", [])
        # type==0 means text block in PyMuPDF dict output
        for b in blocks:
            if b.get("type") == 0:
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        if (span.get("text") or "").strip():
                            return True
    return False

# ---------- geometry helpers ----------

def _median(xs: List[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 == 1 else 0.5 * (xs[mid - 1] + xs[mid])

def _cluster_rows(words: List[Tuple[float, float, float, float, str, int, int, int]], y_tol: float = 2.5):
    """
    Cluster words into rows by y0 coordinate.
    Returns list of rows; each row is list of word tuples.
    """
    if not words:
        return []

    # sort by y then x
    words_sorted = sorted(words, key=lambda w: (w[1], w[0]))
    rows: List[List[Tuple[float, float, float, float, str, int, int, int]]] = []

    for w in words_sorted:
        x0, y0, x1, y1, text, *_ = w
        placed = False
        for row in rows:
            # compare to row baseline (median y0)
            y0s = [rw[1] for rw in row]
            base = _median(y0s)
            if abs(y0 - base) <= y_tol:
                row.append(w)
                placed = True
                break
        if not placed:
            rows.append([w])

    # sort words within each row by x
    for row in rows:
        row.sort(key=lambda w: w[0])
    return rows

def _row_to_cells(row_words, x_gap: float = 10.0) -> List[str]:
    """
    Convert row words into "cells" by splitting on large x gaps.
    """
    if not row_words:
        return []
    cells: List[List[str]] = []
    current: List[str] = [row_words[0][4]]
    last_x1 = row_words[0][2]

    for w in row_words[1:]:
        x0, y0, x1, y1, text, *_ = w
        if (x0 - last_x1) >= x_gap:
            cells.append(current)
            current = [text]
        else:
            current.append(text)
        last_x1 = x1

    cells.append(current)
    # join words within each cell
    return [" ".join(c).strip() for c in cells if " ".join(c).strip()]


def _is_table_row(cells: List[str]) -> bool:
    """
    Table rows typically have multiple cells and are not full sentences.
    """
    if len(cells) < 2:
        return False
    # avoid rows that are just one long paragraph split oddly
    long_cells = sum(1 for c in cells if len(c) > 80)
    if long_cells >= 2:
        return False
    return True

def _find_consecutive_runs(flags: List[bool], min_len: int) -> List[Tuple[int, int]]:
    """
    Return list of (start, end_exclusive) runs of True values.
    """
    runs = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if (not f or i == len(flags) - 1) and start is not None:
            end = i + 1 if f and i == len(flags) - 1 else i
            if (end - start) >= min_len:
                runs.append((start, end))
            start = None
    return runs

def _extract_caption_from_lines(lines: List[str], table_start_line: int, lookback: int = 4) -> Optional[str]:
    """
    Best-effort caption from nearby text lines.
    """
    for j in range(max(0, table_start_line - lookback), table_start_line)[::-1]:
        s = (lines[j] or "").strip()
        if not s:
            continue
        if "Table" in s or "TABLE" in s:
            return s
        # also accept short title-like lines
        if len(s) <= 80 and not s.endswith("."):
            return s
    return None

def _run_quality(a: int, b: int, row_cells_text, row_cells_x) -> bool:
    run_len = b - a
    if run_len < 3:
        return False

    # rows that actually have >=2 cells
    multi = [i for i in range(a, b) if len(row_cells_text[i]) >= 2]
    if len(multi) < max(3, int(0.6 * run_len)):
        return False  # too many single-cell lines → likely prose

    # Column count consistency (mode frequency)
    col_counts = [len(row_cells_text[i]) for i in multi]
    mode = max(set(col_counts), key=col_counts.count)
    mode_frac = col_counts.count(mode) / float(len(col_counts))
    if mode_frac < 0.6:
        return False

    # Column x alignment consistency: compare first 2-3 cell starts across rows
    # (roughly: are columns in the same places?)
    def quantize(xs, q=8.0):
        return [int(x / q) for x in xs]

    sigs = []
    for i in multi:
        xs = row_cells_x[i][:min(3, len(row_cells_x[i]))]
        sigs.append(tuple(quantize(xs)))

    sig_mode = max(set(sigs), key=sigs.count)
    sig_frac = sigs.count(sig_mode) / float(len(sigs))
    if sig_frac < 0.55:
        return False

    return True


def extract_tables_from_page_layout(page) -> List[dict]:
    """
    Layout-aware table extraction using PyMuPDF word positions.
    Returns list of dicts: {page, caption, raw_text}
    raw_text is a pipe-separated representation with stable rows.
    """
    page_num = page.number + 1

    words = page.get_text("words") or []
    if not words:
        return []

    rows = _cluster_rows(words, y_tol=2.5)

    row_cells_x: List[List[float]] = []
    row_cells_text: List[List[str]] = []
    for r in rows:
        cx = _row_to_cells_with_x(r, x_gap=7.0)
        row_cells_x.append([x for x, _ in cx])
        row_cells_text.append([t for _, t in cx])

    # Debug summary (optional)
    ge2 = sum(1 for c in row_cells_text if len(c) >= 2)
    ge3 = sum(1 for c in row_cells_text if len(c) >= 3)
    if ge2 >= 10:
        print(f"[debug page {page_num}] total={len(row_cells_text)} ge2={ge2} ge3={ge3}")

    # Table-ish rows: >=2 cells
    flags = [len(cells) >= 2 for cells in row_cells_text]
    runs = _find_consecutive_runs(flags, min_len=3)

    page_text_lines = (page.get_text('text') or "").splitlines()

    tables: List[dict] = []
    for (a, b) in runs:
        if not _run_is_table_like(row_cells_x, row_cells_text, a, b):
            continue

        caption = _extract_caption_from_lines(page_text_lines, table_start_line=a, lookback=6)

        # Tightener: require caption unless the run is "long enough"
        # If no caption, require a longer run AND more columns
        if caption is None:
            if (b - a) < 10:
                continue
            # also require at least 3 columns in the mode row
            multi = [i for i in range(a, b) if len(row_cells_text[i]) >= 2]
            counts = [len(row_cells_text[i]) for i in multi]
            mode = max(set(counts), key=counts.count)
            if mode < 3:
                continue

        max_cols = max(len(row_cells_text[i]) for i in range(a, b))
        out_lines = []
        for i in range(a, b):
            cells = row_cells_text[i]
            padded = cells + [""] * (max_cols - len(cells))
            out_lines.append(" | ".join(padded))

        raw_text = "\n".join(out_lines).strip()
        if raw_text:
            tables.append({"page": page_num, "caption": caption, "raw_text": raw_text})

    return tables


def _row_to_cells_with_x(row_words, x_gap: float = 7.0):
    """
    Returns list of (x_start, cell_text) for a row by splitting on x gaps.
    word tuple: (x0, y0, x1, y1, text, block, line, wordno)
    """
    if not row_words:
        return []
    row_words = sorted(row_words, key=lambda w: w[0])

    cells = []
    cur_x0 = row_words[0][0]
    cur_words = [row_words[0][4]]
    last_x1 = row_words[0][2]

    for w in row_words[1:]:
        x0, y0, x1, y1, text, *_ = w
        if (x0 - last_x1) >= x_gap:
            cell_text = " ".join(cur_words).strip()
            if cell_text:
                cells.append((cur_x0, cell_text))
            cur_x0 = x0
            cur_words = [text]
        else:
            cur_words.append(text)
        last_x1 = x1

    cell_text = " ".join(cur_words).strip()
    if cell_text:
        cells.append((cur_x0, cell_text))

    return cells


def _run_is_table_like(row_cells_x, row_cells_text, a: int, b: int) -> bool:
    """
    Filters out false positives by enforcing:
    - many multi-cell rows
    - stable column count
    - stable column x-start signature
    """
    run_len = b - a
    if run_len < 3:
        return False

    # consider only rows with >=2 cells (multi-column)
    multi = [i for i in range(a, b) if len(row_cells_text[i]) >= 2]
    if len(multi) < max(3, int(0.85 * run_len)):  # must mostly be multi-cell
        return False

    # column count consistency
    counts = [len(row_cells_text[i]) for i in multi]
    mode = max(set(counts), key=counts.count)
    mode_frac = counts.count(mode) / float(len(counts))
    if mode_frac < 0.85:
        return False

    # column x-start signature consistency (first 2-3 cols)
    def sig(xs, q=8.0, m=3):
        xs = xs[: min(m, len(xs))]
        return tuple(int(x / q) for x in xs)

    sigs = [sig(row_cells_x[i]) for i in multi]
    sig_mode = max(set(sigs), key=sigs.count)
    sig_frac = sigs.count(sig_mode) / float(len(sigs))
    if sig_frac < 0.8:
        return False

    return True


# -----------------------------------------------------------------------
# Legacy text-stream table detection (not used for PDF layout extraction)
def looks_like_table_line(line: str) -> bool:
    """
    Heuristic: table-ish lines often have multiple columns separated by pipes
    or by 2+ spaces repeated across the line.
    """
    s = (line or "").rstrip()
    if not s:
        return False

    # Markdown/pipe tables or PDF-to-text with pipes
    if "|" in s:
        # ignore lines that are just a single pipe
        return s.count("|") >= 2

    # Multiple columns separated by 2+ spaces
    # e.g. "Urban Open Spaces   0 - .25   5 min walk"
    if re.search(r"\S+\s{2,}\S+", s):
        return True

    return False


def looks_like_table_caption(line: str) -> bool:
    """
    Captions often include 'Table' or end with 'Standards', 'Standard', etc.
    Keep it conservative.
    """
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 120:
        return False

    if re.search(r"\bTable\b", s, flags=re.I):
        return True
    if re.search(r"\b(Standards?|Radius|Service)\b", s, flags=re.I) and not s.endswith("."):
        return True

    return False


def extract_table_candidates_from_page(page_text: str, page_num: int, min_lines: int = 3) -> List[dict]:
    """
    Returns a list of candidate tables on a page as dicts:
      {caption, raw_text, page}
    We don't parse rows/cols yet; we just find contiguous table-ish blocks.
    """
    lines = [ln.rstrip() for ln in (page_text or "").splitlines()]
    candidates: List[dict] = []

    buf: List[str] = []
    for i, ln in enumerate(lines):
        if looks_like_table_line(ln):
            buf.append(ln)
        else:
            if len(buf) >= min_lines:
                # look backward for a caption within the last ~3 lines
                caption = None
                for back in range(1, 4):
                    j = i - back
                    if j >= 0 and looks_like_table_caption(lines[j]):
                        caption = lines[j].strip()
                        break
                candidates.append(
                    {"page": page_num, "caption": caption, "raw_text": "\n".join(buf)}
                )
            buf = []

    # flush
    if len(buf) >= min_lines:
        caption = None
        for back in range(1, 4):
            j = len(lines) - back - len(buf)
            if 0 <= j < len(lines) and looks_like_table_caption(lines[j]):
                caption = lines[j].strip()
                break
        candidates.append({"page": page_num, "caption": caption, "raw_text": "\n".join(buf)})

    return candidates

# -----------------------------------------------------------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_text_basic(text: str) -> str:
    """
    Minimal text cleanup for page-level extraction.
    Keeps line breaks but normalizes whitespace noise.
    """
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    # normalize spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # normalize excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines()]


def estimate_repeated_headers_footers(
    pages_lines: List[List[str]],
    top_n: int = 3,
    bottom_n: int = 3,
    repeat_frac: float = 0.40
) -> Tuple[set, set]:
    """
    Heuristic:
      - Look at first `top_n` non-empty lines and last `bottom_n` non-empty lines on each page
      - Lines that repeat on >= repeat_frac of pages are likely headers/footers
    """
    from collections import Counter

    top_counter = Counter()
    bot_counter = Counter()

    for lines in pages_lines:
        non_empty = [ln for ln in lines if ln and len(ln) >= 3]
        top = non_empty[:top_n]
        bot = non_empty[-bottom_n:] if len(non_empty) >= bottom_n else non_empty[-len(non_empty):]

        for t in top:
            top_counter[t] += 1
        for b in bot:
            bot_counter[b] += 1

    n_pages = max(1, len(pages_lines))
    top_repeats = {ln for ln, c in top_counter.items() if (c / n_pages) >= repeat_frac}
    bot_repeats = {ln for ln, c in bot_counter.items() if (c / n_pages) >= repeat_frac}
    return top_repeats, bot_repeats


def remove_headers_footers(lines: List[str], top_repeats: set, bot_repeats: set) -> List[str]:
    out = []
    for ln in lines:
        if ln in top_repeats or ln in bot_repeats:
            continue
        # drop standalone page numbers
        if re.fullmatch(r"\d{1,4}", ln.strip()):
            continue
        out.append(ln)
    return out

def is_divider_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    # robust: contains all three keywords
    return (
        re.search(r"\bGoals\b", s, flags=re.I)
        and re.search(r"\bObjectives\b", s, flags=re.I)
        and re.search(r"\bPolicies\b", s, flags=re.I)
    )

def heading_level(line: str) -> Optional[int]:
    """
    Returns an integer level for hierarchical headings.
    Smaller number = higher-level (more general).
    """
    s = (line or "").strip()
    if not s:
        return None

    # Level 1: Element / Chapter
    if re.search(r"\b(Element|Chapter|Appendix)\b", s) and len(s) <= 80 and not s.endswith("."):
        return 1

    # Level 2: Divider
    if is_divider_heading(s):
        return 2

    # Level 3: Goal
    if re.match(r"^Goal\b", s, flags=re.I):
        return 3

    # Level 4: Objective
    if re.match(r"^Objective\b", s, flags=re.I):
        return 4

    # Level 5: Policy
    if re.match(r"^Policy\b", s, flags=re.I):
        return 5

    # Other headings we detect (caps/numbered) treated as mid-level
    if re.match(r"^\d+(?:\.\d+){1,4}\s+\S+", s):
        return 3

    return None


def looks_like_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False

    # Common plan constructs
    if re.match(r"^(GOAL|OBJECTIVE|POLICY|ACTION)\b", s, flags=re.I):
        return True
    
     # Section divider often used in comp plans (robust to punctuation / spacing / &)
    if re.search(r"\bGoals\b", s, flags=re.I) and re.search(r"\bObjectives\b", s, flags=re.I) and re.search(r"\bPolicies\b", s, flags=re.I):
        # Keep it constrained so we don't match full sentences
        if len(s) <= 60 and not s.endswith("."):
            return True

    # Element / Chapter / Section style headings
    # (these are common in comprehensive plans and are often Title Case)
    if re.search(r"\b(Element|Chapter|Appendix)\b", s) and len(s) <= 80:
        # avoid sentences like "This element provides..."
        # heuristic: short-ish and not ending with a period
        if not s.endswith("."):
            return True

    # Numbered headings like "1.2 ..." or "2.3.1 ..."
    if re.match(r"^\d+(?:\.\d+){1,4}\s+\S+", s):
        return True

    # ALL CAPS-ish headings
    letters = re.sub(r"[^A-Z]", "", s)
    alpha = re.sub(r"[^A-Za-z]", "", s)
    if len(s) >= 10 and alpha and (len(letters) / len(alpha)) > 0.85:
        return True

    return False




def pages_to_sections(pages: List[PageText]) -> List[SectionText]:
    sections: List[SectionText] = []
    current_heading: Optional[str] = None
    current_path: List[str] = []   # NEW
    buf: List[str] = []
    start_page: Optional[int] = None
    sid = 0
    force_emit = False

    def flush(end_page: int):
        nonlocal sid, buf, start_page, current_heading, force_emit, current_path
        text = clean_text_basic("\n".join(buf))
        if text or force_emit:
            sid += 1
            sections.append(
                SectionText(
                    section_id=f"s{sid:04d}",
                    heading=current_heading,
                    heading_path=list(current_path),  # snapshot
                    page_start=start_page if start_page is not None else end_page,
                    page_end=end_page,
                    text=text,
                )
            )
        buf = []
        force_emit = False

    def update_path(new_heading: str):
        nonlocal current_path
        lvl = heading_level(new_heading)
        if lvl is None:
            return

        # Ensure path length fits this level (lvl=1 means index 0)
        # Drop deeper/equal levels, then set this level
        while len(current_path) >= lvl:
            current_path.pop()

        current_path.append(new_heading)

    for p in pages:
        if start_page is None:
            start_page = p.page

        for line in p.text.splitlines():
            if looks_like_heading(line):
                flush(end_page=p.page)
                current_heading = line.strip()

                update_path(current_heading)  # NEW

                start_page = p.page
                buf = []
                force_emit = is_divider_heading(current_heading)
            else:
                buf.append(line)

    if pages:
        flush(end_page=pages[-1].page)

    return sections


# ---------- main extraction ----------
def _estimate_repeated_headers_footers(
    pages_lines: List[List[str]],
    top_n: int = 3,
    bottom_n: int = 3,
    repeat_frac: float = 0.40
) -> Tuple[set, set]:
    """
    Collect first/last few non-empty lines per page and mark lines that repeat
    across many pages as header/footer.
    """
    top_counter = Counter()
    bot_counter = Counter()

    for lines in pages_lines:
        non_empty = [ln for ln in lines if ln and len(ln) >= 3]
        top = non_empty[:top_n]
        bot = non_empty[-bottom_n:] if len(non_empty) >= bottom_n else non_empty[-len(non_empty):]
        for t in top:
            top_counter[t] += 1
        for b in bot:
            bot_counter[b] += 1

    n_pages = max(1, len(pages_lines))
    top_repeats = {ln for ln, c in top_counter.items() if (c / n_pages) >= repeat_frac}
    bot_repeats = {ln for ln, c in bot_counter.items() if (c / n_pages) >= repeat_frac}
    return top_repeats, bot_repeats

def _remove_headers_footers(lines: List[str], top_repeats: set, bot_repeats: set) -> List[str]:
    out = []
    for ln in lines:
        if ln in top_repeats or ln in bot_repeats:
            continue
        # drop standalone page numbers
        if re.fullmatch(r"\d{1,4}", ln.strip()):
            continue
        out.append(ln)
    return out


def extract_pdf_pages(pdf_path: str) -> List[PageText]:
    """Extract cleaned text per page from a PDF (removes repeated headers/footers)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf") from e

    doc = fitz.open(pdf_path)

    # 1) collect raw pages once
    raw_pages: List[str] = []
    pages_lines: List[List[str]] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("text") or ""
        raw_pages.append(raw)
        pages_lines.append(raw.splitlines())

    # 2) estimate repeated header/footer lines across pages
    top_repeats, bot_repeats = _estimate_repeated_headers_footers(pages_lines)

    # 3) build PageText with BOTH raw_text (spacing preserved) and cleaned text
    pages: List[PageText] = []
    for i, raw in enumerate(raw_pages):
        lines = raw.splitlines()
        lines = _remove_headers_footers(lines, top_repeats, bot_repeats)

        raw_after_hf = "\n".join(lines)          # preserve spacing for tables
        cleaned = clean_text_basic(raw_after_hf) # ok to normalize for sectioning

        pages.append(PageText(page=i + 1, text=cleaned, raw_text=raw_after_hf))

    doc.close()
    return pages


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



