from __future__ import annotations

import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Iterable
from collections import Counter
from dataclasses import dataclass
import pdfplumber


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


#-----------------Table Logic -------------------#
def _extract_table_with_pdfplumber_crop(path, page_number, bbox):
    # bbox from PyMuPDF is (x0, y0, x1, y1)
    x0, y0, x1, y1 = bbox

    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_number - 1]

        cropped = page.crop((x0, y0, x1, y1))

        table = cropped.extract_table(
            table_settings={
            "vertical_strategy": "lines",
            "horizontal_strategy": "text",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "intersection_tolerance": 3,
            "min_words_horizontal": 1,
        }
        )
        return table

def _extract_page_tables_pymupdf(
    page: Any,
    path: Path,
    page_number: int,
) -> list[dict]:
    out: list[dict] = []

    text_blocks = _get_sorted_text_blocks(page)
    tabs = page.find_tables(vertical_strategy="lines", horizontal_strategy="lines_strict")

    # PyMuPDF docs show tabs.tables and table.extract() for content extraction. :contentReference[oaicite:1]{index=1}
    for table_idx, tab in enumerate(getattr(tabs, "tables", [])):
        try:
            extracted = _extract_table_with_pdfplumber_crop(path=path, page_number=page_number, bbox=bbox)
        except Exception:
            continue

        rows = _normalize_matrix(extracted)
        if not rows:
            continue

        headers, body = _split_headers_and_rows(rows)
        if not headers and not body:
            continue

        bbox = _normalize_bbox(getattr(tab, "bbox", None))
        caption = _find_caption_for_bbox(text_blocks, bbox)
        table_id, table_title = _split_caption(caption)

        text = _flatten_table_text(
            caption=caption,
            headers=headers,
            rows=body,
        )

        out.append(
            {
                "source_type": "pdf",
                "path": str(path),
                "doc_id": path.stem,
                "page": page_number,
                "table_idx": table_idx,
                "extractor": "pymupdf",
                "bbox": bbox,
                "caption": caption,
                "table_id": table_id,
                "table_title": table_title,
                "headers": headers,
                "rows": body,
                "n_rows": len(body),
                "n_cols": len(headers) if headers else max((len(r) for r in body), default=0),
                "text": text,
            }
        )

    return out

def _extract_page_tables_pdfplumber(
    path: Path,
    page_number: int,
) -> list[dict]:
    if pdfplumber is None:
        return []

    out: list[dict] = []

    with pdfplumber.open(path) as pdf:
        page = pdf.pages[page_number - 1]

        # pdfplumber is useful for detailed PDF table extraction and visual debugging. :contentReference[oaicite:2]{index=2}
        try:
            tables = page.find_tables()

        except Exception:
            tables = []

        words = page.extract_words() or []
        text_lines = _pdfplumber_words_to_lines(words)

        for table_idx, table in enumerate(tables):
            try:
                extracted = table.extract()
            except Exception:
                continue

            rows = _normalize_matrix(extracted)
            if not rows:
                continue

            caption_from_row, rows = _pop_caption_row(rows)
            headers, body = _split_headers_and_rows(rows)
            bbox = _normalize_bbox(getattr(table, "bbox", None))
            caption = caption_from_row or _find_caption_for_bbox_pdfplumber(text_lines, bbox)
            table_id, table_title = _split_caption(caption)

            text = _flatten_table_text(
                caption=caption,
                headers=headers,
                rows=body,
            )

            out.append(
                {
                    "source_type": "pdf",
                    "path": str(path),
                    "doc_id": path.stem,
                    "page": page_number,
                    "table_idx": table_idx,
                    "extractor": "pdfplumber",
                    "bbox": bbox,
                    "caption": caption,
                    "table_id": table_id,
                    "table_title": table_title,
                    "headers": headers,
                    "rows": body,
                    "n_rows": len(body),
                    "n_cols": len(headers) if headers else max((len(r) for r in body), default=0),
                    "text": text,
                }
            )

    return out


def _get_sorted_text_blocks(page: Any) -> list[dict]:
    """
    Returns text blocks sorted top-to-bottom, left-to-right.
    PyMuPDF page.get_text('blocks') returns tuples like:
    (x0, y0, x1, y1, text, block_no, block_type)
    """
    raw_blocks = page.get_text("blocks") or []
    blocks: list[dict] = []

    for b in raw_blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, text = b[:5]
        if not text or not str(text).strip():
            continue
        blocks.append(
            {
                "bbox": [float(x0), float(y0), float(x1), float(y1)],
                "text": _clean_text(str(text)),
            }
        )

    blocks.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
    return blocks

TABLE_CAPTION_RE = re.compile(
    r"^\s*((?:Table|TABLE|Tbl\.?)\s+[A-Za-z0-9.\-]+)\s*[:.\-]?\s*(.*)\s*$"
)

## To extract Heading
def _is_placeholder(x: str) -> bool:
    x = (x or "").strip().lower()
    return not x or x.startswith("column_")


def _pop_caption_row(rows: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    if not rows:
        return None, rows

    first_row = rows[0]

    # existing "Table X" detection
    joined = " ".join((c or "").strip() for c in first_row if c).strip()
    if TABLE_CAPTION_RE.match(joined):
        return joined, rows[1:]

    # new rule: only one real cell
    real_cells = [c.strip() for c in first_row if not _is_placeholder(c)]

    if len(real_cells) == 1:
        return real_cells[0], rows[1:]

    return None, rows
def _find_caption_for_bbox(
    text_blocks: list[dict],
    bbox: list[float] | None,
    max_above_distance: float = 80.0,
    max_below_distance: float = 35.0,
) -> str | None:
    """
    Heuristic:
    1. Prefer nearest caption-like block above the table.
    2. Then allow a small search below.
    """
    if not bbox:
        return None

    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, str]] = []

    for block in text_blocks:
        bx0, by0, bx1, by1 = block["bbox"]
        text = block["text"]

        if not TABLE_CAPTION_RE.match(text):
            continue

        horizontal_overlap = min(x1, bx1) - max(x0, bx0)
        if horizontal_overlap < 0:
            continue

        # Above-table preference
        if by1 <= y0:
            dist = y0 - by1
            if dist <= max_above_distance:
                candidates.append((dist, text))
        # Small below-table fallback
        elif by0 >= y1:
            dist = by0 - y1
            if dist <= max_below_distance:
                candidates.append((1000.0 + dist, text))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]

def _find_caption_for_bbox_pdfplumber(
    lines: list[dict],
    bbox: list[float] | None,
    max_above_distance: float = 120.0,
    max_below_distance: float = 40.0,
) -> str | None:
    if not bbox:
        return None

    x0, y0, x1, y1 = bbox
    candidates: list[tuple[float, str]] = []

    for line in lines:
        text = (line.get("text") or "").strip()
        line_bbox = line.get("bbox")
        if not line_bbox or not TABLE_CAPTION_RE.match(text):
            continue

        lx0, ly0, lx1, ly1 = line_bbox

        if ly1 <= y0:
            ydist = y0 - ly1
            if ydist <= max_above_distance:
                xdist = min(
                    abs(lx0 - x0),
                    abs(lx1 - x1),
                    abs(((lx0 + lx1) / 2) - ((x0 + x1) / 2)),
                )
                score = ydist + 0.15 * xdist
                candidates.append((score, text))

        elif ly0 >= y1:
            ydist = ly0 - y1
            if ydist <= max_below_distance:
                xdist = min(
                    abs(lx0 - x0),
                    abs(lx1 - x1),
                    abs(((lx0 + lx1) / 2) - ((x0 + x1) / 2)),
                )
                score = 1000.0 + ydist + 0.15 * xdist
                candidates.append((score, text))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def _normalize_matrix(matrix: Iterable[Iterable[Any]] | None) -> list[list[str]]:
    if not matrix:
        return []

    rows: list[list[str]] = []
    max_cols = 0

    for row in matrix:
        if row is None:
            continue
        cleaned = [_clean_cell(c) for c in row]
        if not any(cleaned):
            continue
        rows.append(cleaned)
        max_cols = max(max_cols, len(cleaned))

    if not rows:
        return []

    # pad rows to consistent width
    padded: list[list[str]] = []
    for row in rows:
        if len(row) < max_cols:
            row = row + [""] * (max_cols - len(row))
        padded.append(row)

    # remove repeated fully-empty trailing columns
    padded = _trim_empty_edge_columns(padded)

    # remove rows that became empty after trimming
    padded = [r for r in padded if any(cell.strip() for cell in r)]
    return padded


def _split_headers_and_rows(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """
    Assumption:
    - first non-empty row is header row
    - rest are body rows

    This matches many DOCX normalization pipelines and keeps the raw stage simple.
    """
    if not rows:
        return [], []

    headers = [_normalize_header_cell(c, idx) for idx, c in enumerate(rows[0])]
    body = rows[1:] if len(rows) > 1 else []

    # if header row looks empty / useless, downgrade into body
    non_empty_headers = sum(1 for h in headers if h and not h.startswith("column_"))
    if non_empty_headers == 0:
        return [], rows

    # drop repeated header rows from body
    filtered_body: list[list[str]] = []
    normalized_header_sig = tuple(h.strip().lower() for h in headers)

    for row in body:
        row_sig = tuple(c.strip().lower() for c in row)
        if row_sig == normalized_header_sig:
            continue
        filtered_body.append(row)

    return headers, filtered_body


def _normalize_header_cell(value: str, idx: int) -> str:
    value = _clean_text(value)
    return value if value else f"column_{idx + 1}"


def _trim_empty_edge_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return rows

    n_cols = max(len(r) for r in rows)
    keep_left = 0
    keep_right = n_cols - 1

    # trim empty leading columns
    while keep_left < n_cols:
        if any((r[keep_left].strip() if keep_left < len(r) else "") for r in rows):
            break
        keep_left += 1

    # trim empty trailing columns
    while keep_right >= keep_left:
        if any((r[keep_right].strip() if keep_right < len(r) else "") for r in rows):
            break
        keep_right -= 1

    return [r[keep_left : keep_right + 1] for r in rows]


def _split_caption(caption: str | None) -> tuple[str | None, str | None]:
    if not caption:
        return None, None

    m = TABLE_CAPTION_RE.match(caption)
    if not m:
        return None, caption.strip()

    table_id = m.group(1).strip()
    title = (m.group(2) or "").strip()
    return table_id, title or None


def _flatten_table_text(
    caption: str | None,
    headers: list[str],
    rows: list[list[str]],
) -> str:
    parts: list[str] = []

    if caption:
        parts.append(caption)

    if headers:
        parts.append(" | ".join(headers))

    for row in rows:
        parts.append(" | ".join(_clean_text(c) for c in row))

    return "\n".join(p for p in parts if p.strip())


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return _clean_text(str(value))


def _clean_text(text: str) -> str:
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


def _normalize_bbox(bbox: Any) -> list[float] | None:
    if not bbox:
        return None
    try:
        x0, y0, x1, y1 = bbox
        return [float(x0), float(y0), float(x1), float(y1)]
    except Exception:
        return None


def _table_fingerprint(table: dict) -> tuple:
    """
    Used for dedup across extractors.
    """
    headers = tuple(table.get("headers") or [])
    rows = table.get("rows") or []
    row_sample = tuple(tuple(r) for r in rows[:3])
    return (
        table.get("page"),
        headers,
        row_sample,
        table.get("caption"),
    )


def _pdfplumber_words_to_lines(words: list[dict]) -> list[dict]:
    """
    Reconstruct simple text lines from pdfplumber words for caption matching.
    """
    if not words:
        return []

    # group by approximate y position
    grouped: list[list[dict]] = []
    tolerance = 3.0

    words_sorted = sorted(words, key=lambda w: (float(w["top"]), float(w["x0"])))

    for w in words_sorted:
        top = float(w["top"])
        if not grouped:
            grouped.append([w])
            continue

        prev_top = float(grouped[-1][0]["top"])
        if abs(top - prev_top) <= tolerance:
            grouped[-1].append(w)
        else:
            grouped.append([w])

    lines: list[dict] = []
    for group in grouped:
        group = sorted(group, key=lambda w: float(w["x0"]))
        text = " ".join(str(w["text"]) for w in group).strip()
        if not text:
            continue

        x0 = min(float(w["x0"]) for w in group)
        x1 = max(float(w["x1"]) for w in group)
        top = min(float(w["top"]) for w in group)
        bottom = max(float(w["bottom"]) for w in group)

        lines.append(
            {
                "bbox": [x0, top, x1, bottom],
                "text": _clean_text(text),
            }
        )

    return lines

# -----------------------------------------------------------------------


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






