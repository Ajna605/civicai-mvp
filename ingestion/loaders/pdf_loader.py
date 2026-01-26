from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Tuple, Optional
from ingestion.schema.normalized_doc import NormalizedDoc, SourceInfo, Section, Table

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

# ---------- basic helpers ----------

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


from dataclasses import dataclass
from typing import Optional

@dataclass
class SectionText:
    section_id: str
    heading: Optional[str]
    heading_path: List[str]   # NEW
    page_start: int
    page_end: int
    text: str

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



# ---------- output structure ----------

@dataclass
class PageText:
    page: int          # 1-indexed
    text: str          # cleaned text


# ---------- main extraction ----------
from collections import Counter
from typing import Tuple

def _split_lines(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines()]

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
    """Extract cleaned text per page from a PDF (Step 2: remove repeated headers/footers)."""
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf") from e

    doc = fitz.open(pdf_path)

    raw_pages: List[str] = []
    pages_lines: List[List[str]] = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        raw = page.get_text("text") or ""
        raw_pages.append(raw)
        pages_lines.append(_split_lines(raw))

    top_repeats, bot_repeats = _estimate_repeated_headers_footers(pages_lines)

    pages: List[PageText] = []
    for i, raw in enumerate(raw_pages):
        lines = _split_lines(raw)
        lines = _remove_headers_footers(lines, top_repeats, bot_repeats)
        cleaned = clean_text_basic("\n".join(lines))
        pages.append(PageText(page=i + 1, text=cleaned))

    return pages

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def normalize_pdf(pdf_path: str, doc_id: str | None = None) -> NormalizedDoc:
    file_name = os.path.basename(pdf_path)
    doc_id = doc_id or os.path.splitext(file_name)[0]

    pages = extract_pdf_pages(pdf_path)
    secs = pages_to_sections(pages)

    out_sections: list[Section] = []
    for s in secs:
        out_sections.append(
            Section(
                section_id=s.section_id,
                heading_path=s.heading_path,
                page_start=s.page_start,
                page_end=s.page_end,
                text=s.text,
                tables=[],  # tables later
            )
        )

    src = SourceInfo(
        file_name=file_name,
        file_type="pdf",
        sha256=_sha256_file(pdf_path),
        ingested_at=datetime.now().isoformat(),
    )

    return NormalizedDoc(
        doc_id=doc_id,
        source=src,
        sections=out_sections,
    )
