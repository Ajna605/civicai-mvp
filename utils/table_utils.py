from typing import List
from utils.text_utils import clean_text
from docx.table import Table

# ----------------------------
# Table detection
# ----------------------------
def table_dims(raw_text: str) -> tuple[int, int]:
    lines = [ln.strip() for ln in (raw_text or "").splitlines() if ln.strip()]
    rows = len(lines)
    # estimate cols from the first "pipe" line
    pipe_lines = [ln for ln in lines if "|" in ln]
    if not pipe_lines:
        return rows, 0
    # number of cells ≈ number of pipes - 1 (for boundary pipes)
    cols = max((max(0, ln.count("|") - 1) for ln in pipe_lines), default=0)
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
# Table formatting
# ----------------------------
def table_to_markdown(tbl: Table) -> str:
    rows: List[List[str]] = []
    for row in tbl.rows:
        cells = []
        for cell in row.cells:
            c = clean_text(cell.text)
            c = (c or "").replace("|", "\\|")  # escape pipes
            cells.append(c)
        rows.append(cells)

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