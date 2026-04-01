from utils.table_utils import is_repeated_title_row, _single_row_normalization, _align_headers_to_row
from typing import Any
import re

def _is_placeholder_header_cell(x: str) -> bool:
    x = (x or "").strip().lower()
    return not x or x.startswith("column_")

def _merge_row_continuation(prev_row: list[str], continuation: list[str]) -> list[str]:
    out = []
    max_len = max(len(prev_row), len(continuation))

    for i in range(max_len):
        left = prev_row[i].strip() if i < len(prev_row) and prev_row[i] else ""
        right = continuation[i].strip() if i < len(continuation) and continuation[i] else ""

        if _is_placeholder_header_cell(right):
            right = ""

        if left and right:
            out.append(f"{left} {right}".strip())
        else:
            out.append(left or right)

    return out

def _merge_same_caption_tables(tables: list[dict]) -> list[dict]:
    if not tables:
        return tables

    merged = [tables[0]]

    for curr in tables[1:]:
        prev = merged[-1]

        same_caption = (
            prev.get("caption")
            and curr.get("caption")
            and prev.get("caption") == curr.get("caption")
        )

        if not same_caption:
            merged.append(curr)
            continue

        curr_header = curr.get("header_terms") or []
        curr_rows = curr.get("rows") or []

        # first: if current "header" is really a continuation fragment, merge it
        if curr_header and prev.get("rows"):
            prev["rows"][-1] = _merge_row_continuation(prev["rows"][-1], curr_header)

        # then append the actual body rows
        if curr_rows:
            prev["rows"].extend(curr_rows)

        prev["n_rows"] = len(prev["rows"])
        prev["n_cols"] = max(
            len(prev.get("header_terms") or []),
            max((len(r) for r in prev["rows"]), default=0),
        )

    return merged

TABLE_CAPTION_RE = re.compile(
    r"^\s*((?:Table|TABLE|Tbl\.?)\s+[A-Za-z0-9.\-]+)\s*[:.\-]?\s*(.*)\s*$"
)
# Detect New table within table
def _row_is_new_table_caption(row: list[str]) -> str | None:
    real_cells = [c.strip() for c in row if c and c.strip() and not c.strip().lower().startswith("column_")]
    if len(real_cells) != 1:
        return None

    text = real_cells[0]
    if TABLE_CAPTION_RE.match(text):
        return text

    return None

def _make_split_table_base(tbl: dict, caption: str | None = None, keep_headers: bool = False) -> dict:
    return {
        "table_id": tbl.get("table_id"),
        "source_file": tbl.get("source_file"),
        "source_type": tbl.get("source_type"),
        "table_index": tbl.get("table_index"),
        "section_path": tbl.get("section_path", []) or [],
        "preceding_text": None,
        "caption": caption,
        "rows": [],
        "headers": (tbl.get("headers", []) or []) if keep_headers else [],
        "header_terms": (tbl.get("header_terms", []) or []) if keep_headers else [],
        "page": tbl.get("page"),
        "bbox": tbl.get("bbox"),
    }


def _split_on_embedded_table_captions(tbl: dict) -> list[dict]:
    rows = tbl.get("rows", []) or []
    if not rows:
        return [tbl]

    out = []
    current = _make_split_table_base(
        tbl,
        caption=tbl.get("caption"),
        keep_headers=True,   # keep original headers on the first table
    )
    current_rows = []

    for row in rows:
        new_caption = _row_is_new_table_caption(row)

        if new_caption and current_rows:
            current["rows"] = current_rows
            out.append(current)

            current = _make_split_table_base(
                tbl,
                caption=new_caption,
                keep_headers=False,   # fresh table starts clean
            )
            current_rows = []
            continue

        current_rows.append(row)

    current["rows"] = current_rows
    out.append(current)
    return out


def normalize_extracted_tables(raw_tables, path, source_type):
    normalized = []

    expanded_tables = []
    for tbl in raw_tables:
        expanded_tables.extend(_split_on_embedded_table_captions(tbl))
    for tbl in expanded_tables:
        rows = tbl.get("rows", []) or []

        table_title = tbl.get("caption")

        header_terms = tbl.get("headers", []) or tbl.get("header_terms", []) or []
        data_rows = rows

        if rows and is_repeated_title_row(rows[0]):
            if not table_title:
                table_title = rows[0][0].strip()
            data_rows = rows[1:]

        # only infer header from first row if extractor did not already provide one
        if not header_terms and data_rows:
            header_terms = data_rows[0]
            data_rows = data_rows[1:]
        n_rows = len(data_rows)
        n_cols = max(
            len(header_terms) if header_terms else 0,
            max((len(r) for r in data_rows), default=0)
        )

        section_path = tbl.get("section_path", []) or []
        preceding_text = (tbl.get("preceding_text") or "").strip() or None

        search_parts = []
        if table_title:
            search_parts.append(table_title)
        if section_path:
            search_parts.append(" > ".join(section_path))
        if preceding_text:
            search_parts.append(preceding_text)
        if header_terms:
            search_parts.append(" | ".join(x for x in header_terms if x))
        for row in data_rows:
            search_parts.append(" | ".join(x for x in row if x))

        normalized.append({
            "table_id":(tbl.get("table_id") or table_title or "").strip(),
            "source_file": str(path),
            "source_type": source_type,
            "table_index": tbl.get("table_idx", 0),
            "section_path": section_path,
            "caption": table_title,
            "preceding_text": preceding_text,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "header_terms": header_terms,
            "rows": data_rows,
            "search_text": "\n".join(search_parts).strip(),
            "page": tbl.get("page"),
            "bbox": tbl.get("bbox"),
        })

    return _merge_same_caption_tables(normalized)



def normalize_table_rows(table_record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one normalized table record into lean row-level records.
    """
    table_id = table_record.get("table_id")
    source_file = table_record.get("source_file")
    source_type = table_record.get("source_type")
    table_index = table_record.get("table_index")
    section_path = table_record.get("section_path", []) or []
    caption = table_record.get("caption")

    raw_rows = table_record.get("rows", []) or []
    header_terms = table_record.get("header_terms", []) or []

    rows, used_header_as_row = _single_row_normalization(raw_rows, header_terms)
    if not rows:
        return []

    row_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row = [str(cell).strip() if cell else "" for cell in row]

        if used_header_as_row:
            headers = [f"col_{i+1}" for i in range(len(row))]
        else:
            headers = _align_headers_to_row(header_terms, len(row))

        row_values = {
            header: value
            for header, value in zip(headers, row)
            if header
        }

        row_label = next((value for value in row if value), "")

        search_parts: list[str] = []
        if caption:
            search_parts.append(str(caption))
        if row_label:
            search_parts.append(row_label)

        for key, value in row_values.items():
            if value and value != row_label:
                search_parts.append(f"{key} {value}")

        row_records.append({
            "block_type": "table_row",
            "row_id": f"{table_id}__r{row_index}",
            "table_id": table_id,
            "source_file": source_file,
            "source_type": source_type,
            "table_index": table_index,
            "row_index": row_index,
            "row_label": row_label,
            "section_path": section_path,
            "caption": caption,
            "header_terms": headers,
            "row_values": row_values,
            "search_text": " ".join(search_parts).strip(),
        })

    return row_records

def normalize_all_table_rows(table_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for table_record in table_records:
        all_rows.extend(normalize_table_rows(table_record))

    return all_rows