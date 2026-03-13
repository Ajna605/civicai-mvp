from utils.table_utils import is_repeated_title_row
from typing import Any

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

def normalize_extracted_tables(raw_tables, path, source_type):
    normalized = []

    for tbl in raw_tables:
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
            "table_id": f"{path.stem}__tbl_{tbl.get('table_index', 0)}",
            "source_file": str(path),
            "source_type": source_type,
            "table_index": tbl.get("table_index", 0),
            "section_path": section_path,
            "caption": table_title,
            "preceding_text": preceding_text,
            "n_rows": n_rows,
            "n_cols": n_cols,
            "header_terms": header_terms,
            "rows": data_rows,
            "search_text": "\n".join(search_parts).strip()
        })

    return _merge_same_caption_tables(normalized)


def normalize_table_rows(table_record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one normalized table record into lean row-level records.
    """

    table_id = table_record.get("table_id")
    source_file = table_record.get("source_file")
    source_type = table_record.get("source_type")
    table_index = table_record.get("table_index", 0)

    section_path = table_record.get("section_path", []) or []
    caption = table_record.get("caption")

    header_terms = table_record.get("header_terms", []) or []
    rows = table_record.get("rows", []) or []

    row_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row = list(row)

        # align headers with row length
        if len(header_terms) < len(row):
            header_terms_extended = header_terms + [
                f"extra_col_{i+1}" for i in range(len(row) - len(header_terms))
            ]
        else:
            header_terms_extended = header_terms[:]

        if len(header_terms_extended) > len(row):
            row += [""] * (len(header_terms_extended) - len(row))

        row_values = {
            str(col).strip(): (str(val).strip() if val else "")
            for col, val in zip(header_terms_extended, row)
            if str(col).strip()
        }

        # first column usually acts as row label
        row_label = ""
        if header_terms_extended and row:
            first_header = header_terms_extended[0]
            row_label = row_values.get(first_header, "") or str(row[0]).strip()

        # lean search text
        search_parts: list[str] = []

        if caption:
            search_parts.append(str(caption))

        if row_label:
            search_parts.append(row_label)

        for k, v in row_values.items():
            if v and v != row_label:
                search_parts.append(f"{k} {v}")

        search_text = " ".join(search_parts).strip()

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
            "header_terms": header_terms_extended,
            "row_values": row_values,
            "search_text": search_text,
        })

    return row_records

def normalize_all_table_rows(table_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for table_record in table_records:
        all_rows.extend(normalize_table_rows(table_record))

    return all_rows