from utils.table_utils import is_repeated_title_row
from typing import Any

def normalize_extracted_tables(raw_tables, path, source_type):
    normalized = []

    for tbl in raw_tables:
        rows = tbl.get("rows", []) or []

        table_title = tbl.get("caption")
        header_terms = []
        data_rows = rows

        if rows and is_repeated_title_row(rows[0]):
            if not table_title:
                table_title = rows[0][0].strip()
            data_rows = rows[1:]

        if data_rows:
            header_terms = data_rows[0]
            data_rows = data_rows[1:]
        n_rows = len(rows)
        n_cols = max((len(r) for r in rows), default=0)

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
            "search_text": " ".join(search_parts).strip(),
        })

    return normalized


def normalize_table_rows(table_record: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Convert one normalized table record into row-level records.

    Expected input fields on table_record:
      - table_id
      - source_file
      - source_type
      - table_index
      - section_path
      - caption
      - header_terms
      - rows   # data rows only (header already removed)
    """
    table_id = table_record.get("table_id")
    source_file = table_record.get("source_file")
    source_type = table_record.get("source_type")
    table_index = table_record.get("table_index", 0)
    section_path = table_record.get("section_path", []) or []
    caption = table_record.get("caption")
    preceding_text = table_record.get("preceding_text")
    header_terms = table_record.get("header_terms", []) or []
    rows = table_record.get("rows", []) or []

    row_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        # pad/truncate row to header length for consistency
        row = list(row)
        if len(header_terms) > len(row):
            row = row + [""] * (len(header_terms) - len(row))
        elif len(header_terms) < len(row):
            # keep extras by extending headers if needed
            extra_count = len(row) - len(header_terms)
            header_terms_extended = header_terms + [f"extra_col_{i+1}" for i in range(extra_count)]
        else:
            header_terms_extended = header_terms

        if len(header_terms) == len(row):
            header_terms_extended = header_terms

        row_values = {
            str(col).strip(): (str(val).strip() if val is not None else "")
            for col, val in zip(header_terms_extended, row)
            if str(col).strip()
        }

        row_label = ""
        if header_terms_extended and row:
            first_header = header_terms_extended[0]
            row_label = row_values.get(first_header, "") or (str(row[0]).strip() if row else "")

        search_parts: list[str] = []
        if caption:
            search_parts.append(str(caption))
        if section_path:
            search_parts.append(" > ".join(section_path))
        if preceding_text:
            search_parts.append(str(preceding_text))

        for k, v in row_values.items():
            if v:
                search_parts.append(f"{k} {v}")

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
            "search_text": " ".join(search_parts).strip(),
        })

    return row_records

def normalize_all_table_rows(table_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []

    for table_record in table_records:
        all_rows.extend(normalize_table_rows(table_record))

    return all_rows