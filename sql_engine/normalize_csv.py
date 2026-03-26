# sql/csv_normalize.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import pandas as pd
import re
import argparse, json
from utils.text_utils import clean_text, infer_table_name

YEAR_RE = re.compile(r"^(19|20)\d{2}$")

def _clean_header(h: Any) -> str:
    if h is None:
        return ""
    return str(h).strip()

_MISSING = {"", "(X)", "—", "-", "N", "null", "None"}

def _coerce_float(x):
    if x is None:
        return None

    s = str(x)

    # normalize whitespace + quotes
    s = s.replace("\u00a0", " ").strip()
    if s.startswith('"') and s.endswith('"') and len(s) >= 2:
        s = s[1:-1].strip()
    else:
        s = s.replace('"', '').strip()

    # missing/suppressed markers
    if s in _MISSING:
        return None

    # normalize common ACS formatting
    s = s.replace("±", "")          # margin-of-error prefix
    s = s.replace("%", "")          # percent sign
    s = s.replace(",", "")          # thousands separators
    s = s.replace("−", "-")         # unicode minus to ascii

    # final cleanup: keep digits, dot, leading minus
    s = s.strip()

    try:
        return float(s)
    except Exception:
        return None

def _looks_like_year(col: str) -> bool:
    return bool(YEAR_RE.match(col))

def infer_label_col(df: pd.DataFrame) -> str:
    # pick first column that is mostly non-numeric
    best = df.columns[0]
    best_score = -1.0
    for c in df.columns:
        series = df[c].dropna().astype(str).head(50)
        if len(series) == 0:
            continue
        numeric_hits = 0
        for v in series:
            if _coerce_float(v) is not None:
                numeric_hits += 1
        score = 1.0 - (numeric_hits / max(len(series), 1))
        if score > best_score:
            best_score = score
            best = c
    return best

def normalize_csv_to_facts(
    csv_path: Path,
    source_file: str,
    default_geo: Optional[str] = None,
    default_unit: Optional[str] = None,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path)
    df.columns = [_clean_header(c) for c in df.columns]

    # If it already looks tidy:
    lower_cols = {c.lower(): c for c in df.columns}
    if "value" in lower_cols and ("measure" in lower_cols or "metric" in lower_cols):
        # map to canonical
        value_c = lower_cols["value"]
        measure_c = lower_cols.get("measure") or lower_cols.get("metric")
        label_c = lower_cols.get("label") or lower_cols.get("name") or infer_label_col(df)
        year_c = lower_cols.get("year")
        unit_c = lower_cols.get("unit")
        geo_c = lower_cols.get("geo")
        subject_c = lower_cols.get("subject")
        stat_type_c = lower_cols.get("stat_type") or lower_cols.get("stat")  # optional aliases


        out = []
        for i, row in df.iterrows():
            out.append({
                "row_id": None,  # assigned later in DuckDB
                "source_file": source_file,
                "label": str(row.get(label_c, "")).strip(),
                "measure": str(row.get(measure_c, "")).strip(),
                "value": _coerce_float(row.get(value_c)),
                "raw_value": row.get(value_c),
                "year": int(row.get(year_c)) if year_c and _coerce_float(row.get(year_c)) is not None else None,
                "unit": str(row.get(unit_c)).strip() if unit_c and pd.notna(row.get(unit_c)) else default_unit,
                "geo": str(row.get(geo_c)).strip() if geo_c and pd.notna(row.get(geo_c)) else default_geo,
                "subject": str(row.get(subject_c)).strip() if subject_c and pd.notna(row.get(subject_c)) else None,
                "stat_type": str(row.get(stat_type_c)).strip() if stat_type_c and pd.notna(row.get(stat_type_c)) else None,
                # Provenance
                "raw_row": str(row.get(label_c, "")).strip(),
                "raw_col": value_c,
                "orig_row_id": int(i),
                "orig_col_id": int(df.columns.get_loc(value_c)),
            })
        return out

    # Otherwise assume wide → melt
    df = df.reset_index().rename(columns={"index": "orig_row_id"})
    col_id_map = {c: idx for idx, c in enumerate(df.columns)}
    stat_col = infer_label_col(df)
    id_vars = [stat_col]
    value_vars = [c for c in df.columns if c != stat_col]
    
    melted = df.melt(
        id_vars=["orig_row_id"] + id_vars,
        value_vars=value_vars,
        var_name="col",
        value_name="raw_value"
    )
    out: List[Dict[str, Any]] = []

    is_acs = any("!!" in str(c) for c in df.columns)

    for _, r in melted.iterrows():
        row_stat = str(r[stat_col]).strip()   # e.g., "Sex ratio (males per 100 females)"
        col = str(r["col"]).strip()            # e.g., "Coral Gables city, Florida!!Total!!Estimate"
        raw = r["raw_value"]
        val = _coerce_float(raw)

        # default outputs
        label = None
        measure = None
        subject = None
        stat_type = None
        year = None

        orig_row_id = int(r["orig_row_id"])
        orig_col_id = int(col_id_map[col])

        # Case 1: Year columns (typical time-series wide table)
        if _looks_like_year(col):
            year = int(col)
            label = row_stat              # <-- generic case: label is the entity in first column
            measure = "value"             # <-- generic measure
        # Case 2: ACS-style "geo!!subject!!Estimate"
        elif is_acs and"!!" in col:
            parts = [p.strip() for p in col.split("!!") if p is not None]
            # Geo is always first
            geo = parts[0] if len(parts) > 0 else None
            subject = parts[1] if len(parts) > 1 else None
            stat_type = parts[2] if len(parts) > 2 else None

            label = geo                   # ✅ label becomes geography
            measure = row_stat            # ✅ measure becomes row statistic name
        # Case 3: Generic wide table (no years, no !!)
        else:
            # Common pattern: first col is entity, other headers are measures
            label = row_stat              # entity
            measure = col                 # measure name from header

        out.append({
            "row_id": None,
            "source_file": source_file,
            "label": label,
            "measure": measure,
            "value": val,
            "raw_value": raw,
            "year": year,
            "unit": default_unit,
            "geo": label,                 # optional duplicate; or keep default_geo if you prefer
            "subject": subject,
            "stat_type": stat_type,
            "raw_row": clean_text(r[stat_col]),  # or label_c depending on your taste
            "raw_col": r["col"],
            "orig_row_id": orig_row_id ,
            "orig_col_id": orig_col_id  
        })

    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_dir = Path(args.in_dir)
    table_name = infer_table_name(in_dir)
    out_path = Path("data/normalized/csv") / f"{table_name}.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    csv_files = sorted(in_dir.glob("*.csv"))
    total = 0

    with out_path.open("w", encoding="utf-8") as f:
        for csv_path in csv_files:

            facts = normalize_csv_to_facts(
                csv_path=csv_path,
                source_file=csv_path.name,
            )
            for row in facts:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(facts)
            print(f"Normalized {csv_path.name}: {len(facts)} rows")

    print(f"✅ Wrote {total} fact rows to {out_path}")