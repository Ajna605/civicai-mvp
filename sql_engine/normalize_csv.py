# sql/csv_normalize.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Any
import pandas as pd
import re
import argparse, json

YEAR_RE = re.compile(r"^(19|20)\d{2}$")

def _clean_header(h: Any) -> str:
    if h is None:
        return ""
    return str(h).strip()

def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"na", "n/a", "null", "none", "-"}:
        return None
    # remove common formatting
    s2 = s.replace(",", "")
    s2 = s2.replace("$", "")
    s2 = s2.replace("%", "")
    try:
        return float(s2)
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
            })
        return out

    # Otherwise assume wide → melt
    label_col = infer_label_col(df)
    id_vars = [label_col]
    value_vars = [c for c in df.columns if c != label_col]

    melted = df.melt(id_vars=id_vars, value_vars=value_vars, var_name="col", value_name="raw_value")
    out: List[Dict[str, Any]] = []

    for _, r in melted.iterrows():
        label = str(r[label_col]).strip()
        col = str(r["col"]).strip()
        raw = r["raw_value"]
        val = _coerce_float(raw)

        year = int(col) if _looks_like_year(col) else None
        measure = "value" if year is not None else col  # if year columns, measure is generic
        # If year columns, measure might be in filename or elsewhere; keep it simple now.

        out.append({
            "row_id": None,
            "source_file": source_file,
            "label": label,
            "measure": measure,
            "value": val,
            "raw_value": raw,
            "year": year,
            "unit": default_unit,
            "geo": default_geo,
        })
    return out



def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path)
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