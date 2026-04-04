import duckdb, json, argparse
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--table",required=False, default="demographics_facts")
    return p.parse_args()

def col_values(con: duckdb.DuckDBPyConnection, table: str, col: str) -> List[str]:
    rows = con.execute(
        f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col}"
    ).fetchall()
    return [r[0] for r in rows]

def build_measure_groups(con: duckdb.DuckDBPyConnection, table: str):
    """
    Reconstruct heading hierarchy using original row order preserved in orig_row_id.
    Assumes a flat heading -> measure structure within each source_file.
    """

    rows = con.execute(f"""
        SELECT
          source_file,
          orig_row_id,
          ANY_VALUE(measure) AS measure,
          MAX(CASE WHEN value IS NOT NULL AND NOT isnan(value) THEN 1 ELSE 0 END) AS has_numeric_value
        FROM {table}
        WHERE measure IS NOT NULL
          AND TRIM(measure) <> ''
        GROUP BY source_file, orig_row_id
        ORDER BY source_file, orig_row_id
    """).fetchall()

    groups = {}
    measure_to_group = {}
    headings = []
    leaf_measures = []
    source_files = []

    current_heading_by_file = {}

    for source_file, orig_row_id, measure, has_numeric_value in rows:
        if source_file not in source_files:
            source_files.append(source_file)

        m = (measure or "").strip()
        if not m:
            continue

        is_heading = (has_numeric_value == 0)

        if is_heading:
            current_heading_by_file[source_file] = m
            if m not in groups:
                groups[m] = []
                headings.append(m)
            continue

        leaf_measures.append(m)
        h = current_heading_by_file.get(source_file)
        if h:
            groups[h].append(m)
            measure_to_group.setdefault(m, h)

    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    headings = dedupe(headings)
    leaf_measures = dedupe(leaf_measures)

    for h in list(groups.keys()):
        groups[h] = dedupe(groups[h])

    source_files = dedupe(source_files)

    return groups, measure_to_group, headings, leaf_measures, source_files

def export_metadata(con: duckdb.DuckDBPyConnection, table="facts") -> dict:
    def col_values(col):
        rows = con.execute(
            f"""
            SELECT DISTINCT {col}
            FROM {table}
            WHERE {col} IS NOT NULL
            ORDER BY {col}
            """
        ).fetchall()
        return [r[0] for r in rows]

    measure_groups, measure_to_group, measure_headings, leaf_measures, source_files = build_measure_groups(con, table)

    meta = {
        "source_files": source_files,
        "metric_names": col_values("subject"),
        "stat_types": col_values("stat_type"),
        "geos": col_values("label"),
        "groups": leaf_measures,
        "geos": col_values("geo"),
        "years": col_values("year"),
        "units": col_values("unit"),
        "measure_headings": measure_headings,
        "table_name": table,
        "measure_groups": measure_groups,
        "measure_to_group": measure_to_group,
    }
    return meta

if __name__ == "__main__":
    args = parse_args()
    con = duckdb.connect(args.db)
    meta = export_metadata(con, args.table)
    out_path = f"storage/metadata/{args.table}_metadata.json"
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)

    outp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("✅ Metadata json file created:", outp)