## Get all labels from CSV file

import duckdb, json, argparse
from pathlib import Path


out_path = "sql_engine/llm_utils/metadata.json"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    return p.parse_args()

def export_metadata(con: duckdb.DuckDBPyConnection, table="facts") -> dict:
    def col_values(col):
        rows = con.execute(f"SELECT DISTINCT {col} FROM {table} WHERE {col} IS NOT NULL ORDER BY {col}").fetchall()
        return [r[0] for r in rows]

    meta = {
        "subjects": col_values("subject"),
        "stat_types": col_values("stat_type"),
        "labels": col_values("label"),
        "measures": col_values("measure"),
        "geos": col_values("geo"),
        "years": col_values("year"),
        "units": col_values("unit"),
    }
    return meta

if __name__ == "__main__":
    # usage:
    args = parse_args()
    con = duckdb.connect(args.db)
    meta = export_metadata(con, "facts")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        f_out.write(json.dumps(meta))

        # open("metadata.json","w").write(json.dumps(meta))
    print("✅ Metadata json file created")