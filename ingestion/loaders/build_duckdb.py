# one command: load normalized CSVs into DuckDB
# ingestion/build_duckdb.py
import argparse
from pathlib import Path
from sql_engine.duckdb_loader import build_duckdb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--rebuild", default=True)
    p.add_argument("--table_name", required=False)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_duckdb(Path(args.db), rebuild=args.rebuild, table_name=args.table_name)
    print(f"✅ DuckDB ready at {args.db}")