# one command: load normalized CSVs into DuckDB
# ingestion/build_duckdb.py
import argparse
from pathlib import Path
from sql_engine.duckdb_loader import build_duckdb

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--facts", required=True)
    p.add_argument("--db", required=True)
    p.add_argument("--rebuild", action="store_true")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    build_duckdb(Path(args.db), Path(args.facts), rebuild=args.rebuild)
    print(f"✅ DuckDB ready at {args.db}")