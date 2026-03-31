## Build CSV index from duckDB
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional
import duckdb
from llama_index.core import Document, VectorStoreIndex
from rag.settings import apply_settings
apply_settings()


def parse_args():
    p = argparse.ArgumentParser(description="Build a LlamaIndex VectorStoreIndex from DuckDB tables")
    p.add_argument("--duckdb_path", required=True, help="Path to DuckDB database file")
    p.add_argument("--persist_dir", required=True, help="Directory to persist the index to")

    # If omitted, we auto-discover all tables.
    p.add_argument("--tables", nargs="*", default=None, help="Optional list of table names to index. If omitted, indexes all tables found in the DB.",)

    p.add_argument("--limit_per_table", type=int, default=None, help="Optional row limit per table")
    p.add_argument("--include_cols", nargs="*", default=None, help="Optional list of columns to include in embedded text (default: all columns)",)
    return p.parse_args()


def list_user_tables(con) -> list[str]:
    # DuckDB-friendly and simple
    rows = con.execute("SHOW TABLES").fetchall()
    # rows are like [('table1',), ('table2',)]
    return [r[0] for r in rows]


def table_row_count(con, table: str) -> int:
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def row_to_text(table: str, rec: dict) -> str:
    label = rec.get("label") or rec.get("geo") or ""
    value = rec.get("raw_value") or rec.get("value")
    year = rec.get("year")
    parts = [
        f"Table: {table}",
        f"Location: {label}" if label else None,
        f"Measure: {rec.get('measure')}" if rec.get("measure") else None,
        f"Subject: {rec.get('subject')}" if rec.get("subject") else None,
        f"Stat: {rec.get('stat_type')}" if rec.get("stat_type") else None,
        f"Year: {year}" if year not in (None, "") else None,
        f"Value: {value}" if value not in (None, "") else None,
        f"Row: {rec.get('raw_row')}" if rec.get("raw_row") else None,
        f"Column: {rec.get('raw_col')}" if rec.get("raw_col") else None,
    ]
    return " | ".join([p for p in parts if p])


def build_documents_from_duckdb(duckdb_path: Path, persist_dir: Path, limit_per_table: int | None = None) -> list[Document]:
    duckdb_path = Path(duckdb_path).expanduser().resolve()
    persist_dir = Path(persist_dir).expanduser().resolve()

    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB file not found: {duckdb_path}")

    con = duckdb.connect(str(duckdb_path))
    try:
        tables = list_user_tables(con)
        if not tables:
            raise ValueError(f"No tables found in DuckDB: {duckdb_path}")

        docs: list[Document] = []
        for t in tables:
            lim = f" LIMIT {int(limit_per_table)}" if limit_per_table else ""
            df = con.execute(f"SELECT * FROM {t}{lim}").fetchdf()
            for _, row in df.iterrows():
                rec = row.to_dict()

                # Stable id: prefer row_id if present, else fallback
                rid = rec.get("row_id")
                if rid is None:
                    rid = f"{t}:{rec.get('source_file')}:{rec.get('orig_row_id')}:{rec.get('orig_col_id')}"

                text = row_to_text(t, rec)
                if not text.strip():
                    continue

                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "duckdb",
                            "duckdb_file": duckdb_path.name,
                            "table": t,
                            "row_id": rid,
                            "source_file": rec.get("source_file"),
                            "label": rec.get("label") or rec.get("geo"),
                            "measure": rec.get("measure"),
                            "subject": rec.get("subject"),
                            "stat_type": rec.get("stat_type"),
                            "year": rec.get("year"),
                            "value": rec.get("value"),
                            "raw_value": rec.get("raw_value"),
                            "orig_row_id": rec.get("orig_row_id"),
                            "orig_col_id": rec.get("orig_col_id"),
                        },
                    )
                )

        if not docs:
            # This is the “it created nothing” scenario—fail loudly
            raise ValueError(
                f"Discovered {len(tables)} tables but produced 0 documents. "
                f"Check table schemas and the SELECT queries."
            )
        print(f"[duckdb_index] Indexing {len(docs)} rows from {len(tables)} tables into: {persist_dir}")
        return docs

    finally:
        con.close()


def build_duckdb_index(duckdb_path: str | Path, persist_dir: str | Path, limit_per_table: Optional[int] = None) -> None:
    docs = build_documents_from_duckdb(duckdb_path=duckdb_path, persist_dir=persist_dir, limit_per_table=limit_per_table)
    persist_dir = Path(persist_dir).expanduser().resolve()
    index = VectorStoreIndex.from_documents(docs)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))


def main():
    args = parse_args()

    build_duckdb_index(
        duckdb_path=args.duckdb_path,
        persist_dir=args.persist_dir,
        limit_per_table=args.limit_per_table
    )
    print(f"[duckdb_index] Done. Persisted to: {args.persist_dir}")


if __name__ == "__main__":
    main()