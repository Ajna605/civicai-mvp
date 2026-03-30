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


def duckdb_connect(duckdb_path: str | Path):
    duckdb_path = Path(duckdb_path)
    if duckdb_path.exists() and duckdb_path.is_dir():
        raise ValueError(f"duckdb_path must be a file, got directory: {duckdb_path}")
    return duckdb.connect(str(duckdb_path))


def list_tables(con) -> List[str]:
    """
    Return user tables in the current DuckDB database.

    Uses information_schema (stable). Filters out internal/system schemas.
    """
    rows = con.execute(
        """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('information_schema', 'pg_catalog')
        ORDER BY table_schema, table_name
        """
    ).fetchall()

    # Return qualified names if needed. If you only use 'main', you can strip schema.
    tables = []
    for schema, name in rows:
        if schema == "main":
            tables.append(name)
        else:
            tables.append(f"{schema}.{name}")
    return tables


def row_to_text(table: str, row: dict, include_cols: Optional[Iterable[str]] = None) -> str:
    if include_cols:
        cols = [c for c in include_cols if c in row]
    else:
        cols = list(row.keys())

    parts = [f"{c}={row.get(c)}" for c in cols]
    return f"[table={table}] " + " | ".join(parts)


def duckdb_rows_to_documents(
    duckdb_path: str | Path,
    tables: Optional[List[str]] = None,
    limit_per_table: Optional[int] = None,
    include_cols: Optional[List[str]] = None,
) -> List[Document]:
    con = duckdb_connect(duckdb_path)
    try:
        if not tables:
            tables = list_tables(con)
            if not tables:
                raise ValueError("No tables found in DuckDB database.")

        docs: List[Document] = []
        for table in tables:
            print("TABLE", table)
            lim = f" LIMIT {int(limit_per_table)}" if limit_per_table else ""
            df = con.execute(f"SELECT * FROM {table}{lim}").fetchdf()
            for i, row in df.iterrows():
                row_dict = row.to_dict()
                text = row_to_text(table, row_dict, include_cols=include_cols)
                if not text.strip():
                    continue
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source": "duckdb",
                            "table": table,
                            "row_index": int(i),
                        },
                    )
                )
        return docs
    finally:
        con.close()


def build_duckdb_index(
    duckdb_path: str | Path,
    persist_dir: str | Path,
    tables: Optional[List[str]] = None,
    limit_per_table: Optional[int] = None,
    include_cols: Optional[List[str]] = None,
) -> None:
    docs = duckdb_rows_to_documents(
        duckdb_path=duckdb_path,
        tables=tables,
        limit_per_table=limit_per_table,
        include_cols=include_cols,
    )
    index = VectorStoreIndex.from_documents(docs)
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))


def main():
    args = parse_args()

    build_duckdb_index(
        duckdb_path=args.duckdb_path,
        persist_dir=args.persist_dir,
        tables=args.tables,
        limit_per_table=args.limit_per_table,
        include_cols=args.include_cols,
    )
    print(f"[duckdb_index] Done. Persisted to: {args.persist_dir}")


if __name__ == "__main__":
    main()