# # tabular/duckdb_loader.py
# from pathlib import Path
# import duckdb

# FACTS_SCHEMA_SQL = """
# CREATE TABLE IF NOT EXISTS facts (
#   row_id BIGINT,
#   source_file TEXT,
#   label TEXT,
#   measure TEXT,
#   value DOUBLE,
#   raw_value TEXT,
#   year INTEGER,
#   unit TEXT,
#   geo TEXT,
#   subject TEXT,       
#   stat_type TEXT,
#   raw_row TEXT,
#   raw_col TEXT,
#   orig_row_id INTEGER,
#   orig_col_id INTEGER,     
# );
# """

# def build_duckdb(db_path: Path, facts_jsonl: Path, rebuild: bool = False) -> None:
#     db_path.parent.mkdir(parents=True, exist_ok=True)
#     con = duckdb.connect(str(db_path))

#     if rebuild:
#         con.execute("DROP TABLE IF EXISTS facts;")

#     con.execute(FACTS_SCHEMA_SQL)

#     # Load JSONL into a temp relation, then insert with row_id assignment
#     con.execute(f"""
#         CREATE OR REPLACE TEMP TABLE facts_in AS
#         SELECT * FROM read_json_auto('{facts_jsonl.as_posix()}');
#     """)

#     # Assign row_id deterministically
#     con.execute("""
#         INSERT INTO facts
#         SELECT
#           row_number() over (order by raw_row, raw_col),
#           source_file,
#           label,        
#           measure,      
#           value,        
#           CAST(raw_value AS TEXT) AS raw_value,       
#           year,        
#           unit,        
#           geo,        
#           subject,        
#           stat_type,        
#           raw_row,
#           raw_col,
#           orig_row_id,
#           orig_col_id,
#         FROM facts_in;
#     """)

#     con.close()



import re
from pathlib import Path
import duckdb
from utils.text_utils import infer_table_name


def create_facts_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
          row_id BIGINT,
          source_file TEXT,
          label TEXT,
          measure TEXT,
          value DOUBLE,
          raw_value TEXT,
          year INTEGER,
          unit TEXT,
          geo TEXT,
          subject TEXT,
          stat_type TEXT,
          raw_row TEXT,
          raw_col TEXT,
          orig_row_id INTEGER,
          orig_col_id INTEGER
        );
    """)

def build_duckdb(
    db_path: Path,
    rebuild: bool = False,
    table_name: str | None = None,
) -> list[str]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    normalized_dir = Path("data/normalized/csv")

    jsonl_files = sorted(normalized_dir.glob("*.jsonl"))
    print(jsonl_files)

    if not jsonl_files:
        raise FileNotFoundError(f"No .jsonl files found in {normalized_dir}")

    created_tables = []

    try:
        for jsonl_path in jsonl_files:
            table_name = infer_table_name(jsonl_path)

            if rebuild:
                con.execute(f"DROP TABLE IF EXISTS {table_name};")

            create_facts_table(con, table_name)

            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE facts_in AS
                SELECT * FROM read_json_auto('{jsonl_path.as_posix()}');
            """)

            current_max = con.execute(
                f"SELECT COALESCE(MAX(row_id), 0) FROM {table_name}"
            ).fetchone()[0]

            con.execute(f"""
                INSERT INTO {table_name}
                SELECT
                    {current_max} + row_number() OVER (
                        ORDER BY source_file, orig_row_id, orig_col_id, raw_row, raw_col
                    ) AS row_id,
                    source_file,
                    label,
                    measure,
                    value,
                    CAST(raw_value AS TEXT) AS raw_value,
                    year,
                    unit,
                    geo,
                    subject,
                    stat_type,
                    raw_row,
                    raw_col,
                    orig_row_id,
                    orig_col_id
                FROM facts_in;
            """)

            created_tables.append(table_name)
            print(f"{table_name} has been created.")

        return created_tables

    finally:
        con.close()