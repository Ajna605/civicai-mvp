# tabular/duckdb_loader.py
from pathlib import Path
import duckdb

FACTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts (
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
  orig_col_id INTEGER,     
);
"""

def build_duckdb(db_path: Path, facts_jsonl: Path, rebuild: bool = False) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    if rebuild:
        con.execute("DROP TABLE IF EXISTS facts;")

    con.execute(FACTS_SCHEMA_SQL)

    # Load JSONL into a temp relation, then insert with row_id assignment
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE facts_in AS
        SELECT * FROM read_json_auto('{facts_jsonl.as_posix()}');
    """)

    # Assign row_id deterministically
    con.execute("""
        INSERT INTO facts
        SELECT
          row_number() over (order by raw_row, raw_col),
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
          orig_col_id,
        FROM facts_in;
    """)

    con.close()