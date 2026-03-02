
# sql_engine/query_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict

import duckdb

from sql_engine import sql_templates as T

def run_structured_query(db_path: Path, category: str, query: Dict[str, Any]) -> Dict[str, Any]:
    con = duckdb.connect(str(db_path))

    if category == "cell_lookup":
        sql, params = T.cell_lookup_sql(query)
        rows = con.execute(sql, params).fetchall()
        con.close()
        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        value, source_file, row_id = rows[0]
        return {
            "ok": True,
            "data": {"value": value},
            "provenance": {"source_file": source_file, "row_id": row_id},
            "sql": sql,
            "params": params,
        }

    if category == "aggregation":
        sql, params = T.aggregation_sql(query)
        row = con.execute(sql, params).fetchone()
        con.close()
        if row is None:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        value, n_rows = row
        return {
            "ok": True,
            "data": {"value": value, "n_rows": n_rows},
            "provenance": {"note": "aggregation", "n_rows": n_rows},
            "sql": sql,
            "params": params,
        }

    if category == "row_filter":
        sql, params = T.row_filter_sql(query)
        rows = con.execute(sql, params).fetchall()
        con.close()
        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        items = [
            {"label": r[0], "value": r[1], "source_file": r[2], "row_id": r[3]}
            for r in rows
        ]
        return {
            "ok": True,
            "data": {"rows": items},
            "provenance": {"top_rows": [{"source_file": x["source_file"], "row_id": x["row_id"]} for x in items]},
            "sql": sql,
            "params": params,
        }

    if category == "chart_request":
        sql, params = T.chart_request_sql(query)
        rows = con.execute(sql, params).fetchall()
        con.close()
        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        points = [{"x": r[0], "y": r[1]} for r in rows]
        prov = [{"source_file": r[2], "row_id": r[3]} for r in rows[:25]]  # cap for logging
        return {
            "ok": True,
            "data": {"points": points},
            "provenance": {"sample": prov, "points": len(points)},
            "sql": sql,
            "params": params,
        }

    con.close()
    return {"ok": False, "error": f"unknown_category:{category}"}