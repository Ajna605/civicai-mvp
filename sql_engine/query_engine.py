# sql_engine/query_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import duckdb
from sql_engine.retrieval.table_router import TableRouter
from sql_engine.llm_utils.query_guards import resolve_measure_override
from sql_engine.llm_utils.param_gen import llm_make_params, load_metadata
from sql_engine.llm_utils.validator import validate_measure_group_consistency, enforce_deterministic_measures, validate_cell_lookup
from sql_engine.llm_utils.llm_settings import build_answer_llm, llm_verbalize_answer
from sql_engine import sql_templates as T
from analytics.runtime import AnalyticsRuntime

TOP_K = 20
TOP_TABLES = 3
MAX_REPAIRS = 3
ANS_LLM = build_answer_llm()

def run_structured_query(db_path: Path, category: str, query: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    con = duckdb.connect(str(Path(db_path).resolve()))

    if category == "cell_lookup":
        sql, params = T.cell_lookup_sql(query, table_name)
        print("SQL", "PARAMS")
        print(sql, params)
        rows = con.execute(sql, params).fetchall()

        con.close()
        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        value, source_file, row_id = rows[0]
        return {
            "ok": True,
            "category": category,
            "table_name": table_name,
            "query": query,
            "result": {"value": value},
            "provenance": {"source_file": source_file, "row_id": row_id},
            "debug": {"sql": sql, "params": params}
        }

    if category == "aggregation":
        sql, params = T.aggregation_sql(query, table_name)
        row = con.execute(sql, params).fetchone()
        con.close()

        if row is None:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}

        value, n_rows = row

        op = (query.get("op") or "").lower()

        # Deterministic NO_MATCH handling
        if n_rows == 0:
            if op == "count":
                value = 0
            else:
                return {"ok": False, "error": "no_match", "sql": sql, "params": params}

        return {
            "ok": True,
            "category": category,
            "table_name": table_name,
            "query": query,
            "result": {"value": value, "n": n_rows, "op": op},
            "provenance": {"note": "aggregation", "n": n_rows},
            "debug": {"sql": sql, "params": params}
        }

    if category == "row_filter":
        sql, params = T.row_filter_sql(query, table_name)
        rows = con.execute(sql, params).fetchall()
        con.close()

        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}

        select = (query.get("select") or "row").lower()

        items = [{"label": r[0], "measure": r[1], "value": r[2], "source_file": r[3], "row_id": r[4]}
                for r in rows]

        provenance_rows = [{"source_file": r[3], "row_id": r[4]} for r in rows]

        return {
            "ok": True,
            "category": category,
            "table_name": table_name,
            "query": query,
            "result": {"rows": items},
            "provenance": {"top_rows": provenance_rows},
            "debug": {"sql": sql, "params": params}
        }

    if category == "chart_request":
        sql, params = T.chart_request_sql(query, table_name)
        rows = con.execute(sql, params).fetchall()
        con.close()
        if not rows:
            return {"ok": False, "error": "no_match", "sql": sql, "params": params}
        points = [{"x": r[0], "y": r[1]} for r in rows]
        prov = [{"source_file": r[2], "row_id": r[3]} for r in rows[:25]]  # cap for logging
        return {
            "ok": True,
            "category": category,
            "table_name": table_name,
            "query": query,
            "result": {"points": points},
            "provenance": {"sample": prov, "points": len(points)},
            "debug": {"sql": sql, "params": params}
        }

    con.close()
    return {"ok": False, "error": f"unknown_category:{category}"}

DUCKDB_PATH = "storage/duckdb/unaris.duckdb"
# find which table is relevant
def query_sql(query: str, index_path:str):
    
    router = TableRouter(index_dir=index_path, similarity_top_k=TOP_K)
    ranked = router.route(query, top_tables=TOP_TABLES)

    if not ranked:
        print("[table_router] No table candidates found.")
        return

    # for i, r in enumerate(ranked, 1):
    #     print(f"  {i}. {r.table}  hits={r.hits}  score_sum={r.score_sum:.4f}  avg={r.avg:.4f}")
    
    table_name = ranked[0].table
    meta = load_metadata(table_name)
    measure_override = resolve_measure_override(query, meta)
    rec: Dict[str, Any] = {
    "question": query
    }
    try:
        if measure_override and "force_query" in measure_override:
            pred = {
                "category": measure_override["force_category"],
                "query": measure_override["force_query"]}
        else:
            pred = llm_make_params(query, meta, max_repairs=MAX_REPAIRS, constraints=measure_override)
            # for cases with measure group
            pred = enforce_deterministic_measures(pred, query, meta)
            rec["pred"] = pred
            
            pred_cat = pred.get("category")

            if pred_cat == "cell_lookup":
                validate_cell_lookup(pred, meta)

            pred_query = pred.get("query")
            # Most relevant table
            pred["table"] = table_name
            pred["table_candidates"] = [r.table for r in ranked]
            # basic JSON/schema presence
            schema_pass = isinstance(pred_cat, str) and isinstance(pred_query, dict)

            # semantic validation: measure_group consistency
            group_ok, group_reason = validate_measure_group_consistency(pred, meta)

            rec["schema_pass"] = schema_pass and group_ok
            rec["schema_reason"] = "ok" if (schema_pass and group_ok) else (
                "missing_category_or_query" if not schema_pass else group_reason
            )

    except Exception as e:
            rec["error"] = str(e)
            rec["schema_pass"] = False
            rec["failure_type"] = "exception"

    result = run_structured_query(DUCKDB_PATH, pred["category"], pred["query"], table_name)
    # print("result", result)
    runtime = AnalyticsRuntime(out_dir="storage/charts")
    chart_result = runtime.render_from_result(result)

    answer = llm_verbalize_answer(ANS_LLM, result)
    return {"answer": answer}



