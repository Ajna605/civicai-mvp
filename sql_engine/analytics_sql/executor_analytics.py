### For loaded CSV queries
# sql_engine/query_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from sql_engine.llm_utils.llm_settings import build_answer_llm, llm_verbalize_answer
from sql_engine.llm_utils.param_gen import load_metadata
from analytics.runtime import AnalyticsRuntime
from sql_engine.analytics_sql.param_gen import llm_make_params
from sql_engine.retrieval.table_router import TableRouter
import duckdb


def run_structured_query(db_path: Path, category: str, query: Dict[str, Any], table_name: str) -> Dict[str, Any]:
    con = duckdb.connect(str(Path(db_path).resolve()))

    if category == "analytics_rank":
        geo = query.get("geo")
        if not geo:
            return {"ok": False, "error": "missing_geo"}

        stat_type = query.get("stat_type") or "Estimate"

        group_dim = query.get("group_dim") or "subject"
        if group_dim not in ("subject", "label"):
            return {"ok": False, "error": "invalid_group_dim", "details": group_dim}

        time = query.get("time") or {}
        years = time.get("years") or []
        if len(years) == 2:
            year_start, year_end = int(years[0]), int(years[1])
        else:
            # fallback: no year filter
            year_start, year_end = None, None

        inputs = query.get("inputs") or []
        if not inputs:
            return {"ok": False, "error": "no_inputs"}

        evidence_by_metric = {}
        provenance_rows = []
        debug_queries = []

        for inp in inputs:
            metric_key = inp.get("metric_key")
            table_name = inp.get("table_name")   # in your schema, this is literally the table name
            match_any = inp.get("measure_match_any") or []

            if not metric_key or not table_name:
                return {"ok": False, "error": "bad_input", "details": inp}

            # Build the measure filter
            # We match against BOTH measure and raw_row to be more robust.
            like_clauses = []
            params = [stat_type, f"%{geo.lower()}%"]

            for term in match_any:
                t = f"%{str(term).lower()}%"
                like_clauses.append("(LOWER(measure) LIKE ? OR LOWER(raw_row) LIKE ?)")
                params.extend([t, t])

            measure_filter_sql = " OR ".join(like_clauses) if like_clauses else "TRUE"

            year_filter_sql = ""
            if year_start is not None and year_end is not None:
                year_filter_sql = " AND year BETWEEN ? AND ?"
                params.extend([year_start, year_end])

            sql = f"""
                SELECT
                geo,
                subject,
                label,
                measure,
                value,
                year,
                stat_type,
                source_file,
                orig_row_id,
                orig_col_id,
                raw_row,
                raw_col,
                row_id
                FROM {table_name}
                WHERE value IS NOT NULL
                AND (stat_type IS NULL OR stat_type = ?)
                AND LOWER(geo) LIKE ?
                {year_filter_sql}
                AND ({measure_filter_sql})
            """

            rows = con.execute(sql, params).fetchall()

            items = []
            for r in rows:
                item = {
                    "geo": r[0],
                    "subject": r[1],
                    "label": r[2],
                    "measure": r[3],
                    "value": r[4],
                    "year": r[5],
                    "stat_type": r[6],
                    "source_file": r[7],
                    "orig_row_id": r[8],
                    "orig_col_id": r[9],
                    "raw_row": r[10],
                    "raw_col": r[11],
                    "row_id": r[12],
                }
                items.append(item)

                provenance_rows.append({
                    "metric_key": metric_key,
                    "table_name": table_name,
                    "source_file": r[7],
                    "orig_row_id": r[8],
                    "orig_col_id": r[9],
                    "year": r[5],
                    "row_id": r[12],
                })

            evidence_by_metric[metric_key] = items
            debug_queries.append({"metric_key": metric_key, "table": table_name, "sql": sql, "params": params, "n": len(items)})

        # IMPORTANT: ranking/scoring is a separate step; for now return evidence
        # (you can compute later or do it here)
        return {
            "ok": True,
            "category": category,
            "table_name": None,
            "query": query,
            "result": {
                "evidence": evidence_by_metric,
                # later:
                # "ranking": ranking
            },
            "provenance": {"evidence_rows": provenance_rows},
            "debug": {"queries": debug_queries},
        }
    
DUCKDB_PATH = "storage/duckdb/unaris.duckdb"
MAX_REPAIRS = 3

# ANS_LLM = None
# def get_param_llm():
#     global ANS_LLM
#     if ANS_LLM is None:
#         ANS_LLM = build_answer_llm()
#         print(ANS_LLM._model.device)
#     return ANS_LLM

TOP_K = 20
TOP_TABLES = 3
def query_analytics(query: str, index_path:str):
    
    router = TableRouter(index_dir=index_path, similarity_top_k=TOP_K)
    ranked = router.route(query, top_tables=TOP_TABLES)

    if not ranked:
        print("[table_router] No table candidates found.")
        return

    table_name = ranked[0].table
    print("table name", table_name)

    meta = load_metadata(table_name + "_metadata.json")

    rec: Dict[str, Any] = {
    "question": query
    }
    # try:
    pred = llm_make_params(query, meta, max_repairs=MAX_REPAIRS, constraints=None)
    print("pred", pred)
    # for cases with measure group
    rec["pred"] = pred
    
    pred_cat = pred.get("category")

    pred_query = pred.get("query")
    # Most relevant table
    pred["table"] = table_name
    pred["table_candidates"] = [r.table for r in ranked]
    # basic JSON/schema presence
    schema_pass = isinstance(pred_cat, str) and isinstance(pred_query, dict)

    # except Exception as e:
    #         rec["error"] = str(e)
    #         rec["schema_pass"] = False
    #         rec["failure_type"] = "exception"

    # result = run_structured_query(DUCKDB_PATH, pred["category"], pred["query"], table_name)
    # # Generate and save charts
    # runtime = AnalyticsRuntime(out_dir="storage/charts")
    # chart_result = runtime.render_from_result(result)

    # answer = llm_verbalize_answer(ANS_LLM, result)
    return {"answer": pred}



