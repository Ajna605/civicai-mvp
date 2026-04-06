### For loaded CSV queries
# sql_engine/query_engine.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from sql_engine.llm_utils.llm_settings import build_answer_llm
from sql_engine.llm_utils.param_gen import load_metadata
from analytics.runtime import AnalyticsRuntime
from sql_engine.analytics_sql.param_gen import llm_make_params
from sql_engine.retrieval.table_router import TableRouter
from sql_engine.analytics_sql.llm_helper import llm_payload_from_ranking, ANALYTICS_PROMPT, llm_verbalize_analytics_summary
from analytics.time_forecast import forecast, to_chart_request_time_series_with_linear_forecast
import duckdb


# name=sql_engine/analytics_sql/executor.py (example path; paste into your file where run_structured_query lives)

def run_structured_query(db_path: Path, category: str, query: Dict[str, Any], table_name: str, meta: dict) -> Dict[str, Any]:
    con = duckdb.connect(str(Path(db_path).resolve()))

    if category == "analytics_rank":
        # --- required fields in NEW schema ---
        geo = query.get("geo")
        if not geo:
            return {"ok": False, "error": "missing_geo"}

        section = query.get("section")
        if not section:
            return {"ok": False, "error": "missing_section"}

        subject = query.get("subject")
        if not subject:
            return {"ok": False, "error": "missing_subject"}

        # table_name should come from query in the new schema.
        # (You also have table_name as a function arg; prefer query value if present.)
        table_name2 = query.get("table_name") or table_name
        if not table_name2:
            return {"ok": False, "error": "missing_table_name"}

        stat_type = query.get("stat_type") or "Estimate"
        if stat_type not in ("Estimate", "Margin of Error"):
            return {"ok": False, "error": "invalid_stat_type", "details": stat_type}

        # deterministic grouping based on your dataset
        group_dim = "measure"  # groups like "White alone" live in measure

        # time filtering
        time = query.get("time") or {}
        years = time.get("years") or []
        if isinstance(years, list) and len(years) == 2:
            year_start, year_end = int(years[0]), int(years[1])
        else:
            year_start, year_end = None, None

        order = (query.get("order") or "desc").lower()
        if order not in ("asc", "desc"):
            return {"ok": False, "error": "invalid_order", "details": order}

        limit = query.get("limit")
        limit = 10 if limit is None else int(limit)
        if limit <= 0:
            return {"ok": False, "error": "invalid_limit", "details": limit}

        # --- IMPORTANT: section filtering ---
        # Your table must have *some* column that stores the Excel section header.
        # In your earlier code you mentioned raw_row/raw_col; many ingestions store the header in raw_row.
        #
        # I’ll implement it as: section matches raw_row EXACTLY (case-insensitive).
        # If your section is stored in raw_col instead, switch raw_row -> raw_col.
        # assume you pass meta into run_structured_query or can access it globally
        section = query["section"]
        subject = query["subject"]

        # get allowed groups for this section (deterministic)
        group_values = meta.get("measure_groups", {}).get(section)
        if not group_values:
            return {"ok": False, "error": "unknown_section_or_empty_group", "details": section}

        where = []
        params = []

        where.append("value IS NOT NULL")
        where.append("(stat_type IS NULL OR stat_type = ?)")
        params.append(stat_type)

        where.append("LOWER(geo) LIKE ?")
        params.append(f"%{geo.lower()}%")

        where.append("subject = ?")
        params.append(subject)

        # Filter to group rows under the section:
        placeholders = ",".join(["?"] * len(group_values))
        where.append(f"measure IN ({placeholders})")
        params.extend(group_values)

        # year filter
        if year_start is not None and year_end is not None:
            where.append("year BETWEEN ? AND ?")
            params.extend([year_start, year_end])

        sql = f"""
        SELECT
        geo, subject, label, measure, value, year, stat_type,
        source_file, orig_row_id, orig_col_id, raw_row, raw_col, row_id
        FROM {table_name2}
        WHERE {" AND ".join(where)}
        """

        rows = con.execute(sql, params).fetchall()

        # rank groups by value per year (timeseries) or overall; for now: return evidence (like your old behavior)
        items = []
        provenance_rows = []
        for r in rows:
            item = {
                "geo": r[0],
                "subject": r[1],
                "label": r[2],
                "measure": r[3],  # group value
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
                "table_name": table_name2,
                "source_file": r[7],
                "orig_row_id": r[8],
                "orig_col_id": r[9],
                "year": r[5],
                "row_id": r[12],
            })

        debug_queries = [{
            "table": table_name2,
            "sql": sql,
            "params": params,
            "n": len(items),
            "group_dim": group_dim,
            "section_col": section,
        }]

        return {
            "ok": True,
            "category": category,
            "table_name": table_name2,
            "query": query,
            "result": {
                "group_dim": group_dim,
                "evidence": items,
                # Later you can compute ranking here using order/limit.
            },
            "provenance": {"evidence_rows": provenance_rows},
            "debug": {"queries": debug_queries},
        }

    # ... other categories ...
    
DUCKDB_PATH = "storage/duckdb/unaris.duckdb"
MAX_REPAIRS = 3

ANS_LLM = None
def get_ans_llm():
    global ANS_LLM
    if ANS_LLM is None:
        ANS_LLM = build_answer_llm()
        print(ANS_LLM._model.device)
    return ANS_LLM

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
    # print("pred", pred)
    # for cases with measure group
    rec["pred"] = pred
    
    pred_cat = pred.get("category")

    pred_query = pred.get("query")
    # Most relevant table
    pred["table"] = table_name
    pred["table_candidates"] = [r.table for r in ranked]
    # basic JSON/schema presence
    schema_pass = isinstance(pred_cat, str) and isinstance(pred_query, dict)

    result = run_structured_query(DUCKDB_PATH, pred["category"], pred["query"], table_name, meta)

    fc = forecast(
        items=result["result"]["evidence"],
        subject=result["query"]["subject"],
        section=result["query"]["section"],
        limit=result["query"]["limit"],
    )

    runtime = AnalyticsRuntime(out_dir="storage/charts")
    chart_obj = to_chart_request_time_series_with_linear_forecast(
    ranking= fc["ranking"],  # or full ranking for all races
    measure_group="RACE AND HISPANIC OR LATINO ORIGIN",
    title="Coral Gables: Percent Uninsured trend by race (2015–2024)",
    )

    chart_result = runtime.render_from_result(chart_obj)

    payload = llm_payload_from_ranking(
    geo=result["query"]["geo"],
    section=result["query"]["section"],
    subject=result["query"]["subject"],
    years=result["query"]["time"]["years"],
    stat_type=result["query"]["stat_type"],
    ranking=fc["ranking"],
    risk_definition="Higher projected Percent Uninsured next year (latest_value + slope_per_year).",
    top_k=5,
    chart_paths={"trend_png": str(chart_result["chart_path"])},)
    
    text = llm_verbalize_analytics_summary(get_ans_llm(), payload, prompt_template=ANALYTICS_PROMPT, user_question=query)

    return {"answer": text}

