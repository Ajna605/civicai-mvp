# sql_engine/sql_templates.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple

TABLE = "facts"  # change if needed

def cell_lookup_sql(q: Dict[str, Any]) -> Tuple[str, List[Any]]:
    where = ["label = ?", "measure = ?"]
    params: List[Any] = [q["label"], q["measure"]]

    if "year" in q and q["year"] is not None:
        where.append("year = ?")
        params.append(q["year"])

    sql = f"""
      SELECT value, source_file, row_id
      FROM {TABLE}
      WHERE {" AND ".join(where)}
      ORDER BY
        (year IS NULL) ASC, year DESC NULLS LAST,
        row_id ASC
      LIMIT 1
    """
    return sql, params

def aggregation_sql(q: Dict[str, Any]) -> Tuple[str, List[Any]]:
    agg = q.get("agg", "SUM").upper()
    if agg not in {"SUM","AVG","MIN","MAX","COUNT"}:
        raise ValueError(f"Unsupported agg: {agg}")

    where = ["measure = ?"]
    params: List[Any] = [q["measure"]]

    if "year" in q and q["year"] is not None:
        where.append("year = ?")
        params.append(q["year"])

    if "label_in" in q and q["label_in"]:
        placeholders = ",".join(["?"] * len(q["label_in"]))
        where.append(f"label IN ({placeholders})")
        params.extend(q["label_in"])

    sql = f"""
      SELECT {agg}(value) AS value, COUNT(*) AS n_rows
      FROM {TABLE}
      WHERE {" AND ".join(where)}
    """
    return sql, params

def row_filter_sql(q: Dict[str, Any]) -> Tuple[str, List[Any]]:
    order = q.get("order", "DESC").upper()
    if order not in {"ASC","DESC"}:
        raise ValueError("order must be ASC or DESC")
    limit = int(q.get("limit", 1))

    where = ["measure = ?"]
    params: List[Any] = [q["measure"]]

    if "year" in q and q["year"] is not None:
        where.append("year = ?")
        params.append(q["year"])

    sql = f"""
      SELECT label, value, source_file, row_id
      FROM {TABLE}
      WHERE {" AND ".join(where)} AND value IS NOT NULL
      ORDER BY value {order}, row_id ASC
      LIMIT {limit}
    """
    return sql, params

def chart_request_sql(q: Dict[str, Any]) -> Tuple[str, List[Any]]:
    # For now: x must be "year"
    where = ["label = ?", "measure = ?"]
    params: List[Any] = [q["label"], q["measure"]]

    if "year_range" in q and q["year_range"]:
        y0, y1 = q["year_range"]
        where.append("year BETWEEN ? AND ?")
        params.extend([y0, y1])

    sql = f"""
      SELECT year AS x, value AS y, source_file, row_id
      FROM {TABLE}
      WHERE {" AND ".join(where)} AND year IS NOT NULL
      ORDER BY year ASC, row_id ASC
    """
    return sql, params