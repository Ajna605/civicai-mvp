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

    if q.get("subject") is not None:
        where.append("subject = ?")
        params.append(q["subject"])

    if q.get("stat_type") is not None:
        where.append("stat_type = ?")
        params.append(q["stat_type"])

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
    op = q.get("op", "").upper()

    if op not in {"SUM", "AVG", "MIN", "MAX", "COUNT"}:
        raise ValueError(f"Unsupported op: {op}")

    filters = q.get("filters", {})
    where = []
    params: List[Any] = []

    # ---- Exact match filters ----
    for field in ["label", "subject", "stat_type", "year", "unit", "source_file"]:
        if field in filters and filters[field] is not None:
            where.append(f"{field} = ?")
            params.append(filters[field])

    # ---- Measure selector (exactly one allowed) ----
    measure = filters.get("measure")
    measure_contains = filters.get("measure_contains")
    measures_in = filters.get("measures_in")

    selectors = [x is not None for x in [measure, measure_contains, measures_in]]
    if sum(selectors) != 1:
        raise ValueError("Exactly one of measure, measure_contains, or measures_in must be provided.")

    if measure:
        where.append("measure = ?")
        params.append(measure)

    elif measure_contains:
        where.append("measure ILIKE ?")
        params.append(f"%{measure_contains}%")

    elif measures_in:
        placeholders = ",".join(["?"] * len(measures_in))
        where.append(f"measure IN ({placeholders})")
        params.extend(measures_in)

    # Always exclude NULL values
    where.append("value IS NOT NULL")

    where_clause = " AND ".join(where)

    # COUNT is special: no need to aggregate value
    if op == "COUNT":
        sql = f"""
            SELECT
                COUNT(*) AS value,
                COUNT(*) AS n
            FROM {TABLE}
            WHERE {where_clause};
        """
    else:
        sql = f"""
            SELECT
                {op}(value) AS value,
                COUNT(*) AS n
            FROM {TABLE}
            WHERE {where_clause};
        """

    return sql, params


ALLOWED_ORDER = {"ASC", "DESC"}
ALLOWED_SELECT = {"row", "compact"}

def row_filter_sql(q: Dict[str, Any]) -> Tuple[str, List[Any]]:
    order = (q.get("order") or "DESC").upper()
    if order not in ALLOWED_ORDER:
        raise ValueError("order must be ASC or DESC")

    limit = int(q.get("limit", 1))
    if limit <= 0:
        raise ValueError("limit must be >= 1")

    offset = int(q.get("offset", 0))
    if offset < 0:
        raise ValueError("offset must be >= 0")

    filters = q.get("filters") or {}
    where: List[str] = []
    params: List[Any] = []

    # ---- Exact match filters (deterministic) ----
    for field in ["label", "subject", "stat_type", "year", "unit", "source_file"]:
        if field in filters and filters[field] is not None:
            where.append(f"{field} = ?")
            params.append(filters[field])

    # ---- Measure selector: exactly one ----
    measure = filters.get("measure")
    measure_contains = filters.get("measure_contains")
    measures_in = filters.get("measures_in")

    selectors = [measure is not None, measure_contains is not None, measures_in is not None]
    if sum(selectors) != 1:
        raise ValueError("Exactly one of measure, measure_contains, or measures_in must be provided.")

    if measure is not None:
        where.append("measure = ?")
        params.append(measure)
    elif measure_contains is not None:
        where.append("measure ILIKE ?")
        params.append(f"%{measure_contains}%")
    else:
        if not isinstance(measures_in, list) or not measures_in:
            raise ValueError("measures_in must be a non-empty list.")
        placeholders = ",".join(["?"] * len(measures_in))
        where.append(f"measure IN ({placeholders})")
        params.extend(measures_in)

    # Always exclude null values
    where.append("value IS NOT NULL")

    where_clause = " AND ".join(where) if where else "TRUE"

    # Return measure because row_filter questions usually need "which row"
    # Deterministic tie-breaks (after value): year asc, measure asc, row_id asc
    sql = f"""
        SELECT
            label,  measure,  value,  source_file,  row_id
        FROM {TABLE}
        WHERE {where_clause}
        ORDER BY
            value {order},
            year ASC,
            measure ASC,
            label ASC,
            subject ASC,
            stat_type ASC,
            row_id ASC
        LIMIT {limit}
        OFFSET {offset}
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