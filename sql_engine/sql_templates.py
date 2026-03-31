# sql_engine/sql_templates.py
from __future__ import annotations
from typing import Any, Dict, List, Tuple


def cell_lookup_sql(q: Dict[str, Any], table_name: str) -> Tuple[str, List[Any]]:
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
      FROM {table_name}
      WHERE {" AND ".join(where)}
      ORDER BY
        (year IS NULL) ASC, year DESC NULLS LAST,
        row_id ASC
      LIMIT 1
    """
    return sql, params

def aggregation_sql(q: Dict[str, Any], table_name: str) -> Tuple[str, List[Any]]:
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
            FROM {table_name}
            WHERE {where_clause};
        """
    else:
        sql = f"""
            SELECT
                {op}(value) AS value,
                COUNT(*) AS n
            FROM {table_name}
            WHERE {where_clause};
        """

    return sql, params


ALLOWED_ORDER = {"ASC", "DESC"}
ALLOWED_SELECT = {"row", "compact"}

def row_filter_sql(q: Dict[str, Any], table_name: str) -> Tuple[str, List[Any]]:
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
        FROM {table_name}
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

ALLOWED_CHART_TYPES = {"categorical", "time_series", "compare_labels"}
ALLOWED_SORT = {"x_asc", "x_desc", "y_asc", "y_desc"}

def chart_request_sql(q: Dict[str, Any], table_name: str) -> Tuple[str, List[Any]]:
    chart_type = (q.get("chart_type") or "").lower()
    if chart_type not in ALLOWED_CHART_TYPES:
        raise ValueError(f"chart_type must be one of {sorted(ALLOWED_CHART_TYPES)}")

    filters = q.get("filters") or {}
    required = ["label", "subject", "stat_type"]
    missing = [k for k in required if not filters.get(k)]
    if missing:
        raise ValueError(f"Missing required filters for chart_request: {missing}")
    where: List[str] = []
    params: List[Any] = []

    # ---- Exact match filters ----
    for field in ["label", "subject", "stat_type", "unit", "source_file", "year"]:
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

    # Always drop null y-values
    where.append("value IS NOT NULL")

    where_clause = " AND ".join(where) if where else "TRUE"

    # ---- Choose x field based on chart_type (deterministic defaults) ----
    if chart_type == "categorical":
        x_expr = "measure"
        x_is_numeric = False
    elif chart_type == "time_series":
        x_expr = "year"
        x_is_numeric = True
        where.append("year IS NOT NULL")  # ensure year exists
        where_clause = " AND ".join(where)
    else:  # compare_labels
        x_expr = "label"
        x_is_numeric = False

    # ---- Sorting ----
    sort = (q.get("sort") or "x_asc").lower()
    if sort not in ALLOWED_SORT:
        raise ValueError(f"sort must be one of {sorted(ALLOWED_SORT)}")

    if sort == "x_asc":
        order_by = f"{x_expr} ASC, row_id ASC"
    elif sort == "x_desc":
        order_by = f"{x_expr} DESC, row_id ASC"
    elif sort == "y_asc":
        order_by = f"value ASC, {x_expr} ASC, row_id ASC"
    else:  # y_desc
        order_by = f"value DESC, {x_expr} ASC, row_id ASC"

    limit = int(q.get("limit", 500))
    if limit <= 0:
        raise ValueError("limit must be >= 1")

    sql = f"""
        SELECT
            {x_expr} AS x,
            value AS y,
            source_file,
            row_id
        FROM {table_name}
        WHERE {where_clause}
        ORDER BY {order_by}
        LIMIT {limit}
    """
    return sql, params