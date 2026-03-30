"""AnalyticsRuntime – dispatches validated LLM output to chart rendering.

Typical usage::

    from analytics import AnalyticsRuntime

    runtime = AnalyticsRuntime(db_path="storage/duckdb/unaris.duckdb")
    result = runtime.run(obj, metadata)

    # result["ok"]           → True / False
    # result["chart_path"]   → pathlib.Path to the saved PNG file
    # result["html_path"]    → pathlib.Path to the saved HTML file (Plotly), or None
    # result["chart_summary"]→ dict with chart type, axes, row count, title
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from sql_engine.query_engine import run_structured_query
from analytics.data_shaper import points_to_dataframe
from analytics import renderers

logger = logging.getLogger(__name__)

DEFAULT_OUT_DIR = Path("outputs/charts")

_CHART_TYPES = {"categorical", "time_series", "compare_labels"}
_X_COLS: Dict[str, str] = {
    "categorical": "measure",
    "time_series": "year",
    "compare_labels": "label",
}


class AnalyticsRuntime:
    """Execute ``chart_request`` queries and produce chart artifacts.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file (e.g. ``storage/duckdb/unaris.duckdb``).
    out_dir:
        Directory where chart files are written.  Created if it does not exist.
        Defaults to ``outputs/charts``.
    """

    def __init__(
        self,
        db_path: str | Path,
        out_dir: str | Path = DEFAULT_OUT_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.out_dir = Path(out_dir)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def run(
        self,
        obj: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a validated LLM output object and render a chart.

        Parameters
        ----------
        obj:
            Validated LLM output ``{"category": "chart_request", "query": {...}}``.
        metadata:
            Optional metadata dict used for defensive measure validation.

        Returns
        -------
        dict
            ``ok`` (bool), ``chart_path`` (Path|None), ``html_path`` (Path|None),
            ``chart_summary`` (dict), and ``error`` (str) when *ok* is False.
        """
        category = (obj or {}).get("category")
        query = (obj or {}).get("query") or {}

        if category != "chart_request":
            return self._error(f"category must be 'chart_request', got '{category}'")

        if not isinstance(query, dict):
            return self._error("query must be a dict")

        chart_type = (query.get("chart_type") or "categorical").lower()
        if chart_type not in _CHART_TYPES:
            return self._error(f"unsupported chart_type: '{chart_type}'")

        # Defensive metadata check: silently drop any measure not in the
        # approved group so we never render with invented values.
        query = self._sanitize_measures(query, metadata)

        # Execute SQL via the existing engine
        sql_result = run_structured_query(self.db_path, "chart_request", query)
        if not sql_result.get("ok"):
            return self._error(sql_result.get("error", "sql_error"))

        points = (sql_result.get("data") or {}).get("points") or []
        df = points_to_dataframe(points, chart_type)

        if df.empty:
            return self._error("no_data_after_shaping")

        # Ensure output directory exists
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Build a unique output stem
        chart_id = str(uuid.uuid4())[:8]
        mg_slug = (query.get("measure_group") or "chart").replace(" ", "_")[:30]
        out_path = self.out_dir / f"{chart_id}_{mg_slug}"

        # Render
        if chart_type == "categorical":
            png_path, html_path = renderers.render_categorical_bar(df, query, out_path)
        elif chart_type == "time_series":
            png_path, html_path = renderers.render_time_series(df, query, out_path)
        else:
            png_path, html_path = renderers.render_compare_labels(df, query, out_path)

        chart_summary = {
            "chart_type": chart_type,
            "viz_type": query.get("viz_type", "bar"),
            "x_field": _X_COLS.get(chart_type, "x"),
            "y_field": "value",
            "row_count": len(df),
            "measure_group": query.get("measure_group"),
            "title": f"{chart_type} – {query.get('measure_group', '')}",
        }

        return {
            "ok": True,
            "chart_path": png_path,
            "html_path": html_path,
            "chart_summary": chart_summary,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": msg,
            "chart_path": None,
            "html_path": None,
            "chart_summary": {},
        }

    @staticmethod
    def _sanitize_measures(
        query: Dict[str, Any],
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Drop any measure_in values not in the metadata-approved group."""
        if metadata is None:
            return query
        filters = dict(query.get("filters") or {})
        measures_in = filters.get("measures_in") or []
        if not measures_in:
            return query
        mg = query.get("measure_group")
        groups = (metadata or {}).get("measure_groups") or {}
        allowed = set(groups.get(mg) or [])
        if not allowed:
            return query
        cleaned = [m for m in measures_in if m in allowed]
        bad = [m for m in measures_in if m not in allowed]
        if bad:
            logger.warning(
                "Dropping measures not in approved group '%s': %s", mg, bad
            )
        filters["measures_in"] = cleaned
        return dict(query, filters=filters)
