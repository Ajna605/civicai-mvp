from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict

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
    def __init__(
        self,
        out_dir: str | Path = DEFAULT_OUT_DIR,
    ) -> None:
        self.out_dir = Path(out_dir)

    def render_from_result(self, obj: Dict[str, Any]) -> Dict[str, Any]:
        category = (obj or {}).get("category")
        query = (obj or {}).get("query") or {}
        points = ((obj.get("result") or {}).get("points")) or []

        if category != "chart_request":
            return self._error(f"category must be 'chart_request', got '{category}'")
        if not isinstance(query, dict):
            return self._error("query must be a dict")

        chart_type = (query.get("chart_type") or "categorical").lower()
        if chart_type not in _CHART_TYPES:
            return self._error(f"unsupported chart_type: '{chart_type}'")

        viz_type = (query.get("viz_type") or "bar").lower()
        if viz_type not in {"bar", "pie", "line"}:
            return self._error(f"unsupported viz_type: '{viz_type}'")

        df = points_to_dataframe(points, chart_type)
        if df.empty:
            return self._error("no_data_after_shaping")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        chart_id = str(uuid.uuid4())[:8]
        mg_slug = (query.get("measure_group") or "chart").replace(" ", "_")[:30]
        out_path = self.out_dir / f"{chart_id}_{mg_slug}"

        # Render
        if chart_type == "categorical":
            if viz_type == "pie":
                png_path, html_path = renderers.render_categorical_pie(df, query, out_path)
            else:
                png_path, html_path = renderers.render_categorical_bar(df, query, out_path)

        elif chart_type == "time_series":
            # force line for time series (optional)
            png_path, html_path = renderers.render_time_series(df, query, out_path)

        else:  # compare_labels
            # pie doesn't really apply here; keep grouped bar
            png_path, html_path = renderers.render_compare_labels(df, query, out_path)

        chart_summary = {
            "chart_type": chart_type,
            "viz_type": viz_type,
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

    @staticmethod
    def _error(msg: str) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": msg,
            "chart_path": None,
            "html_path": None,
            "chart_summary": {},
        }