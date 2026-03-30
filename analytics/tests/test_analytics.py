"""Unit tests for the analytics package.

Tests cover:
1. Dispatcher behavior for chart_request (wrong category returns error)
2. Chart file creation (PNG + HTML) using a mock SQL engine
3. DataFrame shaping from sample SQL engine outputs
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable when running tests from any directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analytics import AnalyticsRuntime
from analytics.data_shaper import points_to_dataframe
from analytics.renderers import (
    render_categorical_bar,
    render_time_series,
    render_compare_labels,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CATEGORICAL_POINTS = [
    {"x": "White alone", "y": 15000},
    {"x": "Black or African American alone", "y": 3000},
    {"x": "Asian alone", "y": 2000},
    {"x": "Two or more races", "y": 1500},
]

TIME_SERIES_POINTS = [
    {"x": 2019, "y": 12000},
    {"x": 2020, "y": 11500},
    {"x": 2021, "y": 13000},
]

COMPARE_LABELS_POINTS = [
    {"x": "Coral Gables city, Florida", "y": 25000},
    {"x": "Miami city, Florida", "y": 80000},
]

CHART_QUERY = {
    "measure_group": "RACE AND HISPANIC OR LATINO ORIGIN",
    "chart_type": "categorical",
    "viz_type": "bar",
    "filters": {
        "label": "Coral Gables city, Florida",
        "subject": "Insured",
        "stat_type": "Estimate",
        "measures_in": [
            "White alone",
            "Black or African American alone",
            "Asian alone",
            "Two or more races",
        ],
    },
    "x": "measure",
    "y": "value",
}

SAMPLE_METADATA = {
    "measure_groups": {
        "RACE AND HISPANIC OR LATINO ORIGIN": [
            "White alone",
            "Black or African American alone",
            "Asian alone",
            "Two or more races",
            "Hispanic or Latino",
        ]
    }
}


# ---------------------------------------------------------------------------
# Helper: build a fake SQL engine result
# ---------------------------------------------------------------------------

def _make_sql_ok(points):
    return {"ok": True, "data": {"points": points}, "provenance": {}}


def _make_sql_err(error="no_match"):
    return {"ok": False, "error": error}


# ---------------------------------------------------------------------------
# 1. Dispatcher: wrong category
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_wrong_category_returns_error(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        result = runtime.run({"category": "aggregation", "query": {}})
        assert result["ok"] is False
        assert "chart_request" in result["error"]
        assert result["chart_path"] is None
        assert result["html_path"] is None
        assert result["chart_summary"] == {}

    def test_cell_lookup_returns_error(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        result = runtime.run({"category": "cell_lookup", "query": {}})
        assert result["ok"] is False

    def test_missing_category_returns_error(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        result = runtime.run({"query": {}})
        assert result["ok"] is False

    def test_unsupported_chart_type_returns_error(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        obj = {
            "category": "chart_request",
            "query": dict(CHART_QUERY, chart_type="histogram"),
        }
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_ok(CATEGORICAL_POINTS)):
            result = runtime.run(obj)
        assert result["ok"] is False
        assert "histogram" in result["error"]

    def test_sql_error_propagated(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        obj = {"category": "chart_request", "query": CHART_QUERY}
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_err("no_match")):
            result = runtime.run(obj)
        assert result["ok"] is False
        assert result["error"] == "no_match"


# ---------------------------------------------------------------------------
# 2. Chart file creation
# ---------------------------------------------------------------------------


class TestChartFileCreation:
    def _run_chart(self, tmp_path, points, query):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        obj = {"category": "chart_request", "query": query}
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_ok(points)):
            return runtime.run(obj)

    def test_categorical_creates_png(self, tmp_path):
        result = self._run_chart(tmp_path, CATEGORICAL_POINTS, CHART_QUERY)
        assert result["ok"] is True, result.get("error")
        assert result["chart_path"] is not None
        assert result["chart_path"].exists()
        assert result["chart_path"].suffix == ".png"

    def test_categorical_creates_html(self, tmp_path):
        result = self._run_chart(tmp_path, CATEGORICAL_POINTS, CHART_QUERY)
        assert result["ok"] is True, result.get("error")
        # HTML is created when plotly is available
        if result["html_path"] is not None:
            assert result["html_path"].exists()
            assert result["html_path"].suffix == ".html"

    def test_time_series_creates_png(self, tmp_path):
        query = dict(CHART_QUERY, chart_type="time_series", measure_group="YEAR")
        result = self._run_chart(tmp_path, TIME_SERIES_POINTS, query)
        assert result["ok"] is True, result.get("error")
        assert result["chart_path"].exists()

    def test_compare_labels_creates_png(self, tmp_path):
        query = dict(CHART_QUERY, chart_type="compare_labels")
        result = self._run_chart(tmp_path, COMPARE_LABELS_POINTS, query)
        assert result["ok"] is True, result.get("error")
        assert result["chart_path"].exists()

    def test_out_dir_created_if_missing(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=nested)
        obj = {"category": "chart_request", "query": CHART_QUERY}
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_ok(CATEGORICAL_POINTS)):
            result = runtime.run(obj)
        assert result["ok"] is True
        assert nested.exists()

    def test_chart_summary_shape(self, tmp_path):
        result = self._run_chart(tmp_path, CATEGORICAL_POINTS, CHART_QUERY)
        assert result["ok"] is True
        cs = result["chart_summary"]
        assert cs["chart_type"] == "categorical"
        assert cs["x_field"] == "measure"
        assert cs["y_field"] == "value"
        assert cs["row_count"] == len(CATEGORICAL_POINTS)

    def test_empty_points_returns_error(self, tmp_path):
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        obj = {"category": "chart_request", "query": CHART_QUERY}
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_ok([])):
            result = runtime.run(obj)
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 3. DataFrame shaping
# ---------------------------------------------------------------------------


class TestDataShaper:
    def test_categorical_columns(self):
        df = points_to_dataframe(CATEGORICAL_POINTS, "categorical")
        assert list(df.columns) == ["measure", "value"]
        assert len(df) == len(CATEGORICAL_POINTS)

    def test_categorical_values_numeric(self):
        df = points_to_dataframe(CATEGORICAL_POINTS, "categorical")
        assert df["value"].dtype.kind in ("f", "i", "u")

    def test_time_series_columns(self):
        df = points_to_dataframe(TIME_SERIES_POINTS, "time_series")
        assert "year" in df.columns
        assert "value" in df.columns

    def test_time_series_year_numeric(self):
        df = points_to_dataframe(TIME_SERIES_POINTS, "time_series")
        assert df["year"].dtype.kind in ("f", "i", "u")

    def test_compare_labels_columns(self):
        df = points_to_dataframe(COMPARE_LABELS_POINTS, "compare_labels")
        assert "label" in df.columns
        assert "value" in df.columns

    def test_empty_points_returns_empty_df(self):
        df = points_to_dataframe([], "categorical")
        assert df.empty

    def test_non_numeric_y_dropped(self):
        pts = [{"x": "A", "y": "n/a"}, {"x": "B", "y": 100}]
        df = points_to_dataframe(pts, "categorical")
        assert len(df) == 1
        assert df.iloc[0]["measure"] == "B"

    def test_string_year_coerced(self):
        pts = [{"x": "2020", "y": 500}, {"x": "2021", "y": 600}]
        df = points_to_dataframe(pts, "time_series")
        assert df["year"].tolist() == [2020.0, 2021.0]

    def test_unknown_chart_type_uses_x_column(self):
        pts = [{"x": "foo", "y": 1}]
        df = points_to_dataframe(pts, "unknown_type")
        assert "x" in df.columns


# ---------------------------------------------------------------------------
# 4. Metadata sanitization (defensive checks)
# ---------------------------------------------------------------------------


class TestMetadataSanitization:
    def test_unapproved_measures_dropped(self, tmp_path):
        """Measures not in the approved group must be silently dropped."""
        query_with_bad = dict(
            CHART_QUERY,
            filters=dict(
                CHART_QUERY["filters"],
                measures_in=["White alone", "FAKE MEASURE"],
            ),
        )
        obj = {"category": "chart_request", "query": query_with_bad}

        captured_query = {}

        def fake_sql(db_path, category, query):
            captured_query.update(query)
            return _make_sql_ok(CATEGORICAL_POINTS)

        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        with patch("analytics.runtime.run_structured_query", side_effect=fake_sql):
            result = runtime.run(obj, metadata=SAMPLE_METADATA)

        assert result["ok"] is True
        measures = captured_query.get("filters", {}).get("measures_in", [])
        assert "FAKE MEASURE" not in measures
        assert "White alone" in measures

    def test_no_metadata_passes_through(self, tmp_path):
        """When metadata is None the query is executed unchanged."""
        runtime = AnalyticsRuntime(db_path=":memory:", out_dir=tmp_path)
        obj = {"category": "chart_request", "query": CHART_QUERY}
        with patch("analytics.runtime.run_structured_query", return_value=_make_sql_ok(CATEGORICAL_POINTS)):
            result = runtime.run(obj, metadata=None)
        assert result["ok"] is True
