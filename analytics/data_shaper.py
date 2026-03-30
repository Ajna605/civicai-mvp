"""Normalize SQL engine output into a pandas DataFrame for chart rendering.

The SQL engine returns chart data as a list of ``{"x": ..., "y": ...}`` dicts.
These helpers convert that to a tidy DataFrame with semantically named columns
that the renderers expect.
"""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


# Mapping from chart_type to the semantic name of the x-axis column
X_COL: Dict[str, str] = {
    "categorical": "measure",
    "time_series": "year",
    "compare_labels": "label",
}


def points_to_dataframe(
    points: List[Dict[str, Any]],
    chart_type: str,
) -> pd.DataFrame:
    """Convert ``chart_request`` SQL results to a tidy DataFrame.

    Parameters
    ----------
    points:
        List of ``{"x": ..., "y": ...}`` dicts returned by the SQL engine.
    chart_type:
        One of ``"categorical"``, ``"time_series"``, or ``"compare_labels"``.

    Returns
    -------
    pd.DataFrame
        Columns depend on chart_type:

        - ``categorical``    → ``measure`` (str), ``value`` (float)
        - ``time_series``    → ``year`` (numeric), ``value`` (float)
        - ``compare_labels`` → ``label`` (str), ``value`` (float)

        An empty DataFrame with columns ``["x", "y"]`` is returned when
        *points* is empty.
    """
    if not points:
        return pd.DataFrame(columns=["x", "y"])

    df = pd.DataFrame(points)

    if "x" not in df.columns or "y" not in df.columns:
        return pd.DataFrame(columns=["x", "y"])

    x_col = X_COL.get(chart_type, "x")
    df = df.rename(columns={"x": x_col, "y": "value"})

    # Coerce value to numeric, dropping rows that cannot be converted
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    # For time_series coerce year to numeric as well
    if chart_type == "time_series" and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df = df.dropna(subset=["year"]).reset_index(drop=True)

    return df
