"""Chart rendering: matplotlib (PNG) with optional Plotly (HTML).

Three chart types are supported:

- ``categorical``    – bar chart,   x=measure, y=value
- ``time_series``    – line chart,  x=year,    y=value
- ``compare_labels`` – grouped bar, x=label,   y=value

Each render function returns a tuple ``(png_path, html_path|None)``.
``html_path`` is *None* when Plotly is not available.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# Use the non-interactive Agg backend so rendering works in headless environments
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 – must follow matplotlib.use()

logger = logging.getLogger(__name__)

# Plotly is optional; fall back gracefully
_PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as _go  # noqa: F401
    _PLOTLY_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

_ALLOWED_SORTS = {"x_asc", "x_desc", "y_asc", "y_desc"}


def _apply_sort(df: pd.DataFrame, x_col: str, sort: Optional[str]) -> pd.DataFrame:
    """Return a sorted copy of *df* according to the *sort* option."""
    if not sort or sort not in _ALLOWED_SORTS:
        return df
    if sort == "y_desc":
        return df.sort_values("value", ascending=False).reset_index(drop=True)
    if sort == "y_asc":
        return df.sort_values("value", ascending=True).reset_index(drop=True)
    if sort == "x_desc":
        return df.sort_values(x_col, ascending=False).reset_index(drop=True)
    # x_asc is already the SQL default; return as-is
    return df


# ---------------------------------------------------------------------------
# PNG helpers
# ---------------------------------------------------------------------------

def _save_png_bar(x_vals, y_vals, title: str, xlabel: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(max(8, len(x_vals) * 0.9), 5))
    ax.bar(x_vals, y_vals, color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    return png_path


def _save_png_line(x_vals, y_vals, title: str, xlabel: str, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x_vals, y_vals, marker="o", color="steelblue")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Value")
    ax.set_title(title)
    plt.tight_layout()
    png_path = out_path.with_suffix(".png")
    fig.savefig(png_path, dpi=100)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# HTML helpers (Plotly)
# ---------------------------------------------------------------------------

def _save_html_bar(x_vals, y_vals, title: str, xlabel: str, out_path: Path) -> Optional[Path]:
    if not _PLOTLY_AVAILABLE:
        return None
    import plotly.graph_objects as go
    fig = go.Figure(go.Bar(x=x_vals, y=y_vals))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Value")
    html_path = out_path.with_suffix(".html")
    fig.write_html(str(html_path))
    return html_path


def _save_html_line(x_vals, y_vals, title: str, xlabel: str, out_path: Path) -> Optional[Path]:
    if not _PLOTLY_AVAILABLE:
        return None
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(x=x_vals, y=y_vals, mode="lines+markers"))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title="Value")
    html_path = out_path.with_suffix(".html")
    fig.write_html(str(html_path))
    return html_path


# ---------------------------------------------------------------------------
# Public render functions
# ---------------------------------------------------------------------------

def render_categorical_bar(
    df: pd.DataFrame,
    query: Dict[str, Any],
    out_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """Render a categorical bar chart (x=measure, y=value).

    Parameters
    ----------
    df:
        DataFrame with columns ``measure`` and ``value``.
    query:
        Original chart_request query dict (used for title and sort).
    out_path:
        Base output path *without* extension; ``.png`` / ``.html`` are appended.

    Returns
    -------
    (png_path, html_path | None)
    """
    df = _apply_sort(df, "measure", query.get("sort"))
    title = f"Distribution by {query.get('measure_group', 'Measure')}"
    x_vals = df["measure"].tolist()
    y_vals = df["value"].tolist()
    png_path = _save_png_bar(x_vals, y_vals, title, "Measure", out_path)
    html_path = _save_html_bar(x_vals, y_vals, title, "Measure", out_path)
    return png_path, html_path


def render_time_series(
    df: pd.DataFrame,
    query: Dict[str, Any],
    out_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """Render a time-series line chart (x=year, y=value).

    Parameters
    ----------
    df:
        DataFrame with columns ``year`` and ``value``.
    query:
        Original chart_request query dict (used for title).
    out_path:
        Base output path *without* extension.

    Returns
    -------
    (png_path, html_path | None)
    """
    df = df.sort_values("year").reset_index(drop=True)
    title = f"Trend over Time – {query.get('measure_group', '')}"
    x_vals = df["year"].tolist()
    y_vals = df["value"].tolist()
    png_path = _save_png_line(x_vals, y_vals, title, "Year", out_path)
    html_path = _save_html_line(x_vals, y_vals, title, "Year", out_path)
    return png_path, html_path


def render_compare_labels(
    df: pd.DataFrame,
    query: Dict[str, Any],
    out_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """Render a compare-labels grouped bar chart (x=label, y=value).

    Parameters
    ----------
    df:
        DataFrame with columns ``label`` and ``value``.
    query:
        Original chart_request query dict (used for title and sort).
    out_path:
        Base output path *without* extension.

    Returns
    -------
    (png_path, html_path | None)
    """
    df = _apply_sort(df, "label", query.get("sort"))
    title = f"Comparison by Label – {query.get('measure_group', '')}"
    x_vals = df["label"].tolist()
    y_vals = df["value"].tolist()
    png_path = _save_png_bar(x_vals, y_vals, title, "Label", out_path)
    html_path = _save_html_bar(x_vals, y_vals, title, "Label", out_path)
    return png_path, html_path
