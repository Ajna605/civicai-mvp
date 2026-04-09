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


# name=sql_engine/analytics_sql/renderers.py  (or wherever render_time_series lives)

def render_time_series(df, query, out_path):
    import matplotlib.pyplot as plt

    # Pick the series/legend column
    series_col = None
    if "measure" in df.columns:
        series_col = "measure"
    elif "label" in df.columns:
        series_col = "label"

    has_forecast = "is_forecast" in df.columns

    fig, ax = plt.subplots(figsize=(12, 5))

    if series_col is None:
        d = df.sort_values("year")

        if has_forecast:
            obs = d[d["is_forecast"] == False]
            fc = d[d["is_forecast"] == True]

            obs = obs.sort_values("year")
            fc = fc.sort_values("year")

            if not obs.empty:
                ax.plot(obs["year"], obs["value"], marker="o", linewidth=2, label="Observed")
            if not fc.empty:
                # bridge last observed -> first forecast
                if not obs.empty:
                    x = [obs.iloc[-1]["year"]] + fc["year"].tolist()
                    y = [obs.iloc[-1]["value"]] + fc["value"].tolist()
                else:
                    x = fc["year"].tolist()
                    y = fc["value"].tolist()
                ax.plot(x, y, marker="o", linewidth=2, linestyle="--", alpha=0.9, label="Forecast")

            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            ax.plot(d["year"], d["value"], marker="o", linewidth=2)

    else:
        # Multi-series: one line per group, with optional dashed forecast continuation
        for name, g in df.groupby(series_col):
            g = g.sort_values("year")

            if has_forecast:
                obs = g[g["is_forecast"] == False].sort_values("year")
                fc = g[g["is_forecast"] == True].sort_values("year")

                # Observed (solid)
                if not obs.empty:
                    ax.plot(
                        obs["year"], obs["value"],
                        marker="o", linewidth=2, label=str(name)
                    )

                # Forecast (dashed) in same color as observed line:
                if not fc.empty:
                    if not obs.empty:
                        x = [obs.iloc[-1]["year"]] + fc["year"].tolist()
                        y = [obs.iloc[-1]["value"]] + fc["value"].tolist()
                    else:
                        x = fc["year"].tolist()
                        y = fc["value"].tolist()

                    # Use the last line's color so dashed matches the group color
                    color = ax.lines[-1].get_color() if ax.lines else None
                    ax.plot(
                        x, y,
                        marker="o", linewidth=2, linestyle="--", alpha=0.7,
                        color=color
                    )
            else:
                ax.plot(g["year"], g["value"], marker="o", linewidth=2, label=str(name))

        ax.legend(title=series_col, bbox_to_anchor=(1.02, 1), loc="upper left")

        if has_forecast:
            # add a small note on the plot area so users understand dashed meaning
            ax.text(
                0.01, 0.01,
                "Dashed = simple linear projection",
                transform=ax.transAxes,
                fontsize=9,
                alpha=0.8,
                va="bottom",
            )

    ax.set_title(query.get("title") or f"Trend over Time – {query.get('measure_group','')}")
    ax.set_xlabel("Year")
    ax.set_ylabel(query.get("y_label") or "Value")
    ax.grid(True, alpha=0.3)

    png_path = str(out_path) + ".png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    return png_path, None


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


def render_categorical_pie(
    df: pd.DataFrame,
    query: Dict[str, Any],
    out_path: Path,
) -> Tuple[Path, Optional[Path]]:
    """
    Render a categorical pie chart using df columns:
      - measure (labels)
      - value (sizes)
    """
    # Optional sort: default x_asc if missing
    sort = query.get("sort") or "x_asc"
    df = _apply_sort(df, "measure", sort)

    title = f"Distribution by {query.get('measure_group', 'Measure')}"

    labels = df["measure"].astype(str).tolist()
    values = df["value"].tolist()

    # ---- PNG (matplotlib) ----
    png_path = out_path.with_suffix(".png")
    plt.figure(figsize=(10, 8))
    plt.pie(values, labels=None, autopct="%1.1f%%", startangle=90)
    plt.title(title)
    plt.axis("equal")
    # Put legend on the side (labels can be long)
    plt.legend(labels, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize="small")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()

    # ---- HTML (plotly) optional ----
    html_path: Optional[Path] = None
    if _go is not None:
        fig = _go.Figure(data=[_go.Pie(labels=labels, values=values, textinfo="percent+label")])
        fig.update_layout(title=title)
        html_path = out_path.with_suffix(".html")
        fig.write_html(str(html_path))

    return png_path, html_path