from collections import defaultdict
import math
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX

def is_number(x):
    try:
        if x is None:
            return False
        if isinstance(x, float) and math.isnan(x):
            return False
        return True
    except Exception:
        return False

def slope_lr(points):
    # points: [(year:int, value:float)]
    if len(points) < 2:
        return 0.0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    den = sum((x - xbar) ** 2 for x in xs)
    return (num / den) if den != 0 else 0.0

def forecast(items, subject, section, limit, order="desc", horizon_years=5):
    # 1) Build series: group -> list[(year,value)]
    series = defaultdict(list)
    for it in items:
        g = it.get("measure")
        y = it.get("year")
        v = it.get("value")
        if g is None or y is None or not is_number(v):
            continue
        series[g].append((int(y), float(v)))

    # sort + dedupe years
    for g in list(series.keys()):
        pts = sorted(series[g], key=lambda t: t[0])
        dedup = {}
        for y, v in pts:
            dedup[y] = v
        series[g] = [(y, dedup[y]) for y in sorted(dedup.keys())]

    # 2) Score each group
    scored = []
    for g, pts in series.items():
        if len(pts) == 0:
            continue
        first_year, first_val = pts[0]
        latest_year, latest_val = pts[-1]
        delta = latest_val - first_val
        slope = slope_lr(pts)
        forecast_val = latest_val + slope * float(horizon_years)

        scored.append({
            "group": g,
            "n_points": len(pts),
            "first_year": first_year,
            "first_value": first_val,
            "latest_year": latest_year,
            "latest_value": latest_val,
            "delta": delta,
            "slope_per_year": slope,
            "forecast_value": forecast_val,
            "forecast_horizon_years": horizon_years,
            "series": pts,
        })

    reverse = (order == "desc")

    # 3) Rank by forecast (default “most at risk” for Percent Uninsured)
    ranked = sorted(
        scored,
        key=lambda s: (s["forecast_value"], s["latest_value"]),
        reverse=reverse
    )[: int(limit)]

    # 4) Chart payloads
    top_groups = [r["group"] for r in ranked]
    time_series_chart = {
        "chart_type": "time_series",
        "title": f"{subject} trend by group ({section})",
        "series": [
            {"name": g, "data": [{"x": y, "y": v} for (y, v) in series[g]]}
            for g in top_groups
        ],
    }
    latest_bar = {
        "chart_type": "bar",
        "title": f"{subject} (latest year)",
        "data": [{"x": r["group"], "y": r["latest_value"]} for r in ranked],
    }

    return {
        "ranking": ranked,
        "charts": {
            "time_series": time_series_chart,
            "latest": latest_bar,
        }
    }


# Convert format returned by time_series_chart to format compatible with AnalyticsRuntime
def to_chart_request_time_series_with_linear_forecast(
    ranking,
    measure_group,
    title,
    viz_type="line",
    horizon_years=5,
):
    points = []
    for r in ranking:
        group = r["group"]
        series = r.get("series", [])
        if not series:
            continue

        # observed points
        for (year, value) in series:
            points.append({
                "x": int(year),
                "y": float(value),
                "measure": group,
                "is_forecast": False,
            })

        # linear projection from latest point using slope_per_year
        latest_year = int(r["latest_year"])
        latest_val = float(r["latest_value"])
        slope = float(r.get("slope_per_year") or 0.0)

        for step in range(1, int(horizon_years) + 1):
            y = latest_year + step
            v = latest_val + slope * step
            points.append({
                "x": y,
                "y": float(v),
                "measure": group,
                "is_forecast": True,
            })

    return {
        "category": "chart_request",
        "query": {
            "chart_type": "time_series",
            "viz_type": viz_type,
            "measure_group": measure_group,
            "title": title,
            "show_forecast": True,
            "forecast_horizon_years": horizon_years,
        },
        "result": {"points": points},
    }