from typing import Dict, Any
import json

def _parse_schema_error(error: str) -> dict:
    # error like: "schema:missing=a,b;empty=c,d"
    out = {"missing": [], "empty": []}
    if not error.startswith("schema:"):
        return out
    parts = error[len("schema:"):].split(";")
    for p in parts:
        if p.startswith("missing="):
            out["missing"] = [x for x in p[len("missing="):].split(",") if x]
        elif p.startswith("empty="):
            out["empty"] = [x for x in p[len("empty="):].split(",") if x]
    return out

def _field_help(req_path: str, meta: Dict[str, Any], measure_group: str | None) -> str:
    # Give the LLM the right “choose-from” list for each field
    if req_path.endswith(".measure_group"):
        return f"Choose from METADATA.measure_headings only: {json.dumps(meta.get('measure_headings', []), ensure_ascii=False)}"
    if req_path.endswith(".filters.label") or req_path.endswith(".label"):
        # Prefer geo_labels if you have it; avoid generic 'labels' if it mixes concepts
        geo_labels = meta.get("geo_labels") or meta.get("labels") or []
        return f"Choose from METADATA.geo_labels (geography) only: {json.dumps(geo_labels, ensure_ascii=False)}"
    if req_path.endswith(".filters.subject") or req_path.endswith(".subject"):
        return f"Choose from METADATA.subjects only: {json.dumps(meta.get('subjects', []), ensure_ascii=False)}"
    if req_path.endswith(".filters.stat_type") or req_path.endswith(".stat_type"):
        return f"Choose from METADATA.stat_types only: {json.dumps(meta.get('stat_types', []), ensure_ascii=False)}"
    if req_path.endswith(".filters.measures_in"):
        allowed = (meta.get("measure_groups", {}) or {}).get(measure_group or "", [])
        return f"Each item must be in METADATA.measure_groups[{json.dumps(measure_group)}] only: {json.dumps(allowed, ensure_ascii=False)}"
    if req_path.endswith(".order"):
        return 'Set query.order to "asc" or "desc".'
    if req_path.endswith(".limit"):
        return "Set query.limit to an integer (e.g., 1, 5, 10)."
    if req_path.endswith(".op"):
        return 'Set query.op to a valid operation like "sum", "avg", "min", "max".'
    if req_path.endswith(".viz_type"):
        return 'Set query.viz_type to a valid chart type like "pie", "bar", "line".'
    if req_path.endswith(".layout_type"):
        return 'Set query.layout_type to "categorical" (or another valid layout).'
    return ""