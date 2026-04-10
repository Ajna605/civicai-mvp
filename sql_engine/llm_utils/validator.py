import json
from typing import Dict, Any, Tuple, List, Optional
from sql_engine.llm_utils.json_schema import SCHEMA_TEXT, FEWSHOT_TEXT
from sql_engine.llm_utils.query_guards import OVER_HINT, UNDER_HINT, RANGE_HINT, age_under_measures_in, age_over_measures_in, measures_overlapping_range
from sql_engine.llm_utils.json_schema import CATEGORY_SCHEMAS, REQUIRED_BY_CATEGORY
from utils.schema_error_utils import _field_help, _parse_schema_error
ALLOWED_STAT_TYPES = {"Estimate", "Margin of Error"}


def make_param_prompt(question: str, metadata: dict, constraints: Optional[Dict[str, Any]] = None) -> str:
    forced = ""

    if constraints:
        lines = ["FORCED CONSTRAINTS (must follow exactly):"]

        if constraints.get("force_category") is not None:
            lines.append(f'- category MUST be "{constraints["force_category"]}"')
        if constraints.get("force_label") is not None:
            lines.append(f'- label MUST be exactly: "{constraints["force_label"]}"')
        if constraints.get("force_measure_group") is not None:
            lines.append(f'- measure_group MUST be exactly: "{constraints["force_measure_group"]}"')
        if constraints.get("force_measures_in") is not None:
            lines.append(
                f'- measures_in MUST be exactly this list (no extras, no missing): '
                f'{json.dumps(constraints["force_measures_in"], ensure_ascii=False)}'
            )
        if constraints.get("force_measure") is not None:
            lines.append(f'- measure MUST be exactly: "{constraints["force_measure"]}"')
        if constraints.get("force_stat_type") is not None:
            lines.append(f'- stat_type MUST be "{constraints["force_stat_type"]}"')
        if constraints.get("force_subject") is not None:
            lines.append(f'- subject MUST be exactly: "{constraints["force_subject"]}"')
        if constraints.get("force_op") is not None:
            lines.append(f'- op MUST be "{constraints["force_op"]}"')

        # If only the header exists, don't include forced at all
        forced = ("\n".join(lines) + "\n\n") if len(lines) > 1 else ""

    # If force_category exists, show ONLY that schema to reduce nesting mistakes
    forced_cat = constraints.get("force_category") if constraints else None
    schema = SCHEMA_TEXT.strip()

    if forced_cat in CATEGORY_SCHEMAS:
        schema = (
            SCHEMA_TEXT.strip()
            + "\n\n"
            + "ONLY USE THIS SCHEMA (because category is forced):\n"
            + CATEGORY_SCHEMAS[forced_cat]
        )


    return (
        forced
        + schema
        + "\n\n"
        + FEWSHOT_TEXT.strip()
        + "\n\nQUESTION:\n"
        + question.strip()
        + "\n\nMETADATA:\n"
        + json.dumps(metadata, ensure_ascii=False)
    )

def validate_cell_lookup(obj: Dict[str, Any], meta: Dict[str, List[str]]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "root_not_object"
    if obj.get("category") != "cell_lookup":
        return False, "category_must_be_cell_lookup"

    q = obj.get("query")
    if not isinstance(q, dict):
        return False, "query_not_object"

    allowed_keys = {"label", "measure", "subject", "stat_type"}
    if set(q.keys()) != allowed_keys:
        return False, f"query_keys_must_be_exactly_{sorted(list(allowed_keys))}_got_{sorted(list(q.keys()))}"

    if q["stat_type"] not in ALLOWED_STAT_TYPES:
        return False, "stat_type_invalid"

    # exact membership checks
    if q["label"] not in meta.get("labels", []):
        return False, "label_not_in_metadata"
    if q["measure"] not in meta.get("measures", []):
        return False, "measure_not_in_metadata"
    if q["subject"] not in meta.get("subjects", []):
        return False, "subject_not_in_metadata"
    if q["stat_type"] not in meta.get("stat_types", []):
        # if your metadata always includes these two, this is redundant
        return False, "stat_type_not_in_metadata"

    return True, "ok"

# sql_engine/llm_utils/validator.py

ALLOWED_SORT = {"x_asc", "x_desc", "y_asc", "y_desc"}

def default_chart_sort(pred: dict) -> dict:
    """
    If sort missing → default to x_asc for categorical bar charts.
    Leaves other chart types unchanged.
    """
    if pred.get("category") != "chart_request":
        return pred

    q = pred.get("query") or {}
    chart_type = (q.get("chart_type") or "categorical").lower()
    print("chart type", chart_type)
    sort = q.get("sort")

    if sort is not None and sort not in ALLOWED_SORT:
        # either drop invalid sort or error; I'd drop to be robust
        q.pop("sort", None)
        sort = None

    if sort is None and chart_type == "categorical" :
        q["sort"] = "x_asc"

    pred["query"] = q
    return pred

def build_repair_prompt(question: str, bad_json: str, error: str, meta: Dict[str, Any]) -> str:
    extra = ""

    try:
        parsed = json.loads(bad_json)
    except Exception:
        parsed = None

    q = parsed.get("query", {}) if isinstance(parsed, dict) else {}
    measure_group = q.get("measure_group")

    if error == "missing_measure_group":
        extra = (
            "The JSON uses filters.measures_in but is missing query.measure_group.\n"
            "Add measure_group using one value from METADATA.measure_headings.\n"
        )

    elif error.startswith("unknown_measure_group:"):
        bad_group = error.split(":", 1)[1]
        extra = (
            f'The measure_group "{bad_group}" is invalid.\n'
            f'You must choose measure_group from this list only:\n'
            f'{json.dumps(meta.get("measure_headings", []), ensure_ascii=False)}\n'
        )

    elif error.startswith("measures_not_in_group:"):
        allowed = meta.get("measure_groups", {}).get(measure_group, [])
        extra = (
            f'The JSON uses measure_group "{measure_group}", but some values in filters.measures_in '
            f'are not members of that group.\n'
            f'Allowed measures for "{measure_group}" are:\n'
            f'{json.dumps(allowed, ensure_ascii=False)}\n'
            "Keep the same measure_group and regenerate filters.measures_in using only allowed measures.\n"
        )
        meta = {
            "measure_headings": [measure_group],
            "allowed_measures_for_group": allowed,
            "subjects": meta.get("subjects", []),
            "geo_labels": meta.get("geo_labels", meta.get("labels", [])),
            "stat_types": meta.get("stat_types", []),
            "measure_groups": {measure_group: allowed},
        }

    elif error.startswith("schema:"):
        info = _parse_schema_error(error)
        missing = info["missing"]
        empty = info["empty"]

        lines = []
        if missing:
            lines.append("Some required fields are missing: " + ", ".join(missing))
        if empty:
            lines.append("Some required fields are present but EMPTY (null/''/[]/{}): " + ", ".join(empty))

        # Provide field-specific “choose from metadata” guidance
        hints = []
        for p in (missing + empty):
            h = _field_help(p, meta, measure_group)
            if h:
                hints.append(f"- {p}: {h}")

        extra = (
            "\n".join(lines) + "\n\n"
            "Fix the JSON as follows:\n"
            "- Required fields must be filled with valid values.\n"
            "- Choose values ONLY from the METADATA lists for that field. Never invent strings.\n"
            "- If you truly cannot choose a valid value from METADATA, output null (do NOT output an empty string).\n"
            + ("\nField rules:\n" + "\n".join(hints) + "\n" if hints else "")
        )

    return (
        "Your previous JSON failed validation.\n"
        f"ERROR: {error}\n\n"
        f"{extra}\n"
        "Fix it to pass validation.\n"
        "Return ONLY one valid JSON object. No prose.\n"
        'The top-level object must have exactly these keys: "category" and "query".\n'
        "Do not include any prefix such as CORRECT_JSON or FIXED_JSON.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"PREVIOUS_JSON:\n{bad_json}\n\n"
        f"METADATA:\n{json.dumps(meta, ensure_ascii=False)}\n"
    )

## Making sure Measure Headings are not mixed 
def validate_measure_group_consistency(query: dict, meta: dict):
    q = query.get("query", query)
    filters = q.get("filters", {})
    measures_in = filters.get("measures_in")
    measure_group = q.get("measure_group")

    if not measures_in:
        return True, "ok"

    if not measure_group:
        return False, "missing_measure_group"

    groups = meta.get("measure_groups", {})
    allowed = set(groups.get(measure_group, []))

    if not allowed:
        return False, f"unknown_measure_group:{measure_group}"

    bad = [m for m in measures_in if m not in allowed]
    if bad:
        return False, f"measures_not_in_group:{bad}"

    return True, "ok"


#### Post LLM Generation Guard for measures
def resolve_measures_in_from_group(
    question: str,
    measure_group: str,
    meta: dict,
) -> list[str] | None:
    groups = meta.get("measure_groups", {})
    allowed = groups.get(measure_group, [])
    if not allowed:
        return None

    q = question.lower()

    # under X
    m = UNDER_HINT.search(q)
    if m:
        target = int(m.group("n"))
        return age_under_measures_in(target, allowed)

    # over X
    m = OVER_HINT.search(q)
    if m:
        target = int(m.group("n"))
        return age_over_measures_in(target, allowed)

    # from X to Y / between X and Y
    m = RANGE_HINT.search(q)
    if m:
        low = int(m.group("low"))
        high = int(m.group("high"))
        return measures_overlapping_range(low, high, allowed)

    # no explicit range -> full group
    return allowed

##
def enforce_deterministic_measures(pred: dict, question: str, meta: dict) -> dict:
    q = pred.get("query", {})
    measure_group = q.get("measure_group")
    if not measure_group:
        return pred

    resolved = resolve_measures_in_from_group(question, measure_group, meta)
    if resolved:
        q.setdefault("filters", {})
        q["filters"]["measures_in"] = resolved

    pred["query"] = q
    return pred

def validate_against_metadata(obj: dict, meta: dict) -> tuple[bool, str]:
    cat = obj.get("category")
    q = obj.get("query")
    if not isinstance(q, dict):
        return False, "query_not_object"

    if cat == "cell_lookup":
        ok = (
            q.get("label") in (meta.get("labels") or [])
            and q.get("measure") in (meta.get("measures") or [])
            and q.get("subject") in (meta.get("subjects") or [])
            and q.get("stat_type") in (meta.get("stat_types") or [])
        )
        return (ok, "ok" if ok else "cell_lookup_value_not_in_metadata")

    if cat in {"aggregation", "row_filter", "chart_request"}:
        filters = q.get("filters") or {}
        if not isinstance(filters, dict):
            return False, "filters_not_object"

        # basic membership
        if filters.get("label") not in (meta.get("labels") or []):
            return False, "label_not_in_metadata"
        if filters.get("subject") not in (meta.get("subjects") or []):
            return False, "subject_not_in_metadata"
        if filters.get("stat_type") not in (meta.get("stat_types") or []):
            return False, "stat_type_not_in_metadata"

        # measures_in + measure_group consistency
        measures_in = filters.get("measures_in")
        if measures_in:
            mg = q.get("measure_group")
            groups = meta.get("measure_groups") or {}
            if mg not in groups:
                return False, "measure_group_not_in_metadata"
            allowed = set(groups.get(mg) or [])
            bad = [m for m in measures_in if m not in allowed]
            if bad:
                return False, f"measures_in_not_in_group:{bad}"

        return True, "ok"

    return True, "ok"

_MISSING = object()

def _get_path(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur

def _is_empty_value(v: Any) -> bool:
    # treat "", "   ", [], {} as empty; keep 0/False as valid
    if v is _MISSING:
        return False
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip() == ""
    if isinstance(v, (list, dict, tuple, set)):
        return len(v) == 0
    return False

def schema_check(obj: Dict[str, Any]) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "schema:not_an_object"
    if "category" not in obj or "query" not in obj:
        return False, "schema:missing_category_or_query"

    cat = obj.get("category")
    if cat not in REQUIRED_BY_CATEGORY:
        return False, f"schema:unknown_category:{cat}"

    missing: List[str] = []
    empty: List[str] = []

    for req in REQUIRED_BY_CATEGORY[cat]:
        v = _get_path(obj, req)
        if v is _MISSING:
            missing.append(req)
        elif _is_empty_value(v):
            empty.append(req)

    if missing or empty:
        parts = []
        if missing:
            parts.append("missing=" + ",".join(missing))
        if empty:
            parts.append("empty=" + ",".join(empty))
        return False, "schema:" + ";".join(parts)

    return True, "ok"