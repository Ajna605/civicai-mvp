import json
from typing import Dict, Any, Tuple, List, Optional
from sql_engine.llm_utils.json_schema import SCHEMA_TEXT, FEWSHOT_TEXT
from sql_engine.llm_utils.query_guards import OVER_HINT, UNDER_HINT, RANGE_HINT, age_under_measures_in, age_over_measures_in, measures_overlapping_range

ALLOWED_STAT_TYPES = {"Estimate", "Margin of Error"}

## need to be dynamic for deterministic logic
def make_param_prompt(question: str, metadata: dict, constraints: Optional[Dict[str, Any]] = None) -> str:
    forced = ""
    if constraints:
        # Keep this short & absolute
        lines = ["FORCED CONSTRAINTS (must follow exactly):"]
        if "force_category" in constraints:
            lines.append(f'- category MUST be "{constraints["force_category"]}"')
        if "force_measure_group" in constraints:
            lines.append(f'- measure_group MUST be exactly: "{constraints["force_measure_group"]}"')
        if "force_measure" in constraints:
            lines.append(f'- measure MUST be exactly: "{constraints["force_measure"]}"')
        if "force_stat_type" in constraints:
            lines.append(f'- stat_type MUST be "{constraints["force_stat_type"]}"')
        if "force_subject" in constraints:
            lines.append(f'- subject MUST be exactly: "{constraints["force_subject"]}"')
        forced = "\n".join(lines) + "\n\n"

    return (
        forced
        + SCHEMA_TEXT.strip()
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
            "labels": meta.get("labels", []),
            "stat_types": meta.get("stat_types", []),
        }

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
