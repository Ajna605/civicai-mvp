import json
from typing import Dict, Any, Tuple, List, Optional
from sql_engine.llm_utils.json_schema import SCHEMA_TEXT, FEWSHOT_TEXT

ALLOWED_STAT_TYPES = {"Estimate", "Margin of Error"}

## need to be dynamic for deterministic logic
def make_param_prompt(question: str, metadata: dict, constraints: Optional[Dict[str, Any]] = None) -> str:
    forced = ""
    if constraints:
        # Keep this short & absolute
        lines = ["FORCED CONSTRAINTS (must follow exactly):"]
        if "force_category" in constraints:
            lines.append(f'- category MUST be "{constraints["force_category"]}"')
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
    return (
        "Your previous JSON failed validation.\n"
        f"ERROR: {error}\n\n"
        "Fix it to pass validation.\n"
        "Return ONLY valid JSON. No prose.\n\n"
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


