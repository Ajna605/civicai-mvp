## Functions for LLM to generate parameters

import json
from pathlib import Path
from sql_engine.llm_utils.validator import build_repair_prompt, make_param_prompt, validate_measure_group_consistency, validate_against_metadata, schema_check
from typing import Dict, Any
from sql_engine.llm_utils.llm_settings import generate_json_only, build_param_llm
from utils.text_utils import extract_first_valid_param_obj
from sql_engine.llm_utils.query_guards import resolve_measure_override
from copy import deepcopy
import time


def load_metadata(table_name: str) -> Dict[str, Any]:
    metadata_dir = Path("storage/metadata").expanduser().resolve()
    path = metadata_dir / f"{table_name}"
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata for table '{table_name}': {path}")
    return json.loads(path.read_text(encoding="utf-8"))

_PARAM_LLM = None
def get_param_llm():
    global _PARAM_LLM
    if _PARAM_LLM is None:
        _PARAM_LLM = build_param_llm()
        print(_PARAM_LLM._model.device)
    return _PARAM_LLM

def apply_forced_constraints(obj: Dict[str, Any], constraints: dict | None) -> Dict[str, Any]:
    if not constraints:
        return obj
    out = deepcopy(obj)
    if "force_category" in constraints:
        out["category"] = constraints["force_category"]
    out.setdefault("query", {})
    if not isinstance(out["query"], dict):
        out["query"] = {}

    # cell_lookup enforcement
    if out.get("category") == "cell_lookup":
        q = out["query"]
        if "force_label" in constraints:
            q["label"] = constraints["force_label"]
        if "force_measure" in constraints:
            q["measure"] = constraints["force_measure"]
        if "force_subject" in constraints:
            q["subject"] = constraints["force_subject"]
        if "force_stat_type" in constraints:
            q["stat_type"] = constraints["force_stat_type"]
        out["query"] = q

    # aggregation enforcement (if you use it)
    if out.get("category") == "aggregation":
        q = out["query"]
        if "force_op" in constraints:
            q["op"] = constraints["force_op"]
        if "force_measures_in" in constraints:
            q.setdefault("filters", {})
            q["filters"]["measures_in"] = constraints["force_measures_in"]
        out["query"] = q

    return out


def llm_make_params(
    question: str,
    metadata: Dict[str, Any],
    constraints: dict | None = None,
    max_repairs: int = 0,  # DEBUG
) -> Dict[str, Any]:
    
    # IMPORTANT: if constraints are provided by the caller (e.g., evaluation harness),
    # do not overwrite them. Otherwise compute deterministic override here.
    if constraints is None:
        constraints = resolve_measure_override(question, metadata)

    prompt = make_param_prompt(question, metadata, constraints=constraints)

    t0 = time.time()
    raw = generate_json_only(get_param_llm(), prompt).strip()
    print("llm seconds:", round(time.time() - t0, 2))

    for attempt in range(max_repairs + 1):
        try:
            obj = extract_first_valid_param_obj(raw)
        except Exception:
            if attempt == max_repairs:
                raise ValueError(
                    f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
                )
            error = "invalid_json"
            raw = generate_json_only(
                get_param_llm(),
                build_repair_prompt(question, raw, error, metadata)
            ).strip()
            continue

        if not isinstance(obj, dict) or "category" not in obj or "query" not in obj:
            if attempt == max_repairs:
                raise ValueError(
                    f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
                )
            error = "missing_category_or_query"
            raw = generate_json_only(
                get_param_llm(),
                build_repair_prompt(question, raw, error, metadata)
            ).strip()
            
            continue

        # semantic validation
        group_ok, group_reason = validate_measure_group_consistency(obj, metadata)
        if not group_ok:
            if attempt == max_repairs:
                raise ValueError(
                    f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
                )
            repair_prompt = build_repair_prompt(
                question,
                json.dumps(obj, ensure_ascii=False),
                group_reason,
                metadata,
            )
            raw = generate_json_only(get_param_llm(), repair_prompt).strip()
            continue


        # Hard-enforce any forced constraints (prevents invented strings).
        obj = apply_forced_constraints(obj, constraints)

        # If model invented a cell_lookup field not in metadata, force a repair
        # rather than returning junk that will never execute.
        if obj.get("category") == "cell_lookup" and not validate_against_metadata(obj.get("query", {}), metadata):
            error = "cell_lookup_values_not_in_metadata"
            raw = generate_json_only(
                get_param_llm(),
                build_repair_prompt(question, raw, error, metadata)
            ).strip()
            continue
        return obj

    raise ValueError(
        f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
    )


# if __name__ == "__main__":

#     question = "How many males per 100 females?"
#     metadata = load_metadata()

#     # Deterministic logic to override LLM guessing
#     measure_override = direct_measure_first_guard(question, metadata)

#     result = llm_make_params(question, metadata, measure_override)
#     print(json.dumps(result, indent=2))