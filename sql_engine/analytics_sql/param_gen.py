from sql_engine.analytics_sql.analytics_schema import SCHEMA_TEXT, FEWSHOT_TEXT
from sql_engine.analytics_sql.validator import build_repair_prompt
from sql_engine.llm_utils.llm_settings import generate_json_only, build_param_llm
from utils.text_utils import extract_first_valid_param_obj
from typing import Optional, Any, Dict
import json
import time
import os

def make_analytics_prompt(question: str, metadata: dict, constraints: Optional[Dict[str, Any]] = None) -> str:
    forced = ""
    if constraints:
        lines = ["FORCED CONSTRAINTS (must follow exactly):"]
        if constraints.get("allowed_measure_groups") is not None:
            lines.append(
                f'- inputs[].measure_group MUST be one of: '
                f'{json.dumps(constraints["allowed_measure_groups"], ensure_ascii=False)}'
            )
        forced = ("\n".join(lines) + "\n\n") if len(lines) > 1 else ""

    anti_nesting = (
        "\n\nANTI-NESTING RULE (critical):\n"
        "- The value of top-level key \"query\" MUST be the INNER query object ONLY.\n"
        "- Never put keys \"category\" or \"query\" inside the \"query\" object.\n"
    )

    return (
        forced
        + SCHEMA_TEXT.strip()
        + anti_nesting
        + "\n\n"
        + FEWSHOT_TEXT.strip()
        + "\n\nQUESTION:\n"
        + question.strip()
        + "\n\nMETADATA:\n"
        + json.dumps(metadata, ensure_ascii=False)
    )

_PARAM_LLM = None
def get_param_llm():
    global _PARAM_LLM
    if _PARAM_LLM is None:
        _PARAM_LLM = build_param_llm()
        print(_PARAM_LLM._model.device)
    
    print("BUILDING PARAM LLM in PID", os.getpid())
    return _PARAM_LLM

def llm_make_params(
    question: str,
    metadata: Dict[str, Any],
    constraints: dict | None = None,
    max_repairs: int = 0,  # DEBUG
) -> Dict[str, Any]:
    
    # IMPORTANT: if constraints are provided by the caller (e.g., evaluation harness),
    # do not overwrite them. Otherwise compute deterministic override here.

    prompt = make_analytics_prompt(question, metadata, constraints=constraints)
    # print("PROMPT", prompt)
    t0 = time.time()
    raw = generate_json_only(get_param_llm(), prompt).strip()
    print("raw", raw)
    obj = extract_first_valid_param_obj(raw)
    # for attempt in range(max_repairs + 1):
    #     try:
    #         obj = extract_first_valid_param_obj(raw)
    #     except Exception:
    #         if attempt == max_repairs:
    #             raise ValueError(
    #                 f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
    #             )
    #         error = "invalid_json"
    #         raw = generate_json_only(get_param_llm(), build_repair_prompt(question, raw, error, metadata)
    #         ).strip()
    #         continue
    print(obj)

    #     if not isinstance(obj, dict) or "category" not in obj or "query" not in obj:
    #         if attempt == max_repairs:
    #             raise ValueError(
    #                 f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
    #             )
    #         error = "missing_category_or_query"
    #         raw = generate_json_only(
    #             get_param_llm(),
    #             build_repair_prompt(question, raw, error, metadata)
    #         ).strip()
    #         continue

    #     # semantic validation
    #     group_ok, group_reason = validate_measure_group_consistency(obj, metadata)
    #     if not group_ok:
    #         if attempt == max_repairs:
    #             raise ValueError(
    #                 f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
    #             )
    #         repair_prompt = build_repair_prompt(
    #             question,
    #             json.dumps(obj, ensure_ascii=False),
    #             group_reason,
    #             metadata,
    #         )
    #         raw = generate_json_only(get_param_llm(), repair_prompt).strip()
    #         continue


    #     # Hard-enforce any forced constraints (prevents invented strings).
    #     obj = apply_forced_constraints(obj, constraints)

    #     # If model invented a cell_lookup field not in metadata, force a repair
    #     # rather than returning junk that will never execute.
    #     if obj.get("category") == "cell_lookup" and not validate_against_metadata(obj.get("query", {}), metadata):
    #         error = "cell_lookup_values_not_in_metadata"
    #         raw = generate_json_only(
    #             get_param_llm(),
    #             build_repair_prompt(question, raw, error, metadata)
    #         ).strip()
    #         continue
    #     return obj

    # raise ValueError(
    #     f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}"
    # )