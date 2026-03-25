## Functions for LLM to generate parameters

import json
from pathlib import Path
from sql_engine.llm_utils.validator import build_repair_prompt, make_param_prompt
from typing import Dict, Any
from sql_engine.llm_utils.llm_settings import generate_json_only, build_param_llm
from sql_engine.llm_utils.query_guards import resolve_measure_override
import time


def load_metadata(path: str | None = None) -> dict:
    # If no path passed, load metadata.json that sits next to this file (llm_utils/)
    if path is None:
        path = str(Path(__file__).resolve().parent / "metadata.json")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

PARAM_LLM = build_param_llm()
print(PARAM_LLM._model.device)

def llm_make_params(
    question: str,
    metadata: Dict[str, Any],
    constraints: dict | None = None,
    max_repairs: int = 0 ##DEBUG
) -> Dict[str, Any]:

    constraints = resolve_measure_override(question, metadata)

    prompt = make_param_prompt(question, metadata, constraints=constraints)
    t0 = time.time()
    raw = generate_json_only(PARAM_LLM, prompt).strip()
    print("llm seconds:", round(time.time() - t0, 2))

    for attempt in range(max_repairs + 1):
        try:
            obj = json.loads(raw)
        except Exception:
            error = "invalid_json"
            raw = generate_json_only(
                PARAM_LLM,
                build_repair_prompt(question, raw, error, metadata)
            ).strip()
            continue

        # Optional: plug in your real validator here
        if not isinstance(obj, dict) or "category" not in obj or "query" not in obj:
            error = "missing_category_or_query"
            raw = generate_json_only(
                PARAM_LLM,
                build_repair_prompt(question, raw, error, metadata)
            ).strip()
            continue

        return obj  # success

    raise ValueError(f"Failed to generate valid JSON after {max_repairs} repairs.\nLast output:\n{raw}")

# if __name__ == "__main__":

#     question = "How many males per 100 females?"
#     metadata = load_metadata()

#     # Deterministic logic to override LLM guessing
#     measure_override = direct_measure_first_guard(question, metadata)

#     result = llm_make_params(question, metadata, measure_override)
#     print(json.dumps(result, indent=2))