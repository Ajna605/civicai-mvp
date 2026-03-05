from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from sql_engine.llm_utils.query_guards import direct_measure_first_guard
from sql_engine.llm_utils.param_gen import llm_make_params, load_metadata
DEFAULT_SUBJECT = "Total"
DEFAULT_STAT_TYPE = "Estimate"

def load_questions_json(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON array of questions.")
    return data

def normalize_gold(ex: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing subject/stat_type in gold cell_lookup queries deterministically."""
    ex = dict(ex)
    q = dict(ex.get("query", {}))
    if ex.get("category") == "cell_lookup":
        q.setdefault("subject", DEFAULT_SUBJECT)
        q.setdefault("stat_type", DEFAULT_STAT_TYPE)
    ex["query"] = q
    return ex

def exact_match(a: Any, b: Any) -> bool:
    return a == b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", required=True, help="Path to test_questions.json")
    ap.add_argument("--out_path", default="eval/paramgen_results.jsonl")
    ap.add_argument("--max_repairs", type=int, default=2)
    args = ap.parse_args()

    meta = load_metadata()
    items = [normalize_gold(x) for x in load_questions_json(args.eval_path)]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(items)
    cat_ok = 0
    json_and_schema_ok = 0
    query_exact_ok = 0

    rows = []
    for ex in items:
        qid = ex.get("id")
        question = ex["question"]
        print(question)
        gold_cat = ex["category"]
        gold_query = ex.get("query", {})

        measure_override = direct_measure_first_guard(question, meta)

        rec: Dict[str, Any] = {"id": qid, "question": question, "gold_category": gold_cat, "gold_query": gold_query}

        try:
            pred = llm_make_params(question, meta, max_repairs=args.max_repairs, constraints=measure_override)
            rec["pred"] = pred

            pred_cat = pred.get("category")
            pred_query = pred.get("query")

            # basic “schema present” check (swap with your real validator later)
            schema_pass = isinstance(pred_cat, str) and isinstance(pred_query, dict)
            rec["schema_pass"] = schema_pass
            if schema_pass:
                json_and_schema_ok += 1

            rec["pred_category"] = pred_cat
            rec["category_match"] = (pred_cat == gold_cat)
            if rec["category_match"]:
                cat_ok += 1

            # Exact match against gold query (after gold normalization)
            rec["query_exact_match"] = exact_match(pred_query, gold_query)
            if rec["query_exact_match"]:
                query_exact_ok += 1

        except Exception as e:
            rec["error"] = str(e)
            rec["schema_pass"] = False
            rec["category_match"] = False
            rec["query_exact_match"] = False

        rows.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== ParamGen Eval (vs test_questions.json) ===")
    print(f"Total: {total}")
    print(f"Schema present: {json_and_schema_ok}/{total} = {json_and_schema_ok/total:.2%}")
    print(f"Category exact: {cat_ok}/{total} = {cat_ok/total:.2%}")
    print(f"Query exact: {query_exact_ok}/{total} = {query_exact_ok/total:.2%}")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()