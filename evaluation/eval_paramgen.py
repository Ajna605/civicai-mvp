from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict, List
from collections import Counter, defaultdict
from sql_engine.llm_utils.query_guards import resolve_measure_override
from sql_engine.llm_utils.param_gen import llm_make_params, load_metadata
from sql_engine.llm_utils.validator import validate_measure_group_consistency, enforce_deterministic_measures, validate_cell_lookup

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
    # same type required
    if type(a) != type(b):
        return False

    # dict → compare keys + recurse
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(exact_match(a[k], b[k]) for k in a)

    # list → order-insensitive comparison
    if isinstance(a, list):
        if len(a) != len(b):
            return False

        # try simple sorted comparison first (fast path)
        try:
            return sorted(a) == sorted(b)
        except TypeError:
            # fallback for nested/unhashable items
            b_used = [False] * len(b)
            for item_a in a:
                found = False
                for i, item_b in enumerate(b):
                    if not b_used[i] and exact_match(item_a, item_b):
                        b_used[i] = True
                        found = True
                        break
                if not found:
                    return False
            return True

    # everything else → direct equality
    return a == b

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_path", required=True, help="Path to test_questions.json")
    ap.add_argument("--out_path", default="eval/paramgen_results.jsonl")
    ap.add_argument("--meta_path", default="storage/metadata/demographics_facts_metadata.json")
    ap.add_argument("--max_repairs", type=int, default=2)
    args = ap.parse_args()

    meta = load_metadata(args.meta_path)
    items = [normalize_gold(x) for x in load_questions_json(args.eval_path)]

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = len(items)
    cat_ok = 0
    json_and_schema_ok = 0
    query_exact_ok = 0

    rows = []

    # Overall failure counts
    failure_counts = Counter()

    # Per-category stats
    per_category = defaultdict(lambda: {
        "total": 0,
        "schema_pass": 0,
        "category_match": 0,
        "query_exact_match": 0,
        "failure_types": Counter(),
    })

    for ex in items:
        qid = ex.get("id")
        question = ex["question"]
        print("QUESTION", question)
        gold_cat = ex["category"]
        gold_query = ex.get("query", {})

        measure_override = resolve_measure_override(question, meta)
        print("MEASURE OVERRIDE")
        print(measure_override)


        rec: Dict[str, Any] = {
            "id": qid,
            "question": question,
            "gold_category": gold_cat,
            "gold_query": gold_query,
        }

        per_category[gold_cat]["total"] += 1

        try:
            if measure_override and "force_query" in measure_override:
                pred = {
                    "category": measure_override["force_category"],
                    "query": measure_override["force_query"],

                }
            else:
                pred = llm_make_params(question, meta, max_repairs=args.max_repairs, constraints=measure_override)
                # for cases with measure group
                pred = enforce_deterministic_measures(pred, question, meta)

            rec["pred"] = pred
            
            pred_cat = pred.get("category")

            if pred_cat == "cell_lookup":
                validate_cell_lookup(pred, meta)

            pred_query = pred.get("query")

            # basic JSON/schema presence
            schema_pass = isinstance(pred_cat, str) and isinstance(pred_query, dict)

            # semantic validation: measure_group consistency
            group_ok, group_reason = validate_measure_group_consistency(pred, meta)

            rec["schema_pass"] = schema_pass and group_ok
            rec["schema_reason"] = "ok" if (schema_pass and group_ok) else (
                "missing_category_or_query" if not schema_pass else group_reason
            )

            if rec["schema_pass"]:
                json_and_schema_ok += 1
                per_category[gold_cat]["schema_pass"] += 1

            rec["pred_category"] = pred_cat
            rec["category_match"] = (pred_cat == gold_cat)
            if rec["category_match"]:
                cat_ok += 1
                per_category[gold_cat]["category_match"] += 1

            # Only do exact query match if schema/semantic validation passed
            rec["query_exact_match"] = rec["schema_pass"] and exact_match(pred_query, gold_query)
            if rec["query_exact_match"]:
                query_exact_ok += 1
                per_category[gold_cat]["query_exact_match"] += 1

            # failure type
            if not schema_pass:
                failure_type = "schema_missing"
            elif not group_ok:
                failure_type = "measure_group_invalid"
            elif not rec["category_match"]:
                failure_type = "category_mismatch"
            elif not rec["query_exact_match"]:
                failure_type = "query_mismatch"
            else:
                failure_type = "success"

            rec["failure_type"] = failure_type
            failure_counts[failure_type] += 1
            per_category[gold_cat]["failure_types"][failure_type] += 1

        except Exception as e:
            rec["error"] = str(e)
            rec["schema_pass"] = False
            rec["category_match"] = False
            rec["query_exact_match"] = False
            rec["failure_type"] = "exception"

            failure_counts["exception"] += 1
            per_category[gold_cat]["failure_types"]["exception"] += 1

        rows.append(rec)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("=== ParamGen Eval (vs test_questions.json) ===")
    print(f"Total: {total}")
    print(f"Schema present: {json_and_schema_ok}/{total} = {json_and_schema_ok/total:.2%}")
    print(f"Category exact: {cat_ok}/{total} = {cat_ok/total:.2%}")
    print(f"Query exact: {query_exact_ok}/{total} = {query_exact_ok/total:.2%}")
    print()

    print("=== Overall Failure Types ===")
    for k, v in failure_counts.most_common():
        print(f"{k}: {v}/{total} = {v/total:.2%}")
    print()

    print("=== Per-Category Breakdown ===")
    for cat, stats in sorted(per_category.items()):
        cat_total = stats["total"]
        print(f"[{cat}] total={cat_total}")
        print(f"  schema_pass:      {stats['schema_pass']}/{cat_total} = {stats['schema_pass']/cat_total:.2%}")
        print(f"  category_match:   {stats['category_match']}/{cat_total} = {stats['category_match']/cat_total:.2%}")
        print(f"  query_exact:      {stats['query_exact_match']}/{cat_total} = {stats['query_exact_match']/cat_total:.2%}")
        print("  failure_types:")
        for ft, count in stats["failure_types"].most_common():
            print(f"    - {ft}: {count}/{cat_total} = {count/cat_total:.2%}")
        print()

    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()