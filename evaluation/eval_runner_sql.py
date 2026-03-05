# evaluation/eval_runner_structured.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict
from sql_engine.llm_utils.param_gen import load_metadata, llm_make_params

from sql_engine.query_engine import run_structured_query

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--tests", required=True)
    p.add_argument("--out", default="eval_outputs/structured/results.jsonl")
    return p.parse_args()

def numeric_pass(pred: Any, exp: Any, tol) -> bool:
    # tol can be dict ({"abs":..., "rel":...}) OR a number (0, 1, 0.5)
    if pred is None:
        return False

    if tol is None:
        abs_tol = 0.0
        rel_tol = 0.0
    elif isinstance(tol, (int, float)):
        abs_tol = float(tol)
        rel_tol = 0.0
    elif isinstance(tol, dict):
        abs_tol = float(tol.get("abs", 0.0))
        rel_tol = float(tol.get("rel", 0.0))
    else:
        raise ValueError(f"Unsupported tolerance type: {type(tol)}")

    try:
        pred_f = float(pred)
        exp_f = float(exp)
    except Exception:
        return False

    if abs(pred_f - exp_f) <= abs_tol:
        return True
    if rel_tol > 0 and exp_f != 0:
        return abs(pred_f - exp_f) / abs(exp_f) <= rel_tol
    return False

if __name__ == "__main__":
    args = parse_args()
    # db_path = Path(args.db)

    db_path = Path(args.db).resolve()
    tests_path = Path(args.tests)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0


    tests_path = Path(args.tests)
    text = tests_path.read_text(encoding="utf-8-sig").strip()
    tests = json.loads(text)  
    with out_path.open("w", encoding="utf-8") as f_out:
        for t in tests:
            total += 1

            cat = t["category"]
            q = t.get("query", {})
            exp = t.get("expected", {})

            res = run_structured_query(db_path, cat, q)

            ok = False
            reason = None

            if not res.get("ok"):
                ok = False
                reason = res.get("error")

            else:
                if cat in {"cell_lookup", "aggregation"}:
                    pred_val = (res.get("data") or {}).get("value")

                    exp_nums = exp.get("expected_numbers")
                    exp_val = exp_nums[0] if isinstance(exp_nums, list) and exp_nums else exp_nums

                    tol = exp.get("tolerance", 0)
                    ok = numeric_pass(pred_val, exp_val, tol)

                    if not ok:
                        reason = f"numeric_mismatch pred={pred_val} exp={exp_val}"

                elif cat == "row_filter":
                    rows = (res.get("data") or {}).get("rows", [])
                    # Build a deterministic text haystack from returned rows
                    def row_text(r):
                        parts = []
                        for k in ("measure", "label", "subject", "stat_type"):
                            v = r.get(k)
                            if v is not None:
                                parts.append(str(v))
                        return " ".join(parts)

                    haystack = " | ".join(row_text(r) for r in rows)

                    must_all = (exp or {}).get("must_mention_all", []) or []
                    ok = all(str(p).lower() in haystack.lower() for p in must_all)

                    if not ok:
                        reason = f"must_mention_missing haystack={haystack}"

                elif cat == "chart_request":
                    points = (res.get("data") or {}).get("points", [])
                    min_points = int(exp.get("min_points", 0))
                    # Check min points
                    ok = len(points) >= exp.get("min_points", 0)
                    xs = [str(p.get("x", "")) for p in points]
                    haystack = " | ".join(xs)
                    must = exp.get("must_contain_labels", [])
                    # Check labels in horizontal axis
                    ok = all(label.lower() in haystack.lower() for label in must)
                    if not ok:
                        reason = f"insufficient_points pred={len(points)} min={min_points}"

                else:
                    ok = False
                    reason = f"no_scorer_for_category:{cat}"

            if ok:
                passed += 1

            out_row = {
                "id": t.get("id"),
                "category": cat,
                "pass": ok,
                "reason": reason,
                "question": t.get("question"),
                "query": q,
                "expected": exp,
                "result": res,
            }
            f_out.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"✅ Structured eval: {passed}/{total} passed")
    print(f"Results written to: {out_path}")