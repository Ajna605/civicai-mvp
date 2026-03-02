# evaluation/eval_runner_structured.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Any, Dict

from sql_engine.query_engine import run_structured_query

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--tests", required=True)
    p.add_argument("--out", default="eval_outputs/structured/results.jsonl")
    return p.parse_args()

def numeric_pass(pred: Any, exp: Any, tol: Dict[str, Any]) -> bool:
    if pred is None:
        return False
    abs_tol = float(tol.get("abs", 0.0))
    rel_tol = float(tol.get("rel", 0.0))
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
    db_path = Path(args.db)
    tests_path = Path(args.tests)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    passed = 0


    tests_path = Path(args.tests)
    text = tests_path.read_text(encoding="utf-8-sig").strip()
    tests = json.loads(text)  

    for t in tests:
        f_out = out_path.open("w", encoding="utf-8") 
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
                pred_val = res["data"].get("value")
                ok = numeric_pass(pred_val, exp.get("value"), exp.get("tolerance", {"abs": 0.0}))
                if not ok:
                    reason = f"numeric_mismatch pred={pred_val} exp={exp.get('value')}"
            elif cat == "row_filter":
                rows = res["data"].get("rows", [])
                first = rows[0]["label"] if rows else None
                ok = (first == exp.get("first_label"))
                if not ok:
                    reason = f"label_mismatch pred={first} exp={exp.get('first_label')}"
            elif cat == "chart_request":
                points = res["data"].get("points", [])
                min_points = int(exp.get("min_points", 0))
                ok = (len(points) >= min_points)
                if not ok:
                    reason = f"insufficient_points pred={len(points)} min={min_points}"
            else:
                ok = False
                reason = "no_scorer_for_category"

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