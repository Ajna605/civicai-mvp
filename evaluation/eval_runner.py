# Run evaluation loop, log pass/fail,
# flag repetition and hallucination

import argparse
import csv
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple


def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def contains_any(text: str, needles: List[str]) -> bool:
    t = normalize(text)
    return any(normalize(n) in t for n in (needles or []) if n and n.strip())


def count_contains(text: str, needles: List[str]) -> int:
    t = normalize(text)
    return sum(1 for n in (needles or []) if n and normalize(n) in t)


def load_tests(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_retrieved_nodes(query: str, top_k: int):
    """
    Retrieval-only: returns a list of nodes with .node.get_text() and .node.metadata
    Uses your existing index.
    """
    from rag.build_index import load_index  # local import so eval_runner works from repo root

    index = load_index()
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def is_relevant(node_text: str, expected: Dict[str, Any]) -> bool:
    """
    Define relevance:
    - If must_mention exists: relevant if chunk contains ALL must_mention tokens
      (strict for exact lookups)
    - Else if key_phrases exists: relevant if chunk contains ANY key phrase
      (thematic/broad questions)
    - Else: fallback to False (no ground truth provided)
    """
    must = expected.get("must_mention") or []
    key_phrases = expected.get("key_phrases") or []

    if must:
        # strict: require all must tokens present
        t = normalize(node_text)
        return all(normalize(m) in t for m in must if m and m.strip())

    if key_phrases:
        return contains_any(node_text, key_phrases)

    return False


def evaluate_one(test: Dict[str, Any], retrieved_nodes, k_eval: int) -> Dict[str, Any]:
    qid = test.get("id", "")
    question = test.get("question", "")
    ttype = test.get("type", "")

    expected = test.get("expected", {}) or {}
    must = expected.get("must_mention") or []
    should = expected.get("should_mention") or []
    should_not = expected.get("should_not_mention") or []

    # Consider only top K for metrics
    nodes_k = retrieved_nodes[:k_eval]

    # Compute relevance per rank
    relevances: List[bool] = []
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for r in nodes_k:
        txt = r.node.get_text() if hasattr(r, "node") else str(r)
        md = (r.node.metadata or {}) if hasattr(r, "node") else {}
        texts.append(txt)
        metas.append(md)
        relevances.append(is_relevant(txt, expected))

    # Hit@K
    hit_at_k = 1 if any(relevances) else 0

    # MRR@K
    mrr = 0.0
    if any(relevances):
        first_idx = relevances.index(True)  # 0-based
        mrr = 1.0 / float(first_idx + 1)

    # Precision@K
    precision = (sum(1 for x in relevances if x) / float(k_eval)) if k_eval > 0 else 0.0

    # Drift / forbidden mentions in top K
    # violation if ANY of should_not tokens appear in ANY top-K node
    violation = 1 if any(contains_any(txt, should_not) for txt in texts) else 0

    # Optional context: did we retrieve any chunk containing at least one should token?
    should_hit = 1 if (should and any(contains_any(txt, should) for txt in texts)) else 0

    # Helpful debugging: where was first relevant?
    first_relevant_rank = None
    if any(relevances):
        first_relevant_rank = relevances.index(True) + 1

    # Minimal “evidence”: top 2 retrieved snippets + first relevant snippet
    def snippet(s: str, n: int = 220) -> str:
        s = " ".join((s or "").split())
        return s[:n] + ("…" if len(s) > n else "")

    top_snips = [snippet(texts[i]) for i in range(min(2, len(texts)))]
    rel_snip = ""
    rel_source = ""
    if first_relevant_rank is not None:
        i = first_relevant_rank - 1
        rel_snip = snippet(texts[i])
        md = metas[i] or {}
        rel_source = md.get("file_name") or md.get("source") or "unknown"

    return {
        "id": qid,
        "type": ttype,
        "question": question,
        "k_eval": k_eval,
        "hit_at_k": hit_at_k,
        "mrr_at_k": round(mrr, 4),
        "precision_at_k": round(precision, 4),
        "first_relevant_rank": first_relevant_rank,
        "should_hit": should_hit if should else None,
        "should_not_violation": violation if should_not else None,
        "must_mention": must,
        "key_phrases": expected.get("key_phrases") or [],
        "top_snippets": top_snips,
        "first_relevant_source": rel_source or None,
        "first_relevant_snippet": rel_snip or None,
        "known_issue": bool(test.get("known_issue") or expected.get("known_issue")),
        "notes": test.get("notes", ""),
    }


def write_csv(rows: List[Dict[str, Any]], path: str):
    # Flatten some list fields for CSV
    fieldnames = [
        "id", "type", "k_eval", "hit_at_k", "mrr_at_k", "precision_at_k",
        "first_relevant_rank", "should_hit", "should_not_violation",
        "known_issue", "notes"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="test_questions.json", help="Path to test_questions.json")
    ap.add_argument("--top_k_retrieve", type=int, default=20, help="How many nodes to retrieve")
    ap.add_argument("--k_eval", type=int, default=5, help="Evaluate metrics on top K results")
    ap.add_argument("--out_dir", default="eval_outputs", help="Directory to write results")
    args = ap.parse_args()

    tests = load_tests(args.tests)

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.out_dir, f"results_{stamp}.json")
    out_csv = os.path.join(args.out_dir, f"results_{stamp}.csv")

    results: List[Dict[str, Any]] = []

    for t in tests:
        q = t.get("question", "")
        retrieved = get_retrieved_nodes(q, top_k=args.top_k_retrieve)
        res = evaluate_one(t, retrieved, k_eval=args.k_eval)
        results.append(res)

        # Console summary line (quick feedback)
        print(
            f"[{res['id']}] Hit@{args.k_eval}={res['hit_at_k']} "
            f"MRR@{args.k_eval}={res['mrr_at_k']} "
            f"P@{args.k_eval}={res['precision_at_k']} "
            f"FirstRel={res['first_relevant_rank']} "
            f"Violation={res['should_not_violation']}"
        )

    # Write outputs
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_csv(results, out_csv)

    # Overall averages (excluding known_issue if you want)
    scored = [r for r in results if not r.get("known_issue")]
    if scored:
        avg_hit = sum(r["hit_at_k"] for r in scored) / len(scored)
        avg_mrr = sum(r["mrr_at_k"] for r in scored) / len(scored)
        avg_p = sum(r["precision_at_k"] for r in scored) / len(scored)
        print("\nAverages (excluding known_issue=true):")
        print(f"  Hit@{args.k_eval}: {avg_hit:.3f}")
        print(f"  MRR@{args.k_eval}: {avg_mrr:.3f}")
        print(f"  Precision@{args.k_eval}: {avg_p:.3f}")

    print(f"\nWrote:\n  {out_json}\n  {out_csv}")


if __name__ == "__main__":
    main()

