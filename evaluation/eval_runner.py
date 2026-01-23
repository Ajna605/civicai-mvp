<<<<<<< Updated upstream
# Run evaluation loop, log pass/fail,
# flag repetition and hallucination
=======
"""
eval_runner.py (rewritten)
- Adds chunk_len_chars + header_distance diagnostics
- Supports new test schema with: category + expected.must_mention_any/must_mention_all/key_phrases
- Computes metrics per-category and prints category summaries
- Applies acceptance criteria per category (CI-style gate) and exits nonzero if gate fails

Assumptions:
- LlamaIndex-style retrieval results where each item has `.node` and `.node.get_text()` and `.node.metadata`
- index.docstore.docs is dict[node_id -> node] with node.get_text()

Usage:
  python eval_runner.py --tests test_questions.json --k_eval 5 --diag_k 200
"""

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Text utils
# -----------------------------

def normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def contains_any(text: str, needles: List[str]) -> bool:
    t = normalize(text)
    return any(normalize(n) in t for n in (needles or []) if n and n.strip())


def contains_all(text: str, needles: List[str]) -> bool:
    t = normalize(text)
    return all(normalize(n) in t for n in (needles or []) if n and n.strip())


def load_tests(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Index + retrieval
# -----------------------------

def get_index():
    from rag.build_index import load_index
    return load_index()


def get_retrieved_nodes(index, query: str, top_k: int):
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)


def safe_node_text(r: Any) -> str:
    """
    Handles retrieval result wrappers.
    Expect r.node.get_text().
    """
    try:
        return (r.node.get_text() or "")
    except Exception:
        # fallback shapes
        if hasattr(r, "get_text"):
            try:
                return r.get_text() or ""
            except Exception:
                return ""
        if hasattr(r, "text"):
            return r.text or ""
        return ""


def safe_node_meta(r: Any) -> Dict[str, Any]:
    try:
        return r.node.metadata or {}
    except Exception:
        return {}


# -----------------------------
# Expected schema helpers
# -----------------------------

def expected_tokens(expected: Dict[str, Any]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns (must_any, must_all, key_phrases, should_not)
    """
    must_any = expected.get("must_mention_any") or []
    must_all = expected.get("must_mention_all") or []
    key_phrases = expected.get("key_phrases") or []
    should_not = expected.get("should_not_mention") or []
    return must_any, must_all, key_phrases, should_not


def is_relevant_lookup(node_text: str, expected: Dict[str, Any]) -> bool:
    must_any, must_all, _, _ = expected_tokens(expected)

    # If must_all provided, all must be present.
    if must_all:
        if not contains_all(node_text, must_all):
            return False

    # If must_any provided, at least one must be present.
    if must_any:
        return contains_any(node_text, must_any)

    # If neither provided, fallback: not relevant (lookup tasks should define tokens)
    return False


def is_relevant_summary(node_text: str, expected: Dict[str, Any]) -> bool:
    _, _, key_phrases, _ = expected_tokens(expected)
    return contains_any(node_text, key_phrases) if key_phrases else False


def is_relevant(node_text: str, category: str, expected: Dict[str, Any]) -> bool:
    if category in ("policy_lookup", "section_lookup", "table_lookup"):
        return is_relevant_lookup(node_text, expected)
    if category in ("general_summary",):
        return is_relevant_summary(node_text, expected)
    # Unknown category: try both signals
    must_any, must_all, key_phrases, _ = expected_tokens(expected)
    if must_any or must_all:
        return is_relevant_lookup(node_text, expected)
    if key_phrases:
        return is_relevant_summary(node_text, expected)
    return False


def primary_token_for_diagnostics(expected: Dict[str, Any]) -> Optional[str]:
    """
    Used for corpus_match_count + token_pos.
    Preference:
      - first must_any, else first must_all, else None
    """
    must_any, must_all, _, _ = expected_tokens(expected)
    if must_any:
        return must_any[0]
    if must_all:
        return must_all[0]
    return None


# -----------------------------
# Diagnostics: corpus_match_count, token_pos, chunk_len, header_distance
# -----------------------------

def find_token_pos(text: str, token: str) -> Optional[int]:
    """
    Returns character position in normalized text (stable across casing/spacing).
    Note: because normalization collapses whitespace, this is an approximate diagnostic.
    """
    if not token:
        return None
    t = normalize(text)
    tok = normalize(token)
    i = t.find(tok)
    return i if i != -1 else None


def chunk_len_chars(text: str) -> int:
    return len(text or "")


_HEADER_REGEXES_DEFAULT = [
    r"^(?:GOAL|OBJECTIVE|POLICY|ACTION)\b.*$",
    r"^\d+(?:\.\d+){1,4}\s+\S.*$",
    r"^[A-Z][A-Z0-9 \-–—,:/()]{8,}$",
]


def header_distance(text: str, header_regexes: Optional[List[str]] = None) -> Optional[int]:
    """
    Finds char offset from start of chunk to first header-like line. None if not found.
    """
    if not text:
        return None
    header_regexes = header_regexes or _HEADER_REGEXES_DEFAULT

    lines = text.splitlines(True)  # keepends
    offset = 0
    for line in lines:
        stripped = line.strip()
        if stripped:
            for rgx in header_regexes:
                if re.match(rgx, stripped):
                    return offset
        offset += len(line)
    return None


def corpus_match_count(index, token: str) -> int:
    """Count chunks in the entire index whose text contains token (case-insensitive normalized)."""
    if not token:
        return 0
    tok = normalize(token)
    cnt = 0
    for _, node in index.docstore.docs.items():
        try:
            if tok in normalize(node.get_text()):
                cnt += 1
        except Exception:
            continue
    return cnt


# -----------------------------
# Failure classification
# -----------------------------

def classify_failure(
    category: str,
    corpus_cnt: Optional[int],
    hit_at_k: int,
    hit_at_diag: Optional[int],
    token_pos: Optional[int],
    precision_at_k: float,
    noise_threshold: float = 0.4
) -> str:
    """
    Deterministic, mutually-exclusive labeling.
    """
    lookup_like = category in ("policy_lookup", "section_lookup", "table_lookup")
    summary_like = category in ("general_summary",)

    if lookup_like:
        if corpus_cnt == 0:
            return "TOKEN_MISSING_FROM_INDEX"
        if hit_at_k == 1:
            if token_pos is not None and token_pos > 200:
                return "CHUNK_DILUTION"
            return "OK"
        # not in top-k
        if hit_at_diag == 1:
            return "RANKING_FAILURE"
        return "NOT_RETRIEVED_AT_DIAG_K"

    if summary_like:
        if hit_at_k == 0:
            return "NO_RELEVANT_CONTEXT_IN_TOP_K"
        if precision_at_k < noise_threshold:
            return "NOISE_HIGH"
        return "OK"

    # Unknown category fallback (treat as summary-ish)
    if hit_at_k == 0:
        return "NO_RELEVANT_CONTEXT_IN_TOP_K"
    if precision_at_k < noise_threshold:
        return "NOISE_HIGH"
    return "OK"


# -----------------------------
# Acceptance criteria per category
# -----------------------------

DEFAULT_ACCEPTANCE: Dict[str, Dict[str, Any]] = {
    # Strict exact ID lookups
    "policy_lookup": {
        "token_preservation_required": True,  # corpus_match_count > 0
        "min_hit_at_k": 0.90,                 # avg over non-known-issue tests in category
        "min_mrr_at_k": 0.75,
        "max_chunk_dilution_rate": 0.10,
        "max_ranking_failure_rate": 0.10,
    },
    # Narrow factual lookups (can be a bit noisier than policy IDs)
    "section_lookup": {
        "token_preservation_required": False,  # often no single canonical ID; still computed
        "min_hit_at_k": 0.80,
        "min_mrr_at_k": 0.60,
        "max_chunk_dilution_rate": 0.15,
        "max_ranking_failure_rate": 0.15,
    },
    # Tables are allowed to fail until preprocessing is fixed; set lenient defaults
    "table_lookup": {
        "token_preservation_required": False,
        "min_hit_at_k": 0.70,
        "min_mrr_at_k": 0.50,
        "max_chunk_dilution_rate": 0.20,
        "max_ranking_failure_rate": 0.20,
    },
    # Broad topical retrieval: focus on avoiding empty + too-noisy
    "general_summary": {
        "max_no_relevant_rate": 0.15,
        "max_noise_high_rate": 0.25,
    },
}


# -----------------------------
# Evaluation
# -----------------------------

def snippet(s: str, n: int = 220) -> str:
    s = " ".join((s or "").split())
    return s[:n] + ("…" if len(s) > n else "")


def select_diag_node_lookup(retrieved_nodes_k: List[Any], expected: Dict[str, Any]) -> Optional[Any]:
    """
    For lookup tasks:
      - first relevant node in top-k (contains must tokens)
      - else top-1 node
    """
    if not retrieved_nodes_k:
        return None
    for r in retrieved_nodes_k:
        if is_relevant_lookup(safe_node_text(r), expected):
            return r
    return retrieved_nodes_k[0]


def select_diag_node_summary(retrieved_nodes_k: List[Any], expected: Dict[str, Any]) -> Optional[Any]:
    """
    For summary tasks:
      - first node matching any key phrase
      - else top-1
    """
    if not retrieved_nodes_k:
        return None
    if is_relevant_summary(safe_node_text(retrieved_nodes_k[0]), expected):
        return retrieved_nodes_k[0]
    for r in retrieved_nodes_k:
        if is_relevant_summary(safe_node_text(r), expected):
            return r
    return retrieved_nodes_k[0]


def evaluate_one(
    index,
    test: Dict[str, Any],
    retrieved_nodes: List[Any],
    k_eval: int,
    diag_k: int
) -> Dict[str, Any]:
    qid = test.get("id", "")
    question = test.get("question", "")

    # New: category (fallback to old `type` if you haven't migrated yet)
    category = test.get("category") or test.get("type") or "unknown"

    expected = test.get("expected", {}) or {}
    should = expected.get("should_mention") or []
    should_not = expected.get("should_not_mention") or []

    nodes_k = retrieved_nodes[:k_eval]

    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    relevances: List[bool] = []

    for r in nodes_k:
        txt = safe_node_text(r)
        md = safe_node_meta(r)
        texts.append(txt)
        metas.append(md)
        relevances.append(is_relevant(txt, category, expected))

    hit_at_k = 1 if any(relevances) else 0

    mrr = 0.0
    if hit_at_k:
        first_idx = relevances.index(True)
        mrr = 1.0 / float(first_idx + 1)

    precision = (sum(1 for x in relevances if x) / float(k_eval)) if k_eval > 0 else 0.0

    violation = 1 if (should_not and any(contains_any(txt, should_not) for txt in texts)) else 0
    should_hit = 1 if (should and any(contains_any(txt, should) for txt in texts)) else 0
    first_relevant_rank = (relevances.index(True) + 1) if hit_at_k else None

    top_snips = [snippet(texts[i]) for i in range(min(2, len(texts)))]

    rel_snip = None
    rel_source = None
    token_pos = None
    diag_match_count = None
    corpus_cnt = None
    hit_at_diag = None

    # Diagnostics node for chunk_len/header_distance/token_pos
    diag_node = None
    if category in ("policy_lookup", "section_lookup", "table_lookup"):
        diag_node = select_diag_node_lookup(nodes_k, expected)
    elif category in ("general_summary",):
        diag_node = select_diag_node_summary(nodes_k, expected)
    else:
        diag_node = nodes_k[0] if nodes_k else None

    diag_text = safe_node_text(diag_node) if diag_node is not None else ""
    diag_chunk_len = chunk_len_chars(diag_text)
    diag_header_dist = header_distance(diag_text)

    # First relevant snippet/source (if any)
    if first_relevant_rank is not None:
        i = first_relevant_rank - 1
        rel_snip = snippet(texts[i])
        md = metas[i] or {}
        rel_source = md.get("file_name") or md.get("source") or "unknown"

    # Lookup-style diagnostics: corpus_match_count + diag_k retrieval
    primary_tok = primary_token_for_diagnostics(expected)
    lookup_like = category in ("policy_lookup", "section_lookup", "table_lookup")

    if lookup_like and primary_tok:
        corpus_cnt = corpus_match_count(index, primary_tok)

        diag_nodes = get_retrieved_nodes(index, question, top_k=diag_k)
        diag_rels = [is_relevant(safe_node_text(n), category, expected) for n in diag_nodes]
        hit_at_diag = 1 if any(diag_rels) else 0
        diag_match_count = sum(1 for x in diag_rels if x)

        # token_pos should be computed on the FIRST relevant chunk if available, else on diag_node
        if first_relevant_rank is not None:
            token_pos = find_token_pos(texts[first_relevant_rank - 1], primary_tok)
        else:
            token_pos = find_token_pos(diag_text, primary_tok)

    # Broad summary: keep corpus_cnt/hit_at_diag None
    failure_label = classify_failure(
        category=category,
        corpus_cnt=corpus_cnt,
        hit_at_k=hit_at_k,
        hit_at_diag=hit_at_diag,
        token_pos=token_pos,
        precision_at_k=precision,
        noise_threshold=0.4,
    )

    return {
        "id": qid,
        "category": category,
        "question": question,
        "k_eval": k_eval,
        "hit_at_k": hit_at_k,
        "mrr_at_k": round(mrr, 4),
        "precision_at_k": round(precision, 4),
        "first_relevant_rank": first_relevant_rank,
        "should_hit": should_hit if should else None,
        "should_not_violation": violation if should_not else None,

        # expected fields (for debug)
        "must_mention_any": expected.get("must_mention_any") or [],
        "must_mention_all": expected.get("must_mention_all") or [],
        "key_phrases": expected.get("key_phrases") or [],

        # snippets/debug
        "top_snippets": top_snips,
        "first_relevant_source": rel_source,
        "first_relevant_snippet": rel_snip,

        # diagnostics
        "corpus_match_count": corpus_cnt,
        "hit_at_diag_k": hit_at_diag,
        "diag_k": diag_k if lookup_like else None,
        "token_pos": token_pos,
        "diag_match_count": diag_match_count,
        "chunk_len_chars": diag_chunk_len,
        "header_distance": diag_header_dist,

        # outcome labels
        "failure_label": failure_label,

        "known_issue": bool(test.get("known_issue") or expected.get("known_issue")),
        "notes": test.get("notes", ""),
    }


# -----------------------------
# Reporting
# -----------------------------

def write_csv(rows: List[Dict[str, Any]], path: str):
    fieldnames = [
        "id", "category", "k_eval", "hit_at_k", "mrr_at_k", "precision_at_k",
        "first_relevant_rank", "should_hit", "should_not_violation",
        "corpus_match_count", "hit_at_diag_k", "diag_k", "diag_match_count", "token_pos",
        "chunk_len_chars", "header_distance",
        "failure_label", "known_issue", "notes"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def summarize_by_category(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Returns summary stats per category (excluding known_issue by default in scoring).
    """
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cat[r.get("category", "unknown")].append(r)

    summaries: Dict[str, Dict[str, Any]] = {}
    for cat, items in by_cat.items():
        scored = [x for x in items if not x.get("known_issue")]
        denom = len(scored) if scored else 0

        avg_hit = sum(x["hit_at_k"] for x in scored) / denom if denom else None
        avg_mrr = sum(x["mrr_at_k"] for x in scored) / denom if denom else None
        avg_p = sum(x["precision_at_k"] for x in scored) / denom if denom else None

        failure_counts = Counter(x.get("failure_label", "UNKNOWN") for x in scored) if scored else Counter()

        # Rates for key failure labels
        def rate(label: str) -> Optional[float]:
            if not denom:
                return None
            return sum(1 for x in scored if x.get("failure_label") == label) / denom

        token_missing_rate = rate("TOKEN_MISSING_FROM_INDEX")
        ranking_failure_rate = rate("RANKING_FAILURE")
        chunk_dilution_rate = rate("CHUNK_DILUTION")
        no_relevant_rate = rate("NO_RELEVANT_CONTEXT_IN_TOP_K")
        noise_high_rate = rate("NOISE_HIGH")

        # Token preservation (for lookup-like categories) - based on corpus_match_count > 0
        token_preservation_rate = None
        if cat in ("policy_lookup", "section_lookup", "table_lookup") and denom:
            token_preservation_rate = sum(1 for x in scored if (x.get("corpus_match_count") or 0) > 0) / denom

        summaries[cat] = {
            "total": len(items),
            "scored": denom,
            "avg_hit_at_k": avg_hit,
            "avg_mrr_at_k": avg_mrr,
            "avg_precision_at_k": avg_p,
            "token_preservation_rate": token_preservation_rate,
            "rates": {
                "TOKEN_MISSING_FROM_INDEX": token_missing_rate,
                "RANKING_FAILURE": ranking_failure_rate,
                "CHUNK_DILUTION": chunk_dilution_rate,
                "NO_RELEVANT_CONTEXT_IN_TOP_K": no_relevant_rate,
                "NOISE_HIGH": noise_high_rate,
            },
            "failure_label_counts": dict(failure_counts),
        }

    return summaries


def apply_acceptance_criteria(
    category_summaries: Dict[str, Dict[str, Any]],
    acceptance: Dict[str, Dict[str, Any]]
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Returns (overall_pass, checks[]).
    Each check: {category, name, value, threshold, pass}
    """
    checks: List[Dict[str, Any]] = []

    def add_check(cat: str, name: str, value: Optional[float], threshold: float, pass_if: str):
        # pass_if: "ge" or "le"
        if value is None:
            # No scored items -> treat as pass (nothing to gate), but record it
            checks.append({"category": cat, "name": name, "value": None, "threshold": threshold, "pass": True})
            return
        ok = (value >= threshold) if pass_if == "ge" else (value <= threshold)
        checks.append({"category": cat, "name": name, "value": value, "threshold": threshold, "pass": ok})

    for cat, cfg in acceptance.items():
        summ = category_summaries.get(cat)
        if not summ:
            continue

        avg_hit = summ.get("avg_hit_at_k")
        avg_mrr = summ.get("avg_mrr_at_k")
        rates = summ.get("rates", {}) or {}

        if cat in ("policy_lookup", "section_lookup", "table_lookup"):
            if cfg.get("token_preservation_required"):
                tpr = summ.get("token_preservation_rate")
                add_check(cat, "token_preservation_rate", tpr, 1.0, "ge")

            add_check(cat, "avg_hit_at_k", avg_hit, float(cfg.get("min_hit_at_k", 0.0)), "ge")
            add_check(cat, "avg_mrr_at_k", avg_mrr, float(cfg.get("min_mrr_at_k", 0.0)), "ge")

            add_check(cat, "chunk_dilution_rate", rates.get("CHUNK_DILUTION"), float(cfg.get("max_chunk_dilution_rate", 1.0)), "le")
            add_check(cat, "ranking_failure_rate", rates.get("RANKING_FAILURE"), float(cfg.get("max_ranking_failure_rate", 1.0)), "le")

        if cat in ("general_summary",):
            add_check(cat, "no_relevant_rate", rates.get("NO_RELEVANT_CONTEXT_IN_TOP_K"), float(cfg.get("max_no_relevant_rate", 1.0)), "le")
            add_check(cat, "noise_high_rate", rates.get("NOISE_HIGH"), float(cfg.get("max_noise_high_rate", 1.0)), "le")

    overall_pass = all(c["pass"] for c in checks) if checks else True
    return overall_pass, checks


def _rates_to_display_for_category(cat: str) -> List[str]:
    """
    Only show failure rates that make sense for that category.
    """
    lookup_cats = {"policy_lookup", "section_lookup", "table_lookup"}
    summary_cats = {"general_summary"}

    if cat in lookup_cats:
        return ["TOKEN_MISSING_FROM_INDEX", "RANKING_FAILURE", "CHUNK_DILUTION", "NOT_RETRIEVED_AT_DIAG_K"]
    if cat in summary_cats:
        return ["NO_RELEVANT_CONTEXT_IN_TOP_K", "NOISE_HIGH"]
    # unknown: show all common
    return [
        "TOKEN_MISSING_FROM_INDEX",
        "RANKING_FAILURE",
        "CHUNK_DILUTION",
        "NOT_RETRIEVED_AT_DIAG_K",
        "NO_RELEVANT_CONTEXT_IN_TOP_K",
        "NOISE_HIGH",
    ]


def print_category_summaries(category_summaries: Dict[str, Dict[str, Any]]):
    print("\n================ CATEGORY SUMMARIES ================")
    for cat, summ in sorted(category_summaries.items(), key=lambda kv: kv[0]):
        print(f"\n[{cat}] total={summ['total']} scored={summ['scored']}")

        if summ["avg_hit_at_k"] is not None:
            print(f"  avg_hit@k: {summ['avg_hit_at_k']:.3f}")
            print(f"  avg_mrr@k: {summ['avg_mrr_at_k']:.3f}")
            print(f"  avg_precision@k: {summ['avg_precision_at_k']:.3f}")
        else:
            print("  (no scored items in this category)")

        # Only meaningful for lookup categories (and only if present)
        if summ.get("token_preservation_rate") is not None:
            print(f"  token_preservation_rate: {summ['token_preservation_rate']:.3f}")

        # Only display relevant rates for this category
        rates = summ.get("rates", {}) or {}
        for label in _rates_to_display_for_category(cat):
            v = rates.get(label)
            if v is not None:
                print(f"  rate[{label}]: {v:.3f}")

        # Show top failures, but only among the relevant set for this category
        flc = summ.get("failure_label_counts", {}) or {}
        if flc:
            allowed = set(_rates_to_display_for_category(cat)) | {"OK"}  # include OK for context
            filtered = [(k, v) for k, v in flc.items() if k in allowed]
            filtered.sort(key=lambda kv: (-kv[1], kv[0]))
            top = filtered[:5]
            if top:
                print("  top_failures:", ", ".join(f"{k}:{v}" for k, v in top))
            else:
                # fallback if none match allowed labels
                top2 = sorted(flc.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
                print("  top_failures:", ", ".join(f"{k}:{v}" for k, v in top2))

    print("====================================================\n")



def print_gate_checks(overall_pass: bool, checks: List[Dict[str, Any]]):
    print("\n================ ACCEPTANCE (CI GATE) ================")
    print(f"OVERALL: {'PASS' if overall_pass else 'FAIL'}")
    for c in checks:
        name = c["name"]
        cat = c["category"]
        thr = c["threshold"]
        val = c["value"]
        ok = c["pass"]
        if val is None:
            print(f" - {'PASS' if ok else 'FAIL'} | {cat} | {name}: (no scored items) threshold={thr}")
        else:
            print(f" - {'PASS' if ok else 'FAIL'} | {cat} | {name}: {val:.3f} threshold={thr}")
    print("======================================================\n")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tests", default="test_questions.json")
    ap.add_argument("--top_k_retrieve", type=int, default=20)
    ap.add_argument("--k_eval", type=int, default=5)
    ap.add_argument("--diag_k", type=int, default=200, help="Diagnostic retrieval depth for lookup categories")
    ap.add_argument("--out_dir", default="eval_outputs")
    ap.add_argument("--fail_on_gate", action="store_true", help="Exit nonzero if acceptance criteria fail")
    args = ap.parse_args()

    tests = load_tests(args.tests)
    os.makedirs(args.out_dir, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.out_dir, f"results_{stamp}.json")
    out_csv = os.path.join(args.out_dir, f"results_{stamp}.csv")
    out_gate = os.path.join(args.out_dir, f"gate_{stamp}.json")

    index = get_index()

    results: List[Dict[str, Any]] = []

    for t in tests:
        q = t.get("question", "")
        retrieved = get_retrieved_nodes(index, q, top_k=args.top_k_retrieve)
        res = evaluate_one(index, t, retrieved, k_eval=args.k_eval, diag_k=args.diag_k)
        results.append(res)

        # Per-test line
        print(
            f"[{res['id']}] cat={res['category']} "
            f"Hit@{args.k_eval}={res['hit_at_k']} "
            f"MRR@{args.k_eval}={res['mrr_at_k']} "
            f"P@{args.k_eval}={res['precision_at_k']} "
            f"Fail={res['failure_label']} "
            f"CorpusCnt={res['corpus_match_count']} "
            f"Hit@{args.diag_k}={res['hit_at_diag_k']} "
            f"len={res['chunk_len_chars']} "
            f"hdrDist={res['header_distance']}"
        )

    # Write outputs
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    write_csv(results, out_csv)

    # Summaries + gate
    cat_summaries = summarize_by_category(results)
    print_category_summaries(cat_summaries)

    overall_pass, checks = apply_acceptance_criteria(cat_summaries, DEFAULT_ACCEPTANCE)
    print_gate_checks(overall_pass, checks)

    with open(out_gate, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall_pass": overall_pass,
                "checks": checks,
                "category_summaries": cat_summaries,
                "acceptance": DEFAULT_ACCEPTANCE,
            },
            f,
            indent=2
        )

    print(f"\nWrote:\n  {out_json}\n  {out_csv}\n  {out_gate}")

    if args.fail_on_gate and not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
>>>>>>> Stashed changes
