from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence


def unwrap_node(node_like: Any) -> Any:
    return getattr(node_like, "node", node_like)


def safe_node_text(node_like: Any) -> str:
    node = unwrap_node(node_like)
    try:
        return node.get_text() or ""
    except Exception:
        pass
    if hasattr(node, "text"):
        try:
            return node.text or ""
        except Exception:
            pass
    return ""


def safe_node_meta(node_like: Any) -> Dict[str, Any]:
    node = unwrap_node(node_like)
    md = getattr(node, "metadata", None)
    if isinstance(md, dict):
        return md
    if isinstance(node, dict):
        return dict(node)
    return {}


def safe_node_id(node_like: Any) -> str:
    md = safe_node_meta(node_like)

    if isinstance(md.get("id"), str) and md["id"]:
        return md["id"]

    table_id = md.get("table_id")
    row_index = md.get("row_index")
    if table_id is not None and row_index is not None:
        return f"{table_id}::row::{row_index}"

    node = unwrap_node(node_like)
    if hasattr(node, "id_"):
        try:
            return node.id_ or ""
        except Exception:
            pass

    txt = safe_node_text(node_like)
    return f"anon::{hash(txt)}"


def get_retrieved_nodes(index: Any, question: str, top_k: int = 10) -> List[Any]:
    retriever = index.as_retriever(similarity_top_k=top_k)
    return list(retriever.retrieve(question))


_WORD_RE = re.compile(r"[a-z0-9]+")

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by",
    "define", "did", "do", "does",
    "for", "find", "from",
    "give",
    "how",
    "in", "is", "it",
    "me",
    "of", "on", "or",
    "say", "show",
    "tell", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "with",
    "about",
}


def norm_text(s: str) -> str:
    return " ".join((s or "").lower().split())


def tokenize(s: str) -> List[str]:
    toks = _WORD_RE.findall((s or "").lower())
    return [t for t in toks if t not in STOP_WORDS and len(t) > 1]


def token_set(s: str) -> set[str]:
    return set(tokenize(s))


def extract_query_phrases(question: str) -> List[str]:
    q = norm_text(question)
    if not q:
        return []

    q = re.sub(
        r"^(what|which|where|when|who|how|is|are|does|do|did|show|find|tell me|give me|define)\b",
        "",
        q,
    ).strip()
    q = re.sub(r"\?$", "", q).strip()

    phrases: List[str] = []
    if len(q.split()) >= 2:
        phrases.append(q)

    toks = [tok for tok in q.split() if tok not in STOP_WORDS]
    for n in (4, 3, 2):
        if len(toks) >= n:
            for i in range(0, len(toks) - n + 1):
                phr = " ".join(toks[i:i+n])
                if len(phr.split()) >= 2:
                    phrases.append(phr)

    seen = set()
    out = []
    for p in phrases:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def exact_substring_score(question: str, text: str) -> float:
    q = norm_text(question)
    t = norm_text(text)
    if not q or not t:
        return 0.0

    score = 0.0
    if q in t:
        score += 10.0

    for phr in extract_query_phrases(question):
        if phr and phr in t:
            score += 3.0

    return score


def token_overlap_score(question: str, text: str) -> float:
    q = token_set(question)
    t = token_set(text)
    if not q or not t:
        return 0.0

    overlap = q & t
    if not overlap:
        return 0.0

    return float(len(overlap)) + (len(overlap) / max(1.0, len(q)))


def numeric_overlap_score(question: str, text: str) -> float:
    q_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", question))
    t_nums = set(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    if not q_nums or not t_nums:
        return 0.0
    return 2.0 * float(len(q_nums & t_nums))


def looks_table_like_question(question: str) -> bool:
    q = norm_text(question)
    cues = [
        "table", "row", "column", "matrix", "schedule", "standard",
        "radius", "density", "acre", "units/acre", "fee", "fees",
        "mileage", "park type", "service radius", "land use",
    ]
    return any(c in q for c in cues)


def node_block_type(node_like: Any) -> Optional[str]:
    bt = safe_node_meta(node_like).get("block_type")
    return bt if isinstance(bt, str) and bt else None


def node_table_id(node_like: Any) -> Optional[str]:
    tid = safe_node_meta(node_like).get("table_id")
    return tid if isinstance(tid, str) and tid else None


def node_caption(node_like: Any) -> Optional[str]:
    cap = safe_node_meta(node_like).get("caption")
    return cap if isinstance(cap, str) and cap else None


def extract_retrieved_table_ids(nodes: Sequence[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for n in nodes:
        if node_block_type(n) not in {"table_summary", "table_row"}:
            continue
        tid = node_table_id(n)
        if tid and tid not in seen:
            seen.add(tid)
            out.append(tid)
    return out


def extract_retrieved_table_captions(nodes: Sequence[Any], max_captions: int = 3) -> List[str]:
    out: List[str] = []
    seen = set()
    for n in nodes:
        if node_block_type(n) != "table_summary":
            continue
        cap = node_caption(n)
        if cap:
            nc = norm_text(cap)
            if nc not in seen:
                seen.add(nc)
                out.append(cap)
        if len(out) >= max_captions:
            break
    return out


def dedupe_nodes(nodes: Sequence[Any]) -> List[Any]:
    seen = set()
    out = []
    for n in nodes:
        nid = safe_node_id(n)
        if nid in seen:
            continue
        seen.add(nid)
        out.append(n)
    return out


def candidate_query_variants(question: str, table_captions: Sequence[str]) -> List[str]:
    variants = [question]
    for cap in table_captions[:3]:
        variants.append(f"{question} {cap}")

    seen = set()
    out = []
    for q in variants:
        nq = norm_text(q)
        if nq and nq not in seen:
            seen.add(nq)
            out.append(q)
    return out


def retrieve_table_rows_for_question(
    question: str,
    row_index: Any,
    *,
    table_ids: Optional[Sequence[str]] = None,
    table_captions: Optional[Sequence[str]] = None,
    top_k: int = 20,
    fetch_multiplier: int = 4,
) -> List[Any]:
    if row_index is None:
        return []

    table_ids = list(table_ids or [])
    table_captions = list(table_captions or [])

    raw_hits: List[Any] = []
    for q in candidate_query_variants(question, table_captions):
        raw_hits.extend(get_retrieved_nodes(row_index, q, top_k=top_k * fetch_multiplier))

    unique_hits = dedupe_nodes(raw_hits)

    if table_ids:
        table_id_set = set(table_ids)
        preferred = [n for n in unique_hits if node_table_id(n) in table_id_set]
        others = [n for n in unique_hits if node_table_id(n) not in table_id_set]
        return preferred + others

    return unique_hits

def build_scoring_text(node_like: Any) -> str:
    """
    Build a compact scoring text.

    - section_text: use the actual chunk text
    - table_summary: use summary text + caption + headers
    - table_row: use only row-focused fields, not full search_text
    """
    txt = safe_node_text(node_like)
    md = safe_node_meta(node_like)
    bt = md.get("block_type")

    parts: List[str] = []

    if bt == "section_text":
        parts.append(txt)

    elif bt == "table_summary":
        parts.append(txt)

        cap = md.get("caption")
        if isinstance(cap, str) and cap:
            parts.append(cap)

        headers = md.get("header_terms")
        if isinstance(headers, list) and headers:
            parts.append(" ".join(str(x) for x in headers if x))

    elif bt == "table_row":
        cap = md.get("caption")
        if isinstance(cap, str) and cap:
            parts.append(cap)

        row_label = md.get("row_label")
        if isinstance(row_label, str) and row_label:
            parts.append(row_label)

        row_values = md.get("row_values")
        if isinstance(row_values, dict) and row_values:
            for k, v in row_values.items():
                if v:
                    parts.append(f"{k} {v}")

    else:
        parts.append(txt)

    return " ".join(p for p in parts if p).strip()


def deterministic_rerank_score(
    question: str,
    node_like: Any,
    *,
    locked_table_ids: Optional[Sequence[str]] = None,
) -> float:
    locked_table_ids = list(locked_table_ids or [])
    locked_set = set(locked_table_ids)

    md = safe_node_meta(node_like)
    bt = md.get("block_type")
    tid = md.get("table_id")

    text = build_scoring_text(node_like)

    score = 0.0
    score += exact_substring_score(question, text)
    score += token_overlap_score(question, text)
    score += numeric_overlap_score(question, text)

    if bt == "table_row":
        score += 8.0
    elif bt == "table_summary":
        score += 3.0
    elif bt == "section_text":
        score += 2.0

    if locked_set:
        if tid in locked_set:
            if bt == "table_row":
                score += 8.0
            elif bt == "table_summary":
                score += 3.0
        else:
            if bt == "table_row":
                score -= 4.0
            elif bt == "table_summary":
                score -= 2.0

    return score


def rerank_table_and_doc_hits(
    question: str,
    doc_nodes: Sequence[Any],
    row_nodes: Sequence[Any],
    *,
    final_top_k: Optional[int] = None,
    locked_table_ids: Optional[Sequence[str]] = None,
) -> List[Any]:
    merged = dedupe_nodes(list(doc_nodes) + list(row_nodes))

    ranked = sorted(
        merged,
        key=lambda n: deterministic_rerank_score(
            question,
            n,
            locked_table_ids=locked_table_ids,
        ),
        reverse=True,
    )

    if final_top_k is not None:
        return ranked[:final_top_k]
    return ranked


def select_locked_table_ids(doc_nodes: Sequence[Any]) -> List[str]:
    """
    Prefer top-ranked table_summary/table_row table_ids from initial doc retrieval.
    Usually lock to the first discovered table.
    """
    found: List[str] = []
    for n in doc_nodes:
        bt = node_block_type(n)
        if bt not in {"table_summary", "table_row"}:
            continue
        tid = node_table_id(n)
        if tid:
            found.append(tid)

    if not found:
        return []

    # strictest behavior: lock to first candidate only
    return [found[0]]


def merge_with_table_rows(
    question: str,
    retrieved_nodes: Sequence[Any],
    row_index: Any,
    *,
    top_k_rows: int = 10,
    final_top_k: Optional[int] = None,
) -> List[Any]:
    doc_nodes = list(retrieved_nodes)

    # If no row index exists, just return doc nodes.
    if row_index is None:
        return doc_nodes[:final_top_k] if final_top_k is not None else doc_nodes

    # Use any discovered table summaries as a soft constraint / boost,
    # but do not require them in order to search rows.
    locked_table_ids = select_locked_table_ids(doc_nodes)
    table_captions = extract_retrieved_table_captions(doc_nodes)

    row_nodes = retrieve_table_rows_for_question(
        question,
        row_index,
        table_ids=locked_table_ids,      # optional preference, not required
        table_captions=table_captions,   # optional query enrichment
        top_k=top_k_rows,
    )

    return rerank_table_and_doc_hits(
        question,
        doc_nodes,
        row_nodes,
        final_top_k=final_top_k,
        locked_table_ids=locked_table_ids,
    )


def table_aware_retrieve(
    question: str,
    *,
    doc_index: Any,
    row_index: Any = None,
    top_k_docs: int = 20,
    top_k_rows: int = 10,
    final_top_k: Optional[int] = None,
) -> List[Any]:
    doc_nodes = get_retrieved_nodes(doc_index, question, top_k=top_k_docs)

    return merge_with_table_rows(
        question,
        doc_nodes,
        row_index,
        top_k_rows=top_k_rows,
        final_top_k=final_top_k or top_k_docs,
    )


def debug_node_summary(node_like: Any) -> Dict[str, Any]:
    md = safe_node_meta(node_like)
    return {
        "id": safe_node_id(node_like),
        "block_type": md.get("block_type"),
        "table_id": md.get("table_id"),
        "caption": md.get("caption"),
        "row_label": md.get("row_label"),
        "preview": build_scoring_text(node_like) # [:220],
    }


def debug_ranked_nodes(nodes: Sequence[Any], limit: int = 10) -> List[Dict[str, Any]]:
    return [debug_node_summary(n) for n in list(nodes)[:limit]]