# rag/retrieval/policy_lookup.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from utils.retrieval_utils import get_node_text


CODE_RE = re.compile(r"\b([A-Z]{2,8}-\d+(?:\.\d+)*)\b")
STOP_WORDS = {
    "what", "does", "the", "is", "are", "about", "regarding", "tell", "me",
    "give", "details", "information", "define", "in", "of", "for", "to",
    "and", "a", "an", "on", "with", "regarding", "about"
}

WORD_RE = re.compile(r"[a-z0-9]+")

def tokenize_keywords(text: str) -> List[str]:
    toks = WORD_RE.findall((text or "").lower())
    return [t for t in toks if t not in STOP_WORDS and len(t) > 2]

def norm_code(c: str) -> str:
    return c.rstrip(".").strip()

def extract_code_from_query(q: str | None) -> str | None:
    if not q:
        return None
    m = CODE_RE.search(q)
    return m.group(1) if m else None

def build_code_map_from_index(index: Any) -> Dict[str, Any]:
    """
    Build code -> node map from the index docstore (robust across llama_index versions).
    Prefer metadata.title if present; else use node text.
    """
    code_map: Dict[str, Any] = {}
    docs = getattr(getattr(index, "docstore", None), "docs", None)
    if not isinstance(docs, dict):
        return code_map

    for _node_id, node in docs.items():
        md = getattr(node, "metadata", None) or {}
        title = (md.get("title") or "").strip()

        candidate = title
        if not candidate:
            if hasattr(node, "get_text"):
                candidate = node.get_text() or ""
            elif hasattr(node, "text"):
                candidate = node.text or ""

        m = CODE_RE.search(candidate)
        if not m:
            continue

        code = norm_code(m.group(1))
        # If duplicates exist, keep the first for now (can upgrade to list later)
        code_map.setdefault(code, node)

    return code_map

class NodeWithScoreShim:
    """Minimal wrapper so eval code that expects .node still works."""
    def __init__(self, node: Any, score: float = 1e9):
        self.node = node
        self.score = score


## more weight to rarer/longer terms by preferring longer keywords
def policy_keyword_bonus(query: str, node_like: Any) -> float:
    text = get_node_text(node_like)
    query_keywords = set(tokenize_keywords(query))
    chunk_keywords = set(tokenize_keywords(text))

    if not query_keywords or not chunk_keywords:
        return 0.0

    overlap = query_keywords & chunk_keywords
    if not overlap:
        return 0.0

    score = 0.0
    score += 2.0 * len(overlap)
    score += len(overlap) / max(1.0, len(query_keywords))
    score += 0.25 * sum(len(tok) for tok in overlap)

    return score


def retrieve_policy_lookup(
    index: Any,
    query: str,
    k_eval: int = 5,
    top_k_retrieve: int = 30,
    code_map: Optional[Dict[str, Any]] = None,
):
    """
    Deterministic policy lookup:
    - If query contains a code and exists in code_map: return it first.
    - Otherwise, fallback to vector retrieval.
    """
    retriever = index.as_retriever(similarity_top_k=top_k_retrieve)
    nodes = retriever.retrieve(query)

    code = extract_code_from_query(query)
    if code and code_map is not None:
        node = code_map.get(norm_code(code))
        if node is not None:
            injected = NodeWithScoreShim(node)
            # best-effort dedupe
            def nid(x: Any):
                n = getattr(x, "node", x)
                return getattr(n, "node_id", None) or getattr(n, "id_", None) or id(n)

            inj = nid(injected)
            nodes = [injected] + [r for r in nodes if nid(r) != inj]
    # rerank retrieved nodes for partial-code / keyword cases
    nodes = sorted(nodes, key=lambda r: getattr(r, "score", 0.0) + policy_keyword_bonus(query, r),
        reverse=True)

    return nodes[:k_eval]



