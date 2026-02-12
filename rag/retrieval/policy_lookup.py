# rag/retrieval/policy_lookup.py
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

CODE_RE = re.compile(r"\b([A-Z]{2,8}-\d+(?:\.\d+)*)\b")

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

    return nodes[:k_eval]
