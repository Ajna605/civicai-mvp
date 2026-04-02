# rag/query_engine.py
import re
from .build_index import load_index
from .retrieval.policy_lookup import retrieve_policy_lookup, extract_code_from_query, build_code_map_from_index
from rag.retrieval.table_rerank import table_aware_retrieve
from rag.build_index import get_index
from rag.llm_settings import llm_verbalize_answer, conv_llm
from utils.retrieval_utils import format_context
from pathlib import Path

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","with","about","what","does","say",
    "document","policy","section","clause","article","chapter","table","figure","regarding"
}

ID_SEPS = set(["-", ".", "/", ":", "_", "§"])

def normalize_token(t: str) -> str:
    # Strip trailing punctuation that commonly appears in questions: "ADM-1.5.3.?" -> "ADM-1.5.3"
    return t.strip().strip(").,;:!?\"'[]{}")

def extract_rare_tokens(q: str) -> list[str]:
    raw = re.findall(r"[A-Za-z0-9§\-\._:/]+", q)
    out, seen = [], set()
    for t in raw:
        t = normalize_token(t)
        tl = t.lower()
        if len(t) < 4 or tl in STOPWORDS:
            continue
        if any(ch.isdigit() for ch in t) or any(sep in t for sep in ID_SEPS):
            if t not in seen:
                out.append(t)
                seen.add(t)
    return out

def is_lookup_question(q: str) -> bool:
    # Intent-based: if we see an ID-like token, treat as lookup
    tokens = extract_rare_tokens(q)
    return len(tokens) > 0

def snippet_around(text: str, token: str, window: int = 260) -> str:
    t = text or ""
    i = t.lower().find(token.lower())
    if i == -1:
        return " ".join(t.split())[:window]
    start = max(0, i - 80)
    end = min(len(t), i + window)
    return " ".join(t[start:end].split())

def ref_from(node) -> dict:
    md = getattr(node, "metadata", None) or {}
    txt = node.get_text() if hasattr(node, "get_text") else str(node)
    return {
        "source": md.get("file_name") or md.get("source") or "unknown",
        "page": md.get("page_label") or md.get("page") or None,
        "snippet": " ".join(txt.split())[:220] + ("…" if len(txt) > 220 else ""),
    }

def get_retrieved_nodes(index, query: str, top_k: int):
    retriever = index.as_retriever(similarity_top_k=top_k)
    return retriever.retrieve(query)

TOP_K = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = PROJECT_ROOT / "storage" / "index"
conv_llm()
def query_civicai(query: str, index_path:str, format:str):
    index = load_index(index_path)
    code_map = build_code_map_from_index(index)

    doc_index = get_index(DEFAULT_BASE, format)
    table_index = None # Optional

    try:
        table_index = get_index(DEFAULT_BASE, f"{format}_tables")
        print(f"Table Index loaded from {format}_tables")
    except Exception as e:
        print(f"[eval_runner_doc] Warning: could not load table index '{args.table_index}': {e}")
        table_index = None

    # LOOKUP MODE

    code = extract_code_from_query(query)
    if code:
        retrieved = retrieve_policy_lookup(
            doc_index,
            query,
            k_eval=1,                 # we only need best for lookup answer
            top_k_retrieve=30,
            code_map=code_map,
        )

        if not retrieved:
            retrieved = []
    
    else:
        retrieved = retrieved = table_aware_retrieve(
                question=query,
                doc_index=doc_index,
                row_index=table_index,
                top_k_docs=TOP_K,
                top_k_rows=20,
                final_top_k=TOP_K,
            )

    # THEMATIC / GENERAL MODE (simple for now)
    # Return top 2 chunks (cleaned) + references
    top_texts = []
    refs = []
    for nws in retrieved[:5]:
        top_texts.append(" ".join(nws.node.get_text().split())[:350])
        refs.append(ref_from(nws.node))

    return {"answer": llm_verbalize_answer(query, format_context(retrieved)), 
            "references": refs}

    # return {
    #     "answer": "\n\n".join(top_texts).strip(),
    #     "references": refs
    # }
