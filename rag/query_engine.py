# rag/query_engine.py
import re
from .build_index import load_index

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

def query_civicai(query: str):
    index = load_index()

    # Retrieve a decent pool; we'll filter/rank within it
    retriever = index.as_retriever(similarity_top_k=30)
    nodes = retriever.retrieve(query)  # list[NodeWithScore]

    if not nodes:
        return {"answer": "No relevant context retrieved.", "references": []}

    # LOOKUP MODE
    if is_lookup_question(query):
        tokens = extract_rare_tokens(query)

        # Find nodes that literally contain any token
        matching = []
        for nws in nodes:
            text = nws.node.get_text()
            if any(tok.lower() in text.lower() for tok in tokens):
                matching.append(nws)

        if not matching:
            return {
                "answer": "I didn’t retrieve text that explicitly mentions the specific item you asked about.",
                "references": []
            }

        # Prefer the match where the token appears earliest (more “focused” chunk)
        def match_rank(nws):
            txt = nws.node.get_text().lower()
            positions = [txt.find(tok.lower()) for tok in tokens if txt.find(tok.lower()) != -1]
            return min(positions) if positions else 10**9

        matching.sort(key=match_rank)
        best_nws = matching[0]
        best_text = best_nws.node.get_text()

        # Use the most salient token (first one)
        tok = tokens[0]
        ans = snippet_around(best_text, tok)

        return {
            "answer": ans,
            "references": [ref_from(best_nws.node)]
        }

    # THEMATIC / GENERAL MODE (simple for now)
    # Return top 2 chunks (cleaned) + references
    top_texts = []
    refs = []
    for nws in nodes[:5]:
        top_texts.append(" ".join(nws.node.get_text().split())[:350])
        refs.append(ref_from(nws.node))

    return {
        "answer": "\n\n".join(top_texts).strip(),
        "references": refs
    }
