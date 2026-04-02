from typing import Any, Dict, List, Optional

def get_node_text(node_like: Any) -> str:
    node = getattr(node_like, "node", node_like)

    md = getattr(node, "metadata", None) or {}
    title = (md.get("title") or "").strip()

    txt = ""
    if hasattr(node, "get_text"):
        txt = node.get_text() or ""
    elif hasattr(node, "text"):
        txt = node.text or ""

    return f"{title} {txt}".strip()


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



def format_context(nodes):
    blocks = []
    for i, n in enumerate(nodes, 1):
        md = safe_node_meta(n)
        title = md.get("caption") or md.get("path_text") or md.get("id") or "Chunk"
        bt = md.get("block_type", "chunk")
        src = md.get("source", md.get("source_type", "unknown"))
        page = md.get("page")

        snippet = (get_node_text(n) or "").strip()
        snippet = snippet[:700]  # hard trim

        blocks.append(f"[{i}] ({bt}) {title}\nSource: {src} Page: {page}\nExcerpt: {snippet}")
    return "\n\n".join(blocks)