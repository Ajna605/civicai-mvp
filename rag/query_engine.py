# Retrieval logic (top_k, merging), query execution,
# logging of retrieved sections


# rag/query_engine.py
from llama_index.core import PromptTemplate
from .build_index import load_index

QA_PROMPT = PromptTemplate(
"""You are CivicAI. Answer the question using ONLY the provided context.
Rules:
- Be concise (max 180 words)
- Do NOT repeat yourself
- If the document does not mention the asked item, say that ONCE.

Format:
Answer:
Reference:

Context:
{context_str}

Question: {query_str}
Answer:

"""
)




def query_civicai(query: str):
    index = load_index()
    query_engine = index.as_query_engine(similarity_top_k=3, response_mode="compact")
    response = query_engine.query(query)

    refs = []
    for sn in response.source_nodes[:2]:
        md = sn.node.metadata or {}
        refs.append({
            "source": md.get("file_name") or md.get("source") or "unknown",
            "page": md.get("page_label") or md.get("page") or None,
            "snippet": sn.node.get_text()[:200].replace("\n", " ")
        })

    return {"answer": response.response, "references": refs}

