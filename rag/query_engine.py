# Retrieval logic (top_k, merging), query execution,
# logging of retrieved sections


# rag/query_engine.py
from llama_index.core import PromptTemplate
from .build_index import load_index

QA_PROMPT = PromptTemplate(
"""Use ONLY the provided context.
Rules:
- Be concise (max 180 words)
- Do NOT repeat yourself
- If the document does not mention the asked item, say that ONCE and suggest the single best place to check next.

Format:
Answer:
Evidence:
What to check next:

Context:
{context_str}

Question: {query_str}
"""
)

def query_civicai(question: str, top_k: int = 3) -> str:
    index = load_index()
    qe = index.as_query_engine(similarity_top_k=top_k, text_qa_template=QA_PROMPT)
    response = qe.query(question)
    return str(response)
