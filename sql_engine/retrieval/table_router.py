# Find right table to retrieve from
# from query find the right table from index of CSVs in duckDb

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from llama_index.core import StorageContext, load_index_from_storage

# If you have an apply_settings() somewhere (so embeddings match how index was built),
# import and call it. If not, you can remove these two lines.
from rag.llm_settings import rag_llm  # adjust if your settings live elsewhere
rag_llm()


@dataclass
class TableScore:
    table: str
    score_sum: float
    hits: int

    @property
    def avg(self) -> float:
        return self.score_sum / self.hits if self.hits else 0.0


class TableRouter:
    def __init__(self, index_dir: str | Path, similarity_top_k: int = 20):
        self.index_dir = Path(index_dir).expanduser().resolve()
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index dir not found: {self.index_dir}")

        storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
        self.index = load_index_from_storage(storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=similarity_top_k)

    def route(self, question: str, top_tables: int = 3) -> List[TableScore]:
        nodes = self.retriever.retrieve(question)

        # rank tables by sum(score) and hits
        score_sum: Dict[str, float] = defaultdict(float)
        hits: Dict[str, int] = defaultdict(int)

        for n in nodes:
            # LlamaIndex NodeWithScore: n.score, n.node.metadata
            md = getattr(n.node, "metadata", {}) or {}
            table = md.get("table")
            if not table:
                continue
            s = float(n.score or 0.0)
            score_sum[table] += s
            hits[table] += 1

        ranked = sorted(
            [TableScore(t, score_sum[t], hits[t]) for t in score_sum.keys()],
            key=lambda x: (x.score_sum, x.hits),
            reverse=True,
        )
        return ranked[:top_tables]

def parse_args():
    p = argparse.ArgumentParser(description="Route a natural-language question to DuckDB table(s) using a vector index.")
    p.add_argument("--index-dir", "--index_dir", dest="index_dir", required=True, help="Persisted csv_duckdb index dir")
    p.add_argument("--question", required=True, help="User question to route")
    p.add_argument("--similarity-top-k", "--similarity_top_k", dest="similarity_top_k", type=int, default=20)
    p.add_argument("--top-tables", "--top_tables", dest="top_tables", type=int, default=3)
    return p.parse_args()


# def main():
#     args = parse_args()
#     router = TableRouter(index_dir=args.index_dir, similarity_top_k=args.similarity_top_k)
#     ranked = router.route(args.question, top_tables=args.top_tables)

#     if not ranked:
#         print("[table_router] No table candidates found.")
#         return

#     print("[table_router] Table candidates:")
#     for i, r in enumerate(ranked, 1):
#         print(f"  {i}. {r.table}  hits={r.hits}  score_sum={r.score_sum:.4f}  avg={r.avg:.4f}")


if __name__ == "__main__":
    main()