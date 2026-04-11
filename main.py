# End-to-end demo entry point
# - load index
# - accept question
# - print answer + sources

import asyncio
import time
from utils.forecast_utils import is_forecast_question
import argparse
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext

PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "storage" / "index"

_DEFAULT_PDF = str(
    PROJECT_ROOT / "data" / "raw"
    / "Coral Gables - Comprehensive Plan - 12-25-2025 20-55 - selectable.pdf"
)
_DEFAULT_RAGANYTHING_CACHE = str(
    PROJECT_ROOT / "storage" / "raganything" / "mineru_cache"
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["pdf", "docx", "csv"], default="pdf",
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--engine", choices=["rag", "sql", "raganything"], required=True,
                   help="Which engine to run")
    p.add_argument("--pdf_path", default=_DEFAULT_PDF,
                   help="Path to PDF corpus (used by raganything engine)")
    p.add_argument("--raganything_cache_dir", default=_DEFAULT_RAGANYTHING_CACHE,
                   help="Cache directory for raganything engine "
                        "(default: storage/raganything/mineru_cache/)")
    return p.parse_args()

DOC_CATEGORIES = ["policy_lookup", "section_lookup", "table_lookup", "general_summary"]
TABULAR_CATEGORIES = ["cell_lookup", "aggregation", "row_filter", "chart_request"]

if __name__ == "__main__":
    args = parse_args()
    # Document questions

    # from tests 
    # q = "What does the document say about Policy ADM-1.5.3.?"
    # q = "What does the Coastal Management Element say about marinas?"
    # q = "How does the city plan to be recognized as a green and sustainable community?" #2.4
    # q = "Does the Coral Gables plan specify residential density limits?" #problematic

    ## Insurance Questions
    # q = "What is the number of insured people in Coral Gables under 64 years?" # aggregation
    # q = "Show how disability affects the percent of people insured" # chart
    ## Demographic Questions
    # q = "How many males per 100 females?" # cell_lookup works
    # q = "What is the population under 24 years in Coral Gables?" # aggregation works
    # q = "What age range is most prominent in Coral Gables?" # row_filter #works
    # q = "Create a bar chart of age groups vs total population." # chart_request # works

    ## BIG Question
    q =  "Which community groups in Coral Gables are most likely to face the greatest healthcare access pressure over the next 5–10 years?"

    index_path = Path(DEFAULT_BASE, args.format)
    print("index path", index_path)

    if is_forecast_question(q):
        from sql_engine.analytics_sql.executor_analytics import query_analytics
        print("analyzeeee")
        print(query_analytics(q, index_path))
    elif args.engine == "rag":
        from rag.query_engine import query_civicai
        start = time.time()
        print(q)
        print(query_civicai(q, index_path, args.format))
        end = time.time()
        print("Time taken: ", (end - start), "seconds")

    elif args.engine == "raganything":
        from evaluation.eval_runner_doc_raganything import (
            _build_raganything,
            _process_pdf_if_needed,
            _retrieve_chunks,
            RagChunkWrapper,
        )
        from rag.llm_settings import llm_verbalize_answer, conv_llm
        from utils.retrieval_utils import format_context

        conv_llm()

        async def _run_raganything(question, pdf_path, cache_dir):
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)

            rag = _build_raganything(cache_path)
            await _process_pdf_if_needed(rag, pdf_path, cache_path)

            raw_chunks = await _retrieve_chunks(rag, question, top_k=5)
            retrieved = [RagChunkWrapper(r) for r in raw_chunks]

            if not retrieved:
                return {"answer": "No relevant context retrieved.", "references": []}

            context = format_context(retrieved)
            answer = llm_verbalize_answer(question, context)

            refs = []
            for r in retrieved:
                snippet = (r.text or "")[:220]
                refs.append({"snippet": snippet + ("…" if len(r.text or "") > 220 else "")})

            return {"answer": answer, "references": refs}

        start = time.time()
        print(q)
        result = asyncio.run(
            _run_raganything(q, args.pdf_path, args.raganything_cache_dir)
        )
        print(result)
        end = time.time()
        print("Time taken: ", (end - start), "seconds")

    else:
        from sql_engine.executor_simple import query_sql
        print(q)
        start = time.time()
        print(query_sql(q, index_path))
        end = time.time()
        print("Time taken: ", (end - start), "seconds")
