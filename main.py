# End-to-end demo entry point
# - load index
# - accept question
# - print answer + sources

import time
# from rag.llm_settings import rag_llm
from rag.query_engine import query_civicai
from sql_engine.query_engine import query_sql
import argparse
from pathlib import Path

from llama_index.core import Document, VectorStoreIndex, StorageContext

PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "storage" / "index"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["pdf", "docx", "csv"], required=True,
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--engine", choices=["rag", "sql"], required=True,
                   help="Which engine to run")
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

    q =  "Which community groups in Coral Gables are most likely to face the greatest healthcare access pressure over the next 5–10 years?"

    ## Insurance Questions
    # q = "What is the number of insured people in Coral Gables under 64 years?" # aggregation
    # q = "Show how disability affects the percent of people insured" # chart
    ## Demographic Questions
    # q = "How many males per 100 females?" # cell_lookup works
    # q = "What is the population under 24 years in Coral Gables?" # aggregation works
    # q = "What age range is most prominent in Coral Gables?" # row_filter #works
    # q = "Create a bar chart of age groups vs total population." # chart_request # works
    index_path = Path(DEFAULT_BASE, args.format)
    print("index path", index_path)
    if args.engine == "rag":
        # rag_llm()     # loads model once
        start = time.time()
        print(q)
        print(query_civicai(q, index_path, args.format))
        end = time.time()
        print("Time taken: ", (end - start), "seconds")

    else:
        print(q)
        start = time.time()
        print(query_sql(q, index_path))
        end = time.time()
        print("Time taken: ", (end - start), "seconds")