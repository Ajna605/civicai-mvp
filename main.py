# End-to-end demo entry point
# - load index
# - accept question
# - print answer + sources

import time
from rag.settings import apply_settings
from rag.query_engine import query_civicai
from sql_engine.sql_query_engine import query_sql
import torch
import argparse
import json
from pathlib import Path
from typing import List

from llama_index.core import Document, VectorStoreIndex, StorageContext

PROJECT_ROOT = Path(__file__).resolve().parents[0]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "storage" / "index"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["pdf", "docx"], required=True,
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--engine", choices=["rag", "sql"], required=True,
                   help="Which engine to run")
    return p.parse_args()

DOC_CATEGORIES = ["policy_lookup", "section_lookup", "table_lookup", "general_summary"]
TABULAR_CATEGORIES = ["cell_lookup", "aggregation", "row_filter", "chart_request"]

if __name__ == "__main__":
    args = parse_args()
    # q = "Does the Coral Gables plan specify housing density limits?"

    # q = "What is the restriction in regards to residential development throughout the coastal area of East of Old Cutler Road?"
    # q = "Does the Coral Gables plan specify housing density limits?"
    # q = "Explain what is mentioned in Policy FLU-1.1.2."
    # q = "What does the document say about Policy ADM-1.5.3.?"
    # q = "Who are partners of the City?"
    #Q2
    # q = "What are the residential density limits in the coastal area east of Old Cutler Road"
    q = "What is the restriction in regard to residential development throughout the coastal area of East of Old Cutler Road?"
    # Q5
    # q =  "What is in the table showing Recreation facilities radius standard?"
    #Q6
    # q = "Does the Coral Gables plan specify housing density limits?"

    if args.engine == "rag":
        apply_settings()     # loads model once

        index_path = Path(DEFAULT_BASE, args.format)

        start = time.time()
       
        print(query_civicai(q, index_path))
        end = time.time()
        print("Time taken: ", (end - start)/60, "minutes")

    else:
        print(query_sql(q, index_path))
