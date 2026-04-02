# Chunk documents, compute embeddings, build and
# save versioned index to storage/

# rag/build_index.py
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
import argparse
from .llm_settings import rag_llm
rag_llm()
import json
import shutil
from typing import List
from utils.text_utils import csv_record_to_text


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "data" / "normalized"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pdf", "docx", "csv"], required=True,
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--base", default=str(DEFAULT_BASE),
                   help="Base directory containing normalized/<source>/rag_chunks.jsonl")
    p.add_argument("--chunks-name", required=False,
        help="Filename for document chunks JSONL inside base/<source>/ (default: rag_chunks.jsonl)")
    p.add_argument("--table-rows-name", required=False,
        help="Filename for table rows JSONL inside base/<source>/doc_tables/ (default: table_rows.jsonl)")
    return p.parse_args()


def load_index(index_dir: Path) -> VectorStoreIndex:
    if not index_dir.exists():
        raise FileNotFoundError(f"Index not found at {index_dir}. Run build_index first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    return load_index_from_storage(storage_context)

def get_index(path: str, format):
    index_path = Path(path, format)
    print("[INDEX] using index_dir:", index_path)
    return load_index(index_path)

def _resolve_jsonl_paths(path_or_pattern: Path) -> List[Path]:
    """
    Supports:
      - direct file path
      - directory path (loads *.jsonl within)
      - glob pattern (e.g. rag_chunks*.jsonl)
    Returns sorted list of file paths.
    """
    # 1) Exact file
    if path_or_pattern.exists() and path_or_pattern.is_file():
        return [path_or_pattern]

    # 2) Directory -> all *.jsonl
    if path_or_pattern.exists() and path_or_pattern.is_dir():
        files = sorted(path_or_pattern.glob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No *.jsonl files found in directory: {path_or_pattern}")
        return files

    # 3) Treat as glob pattern
    parent = path_or_pattern.parent
    pattern = path_or_pattern.name
    if parent.exists() and parent.is_dir():
        files = sorted(parent.glob(pattern))
        files = [p for p in files if p.is_file() and p.suffix == ".jsonl"]
        if not files:
            raise FileNotFoundError(f"No files matched pattern: {path_or_pattern}")
        return files

    raise FileNotFoundError(f"Path/pattern not found: {path_or_pattern}")

def build_index(jsonl_path: Path, index_dir: Path, kind: str = "doc"):
    jsonl_files = _resolve_jsonl_paths(jsonl_path)

    docs = []
    for jf in jsonl_files:
        print("Files being index:", jf)
        with open(jf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                if kind == "doc":
                    text = (rec.get("text") or rec.get("content") or "").strip()
                    if not text:
                        continue

                    meta = {
                        "id": rec.get("id"),
                        "doc_id": rec.get("doc_id"),
                        "source_path": rec.get("source_path"),
                        "section_path": rec.get("section_path"),
                        "section_index": rec.get("section_index"),
                        "title": rec.get("title"),
                        "path_text": rec.get("path_text"),
                        "block_type": rec.get("block_type"),
                        "block_index": rec.get("block_index"),
                        "page": rec.get("page")
                    }

                    if isinstance(rec.get("extra"), dict):
                        meta.update(rec["extra"])

                elif kind == "row":
                    if rec.get("table_index") == None:
                        return 0
                    text = (rec.get("search_text") or "").strip()
                    if not text:
                        continue

                    meta = {
                        "id": rec.get("row_id"),
                        "block_type": rec.get("block_type", "table_row"),
                        "table_id": rec.get("table_id"),
                        "source_file": rec.get("source_file"),
                        "source_type": rec.get("source_type"),
                        "table_index": rec.get("table_index"),
                        "row_index": rec.get("row_index"),
                        "row_label": rec.get("row_label"),
                        "section_path": rec.get("section_path"),
                        "caption": rec.get("caption"),
                        "header_terms": rec.get("header_terms"),
                        "row_values": rec.get("row_values"),
                        "page": rec.get("page")
                    }

                elif kind == "csv":
                    text = csv_record_to_text(rec)
                    if not text.strip():
                        continue

                    meta = {
                        "orig_row_id": rec.get("orig_row_id"),
                        "orig_col_id": rec.get("orig_col_id"),
                        "block_type": "table_cell",
                        "source_file": rec.get("source_file"),
                        "source_type": "csv_jsonl",
                        "raw_row": rec.get("raw_row"),
                        "raw_col": rec.get("raw_col"),
                        "label": rec.get("label") or rec.get("geo"),
                        # "geo": rec.get("geo") or rec.get("label"),
                        # "measure": rec.get("measure"),
                        # "subject": rec.get("subject"),
                        # "stat_type": rec.get("stat_type"),
                        "year": rec.get("year"),
                        "unit": rec.get("unit"),
                        # "value": rec.get("value"),
                        "raw_value": rec.get("raw_value"),
                    }

                else:
                    raise ValueError(f"Unknown index kind: {kind}")

                docs.append(Document(text=text, metadata=meta))

    index = VectorStoreIndex.from_documents(docs)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    return index


def main():
    args = parse_args()

    base = Path(args.base)

    # Look for JSONL files inside base/<source>/
    rag_chunks_path = base / args.source
    table_rows_path = base / args.source / "doc_tables" / "table_rows.jsonl"   

    doc_index_dir = Path(PROJECT_ROOT / "storage/index" / args.source)
    row_index_dir = Path(PROJECT_ROOT / f"storage/index/{args.source}_tables") # for csv only 

    if row_index_dir.exists():
        shutil.rmtree(row_index_dir)

    if doc_index_dir.exists():
        shutil.rmtree(doc_index_dir)

    if not rag_chunks_path.exists():
        raise FileNotFoundError(f"Missing {rag_chunks_path}. Run build_corpus first.")
    ## Checks if table rows file exists - to create separate index
    if args.source == "docx" or "pdf":
        print(f"[build_index] Loading chunks from: {rag_chunks_path}")
        print(f"[build_index] Persisting index to: {doc_index_dir}")
        # Only build from rag_chunks for documents
        build_index(rag_chunks_path/"rag_chunks.jsonl", doc_index_dir, kind = "doc")
        print("[build_index] Done.")
    if table_rows_path.exists():
        # Only build from table_rows for table index
        print(f"[build_index] Loading chunks from: {table_rows_path}")
        print(f"[build_index] Persisting index to: {row_index_dir}")
        build_index(table_rows_path, row_index_dir, kind = "row")

    elif args.source == "csv":
        print("csv branch")
        print(f"[build_index] Loading chunks from: {rag_chunks_path}")
        print(f"[build_index] Persisting index to: {doc_index_dir}")
        build_index(rag_chunks_path, doc_index_dir, kind = "csv")


if __name__ == "__main__":
    main()