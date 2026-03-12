# Chunk documents, compute embeddings, build and
# save versioned index to storage/

# rag/build_index.py
from pathlib import Path
from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage
import argparse
from .settings import apply_settings
apply_settings()
import json
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "data" / "normalized"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pdf", "docx", "csv"], required=True,
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--base", default=str(DEFAULT_BASE),
                   help="Base directory containing normalized/<source>/rag_chunks.jsonl")
    return p.parse_args()

def build_index(jsonl_path: Path, index_dir: Path, kind: str = "doc"):
    if jsonl_path.is_dir():
        raise ValueError(f"Expected a file path, got directory: {jsonl_path}")

    docs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
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
                }

                if isinstance(rec.get("extra"), dict):
                    meta.update(rec["extra"])

            elif kind == "row":
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
                }

            else:
                raise ValueError(f"Unknown index kind: {kind}")

            docs.append(Document(text=text, metadata=meta))

    index = VectorStoreIndex.from_documents(docs)
    index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    return index


def load_index(index_dir: Path) -> VectorStoreIndex:
    if not index_dir.exists():
        raise FileNotFoundError(f"Index not found at {index_dir}. Run build_index first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    return load_index_from_storage(storage_context)

def main():
    args = parse_args()

    base = Path(args.base)
    rag_chunks_path = base / args.source / "rag_chunks.jsonl"
    table_rows_path = base / args.source / "doc_tables" / "table_rows.jsonl"    
    doc_index_dir = Path(PROJECT_ROOT / "storage/index" / args.source)
    row_index_dir = Path(PROJECT_ROOT / f"storage/index/{args.source}_tables")

    if row_index_dir.exists():
        shutil.rmtree(row_index_dir)

    if doc_index_dir.exists():
        shutil.rmtree(doc_index_dir)

    if not rag_chunks_path.exists():
        raise FileNotFoundError(f"Missing {rag_chunks_path}. Run build_corpus first.")
    ## Checks if table rows file exists - to create separate index
    if table_rows_path.exists():
        print(f"[build_index] Loading chunks from: {table_rows_path}")
        print(f"[build_index] Persisting index to: {row_index_dir}")
        build_index(table_rows_path, row_index_dir, kind = "row")


    print(f"[build_index] Loading chunks from: {rag_chunks_path}")
    print(f"[build_index] Persisting index to: {doc_index_dir}")

    build_index(rag_chunks_path, doc_index_dir, kind = "doc")
    print("[build_index] Done.")

if __name__ == "__main__":
    main()