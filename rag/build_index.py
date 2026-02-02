# Chunk documents, compute embeddings, build and
# save versioned index to storage/

# rag/build_index.py
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
import argparse
from .settings import apply_settings
apply_settings()

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # adjust if needed
DEFAULT_BASE = PROJECT_ROOT / "data" / "normalized"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["pdf", "docx"], required=True,
                   help="Which normalized folder to read rag_chunks.jsonl from")
    p.add_argument("--base", default=str(DEFAULT_BASE),
                   help="Base directory containing normalized/<source>/rag_chunks.jsonl")
    p.add_argument("--index-dir", required=True,
                   help="Output directory where the index will be persisted")
    return p.parse_args()

def build_index(rag_chunks_path: Path, index_dir: Path):
    docs = SimpleDirectoryReader(rag_chunks_path).load_data()
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
    rag_chunks_path = base / args.source
    index_dir = Path(args.index_dir)

    if not rag_chunks_path.exists():
        raise FileNotFoundError(f"Missing {rag_chunks_path}. Run build_corpus first.")

    print(f"[build_index] Loading chunks from: {rag_chunks_path}")
    print(f"[build_index] Persisting index to: {index_dir}")

    build_index(rag_chunks_path, index_dir)
    print("[build_index] Done.")

if __name__ == "__main__":
    main()