# Chunk documents, compute embeddings, build and
# save versioned index to storage/

# rag/build_index.py
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from .settings import apply_settings
apply_settings()

INDEX_DIR = Path("storage/index/v1")

def build_index(data_dir: str = "data/processed"):
    docs = SimpleDirectoryReader(data_dir).load_data()
    index = VectorStoreIndex.from_documents(docs)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))
    return index


def load_index() -> VectorStoreIndex:
    if not INDEX_DIR.exists():
        raise FileNotFoundError(f"Index not found at {INDEX_DIR}. Run build_index first.")
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    return load_index_from_storage(storage_context)

if __name__ == "__main__":
    build_index()
    print(f"✅ Index built and saved to {INDEX_DIR}")