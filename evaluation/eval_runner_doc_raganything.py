"""
eval_runner_doc_raganything.py
Retrieval-only evaluation runner that uses HKUDS/RAG-Anything (MinerU) as the
retriever and reuses all scoring / diagnostic logic from eval_runner_doc.py.

No LLM is called – only whether retrieved chunks contain expected tokens is checked
(the same accept/reject semantics already in evaluate_one()).

Install dependencies before running:
    pip install raganything
    # MinerU (CPU or GPU):
    pip install mineru         # CPU
    pip install mineru[gpu]    # CUDA GPU (recommended for large PDFs)
    See evaluation/README_raganything.md for full setup and version notes.

Usage (from repo root):
    python evaluation/eval_runner_doc_raganything.py \\
        --tests evaluation/test_questions_doc.json \\
        --pdf_path "data/raw/Coral Gables - Comprehensive Plan - 12-25-2025 20-55 - selectable.pdf" \\
        --k_eval 5 \\
        --top_k_retrieve 20 \\
        --out_dir eval_outputs_raganything
"""

import argparse
import asyncio
import inspect
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repo root is in sys.path so we can import project packages
# regardless of how the script is invoked.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Reuse all scoring / diagnostic helpers from the existing runner.
# Importing eval_runner_doc triggers rag_llm() which sets up
# Settings.embed_model = HuggingFaceEmbedding("sentence-transformers/all-MiniLM-L6-v2").
# We reuse that same embedding instance for RAG-Anything so both runners
# operate on identical vector spaces.
# ---------------------------------------------------------------------------
from evaluation.eval_runner_doc import (  # noqa: E402
    DEFAULT_ACCEPTANCE,
    apply_acceptance_criteria,
    evaluate_one,
    print_category_summaries,
    print_gate_checks,
    summarize_by_category,
    write_csv,
)
from utils.text_utils import load_tests  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = _REPO_ROOT
_DEFAULT_PDF = str(
    PROJECT_ROOT
    / "data"
    / "raw"
    / "Coral Gables - Comprehensive Plan - 12-25-2025 20-55 - selectable.pdf"
)
_DEFAULT_CACHE = str(PROJECT_ROOT / "storage" / "raganything" / "mineru_cache")
_PROCESSED_MARKER = ".rag_processed"
_EMBED_DIM = 384          # sentence-transformers/all-MiniLM-L6-v2 output dimension
_EMBED_MAX_TOKENS = 512   # model max token length


# ===========================================================================
# Cache helpers
# ===========================================================================

def _is_cached(cache_dir: Path) -> bool:
    """Return True when the PDF has already been processed and the chunk KV
    store is present on disk."""
    marker = cache_dir / _PROCESSED_MARKER
    chunks_kv = cache_dir / "kv_store_text_chunks.json"
    return marker.exists() and chunks_kv.exists()


# ===========================================================================
# Embedding  –  reuse the same model already loaded by llama_index / rag_llm()
# ===========================================================================

def _make_embedding_func():
    from llama_index.core import Settings  # noqa: PLC0415
    from lightrag.utils import EmbeddingFunc  # noqa: PLC0415
    import numpy as np
    import asyncio

    EMBED_DIM = 384  # all-MiniLM-L6-v2

    async def _embed(texts: List[str]) -> np.ndarray:
        # offload sync embedding to a thread so we don't block asyncio loop
        def _run():
            embs = Settings.embed_model.get_text_embedding_batch(texts, show_progress=False)
            arr = np.asarray(embs, dtype=np.float32)
            # Ensure shape (N, 384)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        arr = await asyncio.to_thread(_run)

        # optional strict check
        if arr.shape[1] != EMBED_DIM:
            raise ValueError(f"Unexpected embedding dim {arr.shape[1]} (expected {EMBED_DIM})")

        return arr

    return EmbeddingFunc(embedding_dim=EMBED_DIM, func=_embed)

# ===========================================================================
# Dummy LLM  –  never called in retrieval-only mode
# ===========================================================================

async def _dummy_llm(
    prompt: str,
    system_prompt: Optional[str] = None,
    history_messages: Optional[list] = None,
    **kwargs: Any,
) -> str:
    """Placeholder LLM function.  This runner is retrieval-only; the LLM is
    never invoked.  RAGAnythingConfig requires a callable, so we supply this
    no-op."""
    return "RETRIEVAL_ONLY_NO_LLM"


# ===========================================================================
# RAG-Anything setup
# ===========================================================================

def _build_raganything(cache_dir: Path):
    from raganything import RAGAnything, RAGAnythingConfig  # noqa: PLC0415
    from lightrag.lightrag import LightRAG  # noqa: PLC0415

    embedding_func = _make_embedding_func()

    config = RAGAnythingConfig(
        working_dir=str(cache_dir),
        parser_output_dir=str(cache_dir / "parser_output"),
        parser="mineru",
        parse_method="auto",
    )

    lightrag = LightRAG(
        working_dir=str(cache_dir / "lightrag"),
        embedding_func=embedding_func,
        llm_model_func=_dummy_llm,   # optional but should be callable
    )

    return RAGAnything(
        config=config,
        lightrag=lightrag,
        llm_model_func=_dummy_llm,
        vision_model_func=_dummy_llm,
    )


async def _process_pdf_if_needed(rag: Any, pdf_path: str, cache_dir: Path) -> None:
    """Run MinerU parsing + LightRAG ingestion if not already cached.

    Version compatibility (raganything 1.2.10+):
    ``process_document_complete`` forwards keyword arguments to the MinerU
    parser.  Older versions accepted ``enable_table_processing`` /
    ``enable_image_processing`` / ``enable_equation_processing`` while
    raganything 1.2.10 exposes the underlying MinerU flags directly:
        table (bool)   – enable table extraction  (default True)
        formula (bool)  – enable equation extraction (default True)
    Image processing is not a MinerU-level toggle in 1.2.10, so we skip it
    entirely when the old name is unsupported.

    We inspect ``process_document_complete`` and, where available, the inner
    ``MineruParser._run_mineru_command`` to decide which argument names to use.
    """
    if _is_cached(cache_dir):
        print(f"[raganything] Cache found at '{cache_dir}'. Skipping PDF processing.")
        return

    mineru_output = cache_dir / "mineru_output"
    mineru_output.mkdir(parents=True, exist_ok=True)

    # --- Detect supported keyword arguments for document processing ----------
    proc_sig = inspect.signature(rag.process_document_complete)
    proc_params = set(proc_sig.parameters.keys())

    # Also inspect the low-level MinerU command if available
    mineru_params: set = set()
    try:
        from raganything.parser import MineruParser  # noqa: PLC0415
        mineru_params = set(
            inspect.signature(MineruParser._run_mineru_command).parameters.keys()
        )
    except Exception:
        pass  # older versions may not expose this; fall back to proc_params only

    processing_kwargs: Dict[str, Any] = {}

    # Table processing – enabled by default
    if "enable_table_processing" in proc_params:
        processing_kwargs["enable_table_processing"] = True
    elif "table" in proc_params or "table" in mineru_params:
        processing_kwargs["table"] = True

    # Equation / formula processing
    if "enable_equation_processing" in proc_params:
        processing_kwargs["enable_equation_processing"] = False
    elif "formula" in proc_params or "formula" in mineru_params:
        processing_kwargs["formula"] = True

    # Image processing – only pass when the old-style flag is recognised
    if "enable_image_processing" in proc_params:
        processing_kwargs["enable_image_processing"] = False
    # else: MinerU 1.2.10 doesn't expose an image toggle → omit

    print(f"[raganything] Processing PDF: {pdf_path}")
    print(f"[raganything] process_document_complete kwargs: {processing_kwargs}")
    print("[raganything] First run may take several minutes (MinerU + LightRAG ingestion)…")
    await rag.process_document_complete(
        file_path=pdf_path,
        output_dir=str(mineru_output),
        **processing_kwargs,
    )

    (cache_dir / _PROCESSED_MARKER).touch()
    print(f"[raganything] PDF processed. Cache saved to '{cache_dir}'.")


# ===========================================================================
# Chunk loading  –  for corpus_match_count diagnostics
# ===========================================================================

def _load_all_chunks(cache_dir: Path) -> Dict[str, str]:
    """
    Load every text chunk from LightRAG's KV store file and return
    a mapping {chunk_id -> chunk_text}.  Used by the index adapter so that
    evaluate_one's corpus_match_count() can iterate all corpus chunks.
    """
    chunks_file = cache_dir / "kv_store_text_chunks.json"
    if not chunks_file.exists():
        print(
            f"[raganything] Warning: '{chunks_file}' not found. "
            "Corpus diagnostics (corpus_match_count) will be unavailable."
        )
        return {}

    with open(chunks_file, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    result: Dict[str, str] = {}
    for chunk_id, chunk_data in raw.items():
        if isinstance(chunk_data, dict):
            text = chunk_data.get("content") or chunk_data.get("text") or ""
        else:
            text = str(chunk_data)
        result[chunk_id] = text

    print(f"[raganything] Loaded {len(result)} chunks from KV store.")
    return result


# ===========================================================================
# Retrieval
# ===========================================================================

async def _retrieve_chunks(rag: Any, query: str, top_k: int) -> List[dict]:
    """
    Retrieve top-k chunks from LightRAG's internal chunk vector DB using
    plain semantic (naive) similarity – no graph traversal, no reranking.

    Returns a list of chunk dicts, each containing at least a 'content' key
    with the chunk text.
    """
    results = await rag.lightrag.chunks_vdb.query(query, top_k=top_k)
    return results or []


# ===========================================================================
# Adapter objects  –  bridge between LightRAG chunks and evaluate_one()
# ===========================================================================

class _NodeLike:
    """Inner node object compatible with safe_node_text / safe_node_meta."""

    def __init__(self, text: str, metadata: dict) -> None:
        self._text = text
        self.metadata = metadata or {}

    def get_text(self) -> str:
        return self._text


class RagChunkWrapper:
    """
    Wraps a LightRAG chunk dict so it is fully compatible with
    safe_node_text() / safe_node_meta() from eval_runner_doc.

    Provides:
      .node.get_text()    – primary path used by safe_node_text()
      .node.metadata      – primary path used by safe_node_meta()
      .text               – fallback path in safe_node_text()
    """

    def __init__(self, chunk_dict: dict) -> None:
        text = chunk_dict.get("content") or chunk_dict.get("text") or ""
        meta = {k: v for k, v in chunk_dict.items() if k not in ("content", "text")}
        self.text = text
        self.node = _NodeLike(text, meta)


class _SimpleChunkNode:
    """Minimal node for docstore iteration (corpus_match_count)."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.metadata: Dict[str, Any] = {}

    def get_text(self) -> str:
        return self._text


class _Docstore:
    def __init__(self, chunk_texts: Dict[str, str]) -> None:
        self.docs = {k: _SimpleChunkNode(v) for k, v in chunk_texts.items()}


class _PreloadedRetriever:
    """Synchronous retriever that returns pre-fetched nodes (query arg ignored)."""

    def __init__(self, nodes: List[RagChunkWrapper]) -> None:
        self._nodes = nodes

    def retrieve(self, query: str) -> List[RagChunkWrapper]:
        return self._nodes


class RagAnythingIndexAdapter:
    """
    Makes RAG-Anything data look like a LlamaIndex VectorStoreIndex so that
    evaluate_one()'s internal diagnostic calls (corpus_match_count and
    get_retrieved_nodes) work without modification.

    All async retrieval is done *before* constructing this adapter (in the
    async evaluation loop), so evaluate_one() can call these methods
    synchronously without nested event-loop issues.

    Parameters
    ----------
    all_chunk_texts : dict
        {chunk_id: chunk_text} for the entire corpus.  Used by corpus_match_count.
    preloaded_diag_nodes : list
        Pre-fetched retrieval results for the current question at diag_k depth.
        Returned by as_retriever().retrieve() regardless of top_k argument.
    """

    def __init__(
        self,
        all_chunk_texts: Dict[str, str],
        preloaded_diag_nodes: List[RagChunkWrapper],
    ) -> None:
        self.docstore = _Docstore(all_chunk_texts)
        self._diag_nodes = preloaded_diag_nodes

    def as_retriever(self, similarity_top_k: int = 5) -> _PreloadedRetriever:
        return _PreloadedRetriever(self._diag_nodes)


# ===========================================================================
# Main evaluation loop (async)
# ===========================================================================

async def run_eval(args: argparse.Namespace) -> None:
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set up RAG-Anything instance
    rag = _build_raganything(cache_dir)

    # 2. Process PDF if not already cached
    await _process_pdf_if_needed(rag, args.pdf_path, cache_dir)

    # 3. Load all chunks for corpus_match_count diagnostics
    all_chunk_texts = _load_all_chunks(cache_dir)

    # 4. Load test questions
    tests = load_tests(args.tests)
    print(
        f"[raganything] Evaluating {len(tests)} questions  "
        f"k_eval={args.k_eval}  top_k_retrieve={args.top_k_retrieve}  diag_k={args.diag_k}"
    )

    # 5. Prepare output paths
    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = os.path.join(args.out_dir, f"results_{stamp}.json")
    out_csv  = os.path.join(args.out_dir, f"results_{stamp}.csv")
    out_gate = os.path.join(args.out_dir, f"gate_{stamp}.json")

    # 6. Per-question evaluation
    results: List[Dict[str, Any]] = []

    for t in tests:
        q = t.get("question", "")
        print(q)

        # Async: retrieve main pool (scored at k_eval)
        raw_main = await _retrieve_chunks(rag, q, top_k=args.top_k_retrieve)
        retrieved = [RagChunkWrapper(r) for r in raw_main]

        # Async: retrieve diagnostic pool (used inside evaluate_one for
        # hit_at_diag_k / diag_match_count analysis)
        raw_diag = await _retrieve_chunks(rag, q, top_k=args.diag_k)
        diag_nodes = [RagChunkWrapper(r) for r in raw_diag]

        # Build sync adapter for evaluate_one's internal diagnostic calls
        index_adapter = RagAnythingIndexAdapter(all_chunk_texts, diag_nodes)

        res = evaluate_one(
            index_adapter,
            t,
            retrieved,
            k_eval=args.k_eval,
            diag_k=args.diag_k,
            row_index=None,  # all content (incl. tables) lives in the unified RAG-Anything KB
        )
        results.append(res)

        print(
            f"[{res['id']}] cat={res['category']} "
            f"Hit@{args.k_eval}={res['hit_at_k']} "
            f"MRR@{args.k_eval}={res['mrr_at_k']} "
            f"P@{args.k_eval}={res['precision_at_k']} "
            f"Fail={res['failure_label']} "
            f"CorpusCnt={res['corpus_match_count']} "
            f"Hit@{args.diag_k}={res['hit_at_diag_k']} "
            f"len={res['chunk_len_chars']} "
            f"hdrDist={res['header_distance']}"
        )

    # 7. Write outputs
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    write_csv(results, out_csv)

    # 8. Category summaries + acceptance-criteria gate
    cat_summaries = summarize_by_category(results)
    print_category_summaries(cat_summaries)

    overall_pass, checks = apply_acceptance_criteria(cat_summaries, DEFAULT_ACCEPTANCE)
    print_gate_checks(overall_pass, checks)

    with open(out_gate, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "overall_pass": overall_pass,
                "checks": checks,
                "category_summaries": cat_summaries,
                "acceptance": DEFAULT_ACCEPTANCE,
            },
            fh,
            indent=2,
        )

    print(f"\nWrote:\n  {out_json}\n  {out_csv}\n  {out_gate}")

    if args.fail_on_gate and not overall_pass:
        raise SystemExit(1)


# ===========================================================================
# CLI entry-point
# ===========================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Retrieval-only doc evaluation using RAG-Anything (MinerU). "
            "Reuses evaluate_one() scoring from eval_runner_doc.py."
        )
    )
    ap.add_argument(
        "--tests",
        default=str(PROJECT_ROOT / "evaluation" / "test_questions_doc.json"),
        help="Path to test questions JSON (default: evaluation/test_questions_doc.json)",
    )
    ap.add_argument(
        "--pdf_path",
        default=_DEFAULT_PDF,
        help="Path to the PDF corpus (default: Coral Gables comprehensive plan)",
    )
    ap.add_argument(
        "--cache_dir",
        default=_DEFAULT_CACHE,
        help=(
            "Directory for RAG-Anything / MinerU processed output "
            "(default: storage/raganything/mineru_cache)"
        ),
    )
    ap.add_argument(
        "--top_k_retrieve",
        type=int,
        default=20,
        help="Retrieval pool size passed to the vector DB (default: 20)",
    )
    ap.add_argument(
        "--k_eval",
        type=int,
        default=5,
        help="Top-k cutoff used for hit/MRR/precision scoring (default: 5)",
    )
    ap.add_argument(
        "--diag_k",
        type=int,
        default=200,
        help="Diagnostic retrieval depth for lookup-category failure analysis (default: 200)",
    )
    ap.add_argument(
        "--out_dir",
        default="eval_outputs_raganything",
        help="Output directory for results JSON / CSV / gate JSON (default: eval_outputs_raganything)",
    )
    ap.add_argument(
        "--fail_on_gate",
        action="store_true",
        help="Exit with status 1 if acceptance-criteria gate fails",
    )

    args = ap.parse_args()
    asyncio.run(run_eval(args))


if __name__ == "__main__":
    main()
