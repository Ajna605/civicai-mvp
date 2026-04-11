# RAG-Anything (MinerU) Retrieval Evaluation

Compares vanilla LlamaIndex doc-retrieval quality against
[HKUDS/RAG-Anything](https://github.com/HKUDS/RAG-Anything) (MinerU parser)
on `evaluation/test_questions_doc.json`.

The runner is **retrieval-only** – no LLM is called.  
It reuses the same `evaluate_one()` scoring logic as `evaluation/eval_runner_doc.py`.

---

## 1 — Install Dependencies

```bash
# Activate your existing project virtual-env first
source .venv/bin/activate   # or conda activate <env>

# Core RAG-Anything library
pip install raganything

# MinerU PDF parser (GPU recommended for large documents)
pip install mineru[gpu]    # CUDA GPU
# pip install mineru        # CPU-only fallback

# sentence-transformers is already in requirements.txt,
# but ensure it is present:
pip install sentence-transformers
```

> **Version note**: RAG-Anything depends on LightRAG ≥ 1.0.
> If you hit a `lightrag` import error run:
> ```bash
> pip install "lightrag-hku>=1.0"
> ```

---

## 2 — Run the Evaluation

```bash
# From the repo root:
python evaluation/eval_runner_doc_raganything.py \
    --tests evaluation/test_questions_doc.json \
    --pdf_path "data/raw/Coral Gables - Comprehensive Plan - 12-25-2025 20-55 - selectable.pdf" \
    --k_eval 5 \
    --top_k_retrieve 20 \
    --out_dir eval_outputs_raganything
```

### First run (slow)
MinerU parses the PDF and LightRAG builds the knowledge base.  
Results are cached under `storage/raganything/mineru_cache/`.  
Subsequent runs skip this step.

### Subsequent runs (fast)
The runner detects the cache marker (`storage/raganything/mineru_cache/.rag_processed`)
and proceeds directly to retrieval.

---

## 3 — CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--tests` | `evaluation/test_questions_doc.json` | Question set |
| `--pdf_path` | Coral Gables PDF in `data/raw/` | PDF corpus |
| `--cache_dir` | `storage/raganything/mineru_cache/` | MinerU + LightRAG KV store |
| `--top_k_retrieve` | `20` | Retrieval pool size |
| `--k_eval` | `5` | Top-k for hit@k / MRR / precision scoring |
| `--diag_k` | `200` | Diagnostic retrieval depth (failure analysis) |
| `--out_dir` | `eval_outputs_raganything` | Where to write JSON / CSV / gate files |
| `--fail_on_gate` | off | Exit 1 if acceptance-criteria gate fails |

---

## 4 — Output Files

Each run writes three files to `--out_dir`:

| File | Contents |
|------|----------|
| `results_<stamp>.json` | Per-question scores + diagnostics |
| `results_<stamp>.csv`  | Same data in tabular form |
| `gate_<stamp>.json`    | Category summaries + acceptance-criteria pass/fail |

---

## 5 — Comparing Against the Vanilla Runner

Both runners write to separate `--out_dir` directories so results never
overwrite each other.

```bash
# Vanilla doc eval (already done – do NOT rerun)
# python evaluation/eval_runner_doc.py --format pdf --k_eval 5 --out_dir eval_outputs

# RAG-Anything eval
python evaluation/eval_runner_doc_raganything.py \
    --k_eval 5 --out_dir eval_outputs_raganything
```

Compare `gate_*.json` in both output directories for a side-by-side view of
`avg_hit_at_k`, `avg_mrr_at_k`, and failure-label rates per category.

---

## 6 — Design Notes

* **Same embedding model** – both runners use
  `sentence-transformers/all-MiniLM-L6-v2` (dim=384).  
  The RAG-Anything run reuses the llama_index `Settings.embed_model` instance
  loaded by `rag/llm_settings.py`, ensuring identical vector spaces.

* **Plain semantic retrieval** – only LightRAG's internal `chunks_vdb` vector
  similarity search (`mode="naive"`) is used.  No policy-lookup code path,
  no table-aware reranking, no graph traversal.

* **Table content included** – table extraction is enabled by default so that
  MinerU extracts table rows and includes them in the knowledge base alongside
  narrative text.  No separate `pdf_tables` index is used.

* **Deterministic** – `do_sample=False` / `temperature=0` semantics apply only
  to the LLM (which is never called here).  Vector similarity search is
  deterministic given the same embeddings and FAISS/NanoVDB index.

---

## 7 — raganything==1.2.10 Compatibility

The runner auto-detects the installed raganything version using
`inspect.signature` and `hasattr` checks so it works across versions.
Below is a summary of the differences from earlier releases and how they are
handled.

### 7.1 `RAGAnythingConfig.__init__` – no `llm_model_func`

In raganything ≥ 1.2.10, `RAGAnythingConfig.__init__()` no longer accepts
`llm_model_func` or `vision_model_func` as keyword arguments.  The runner
inspects the constructor signature and only passes these when they are listed
in the parameter set.

### 7.2 MinerU parser flag mapping

The underlying `MineruParser._run_mineru_command` in raganything 1.2.10
accepts **MinerU-native** flags rather than the `enable_*` wrappers used in
earlier releases:

| Old flag (pre-1.2.10) | Old runner value | New flag (1.2.10 MinerU) | New runner value | Notes |
|------------------------|-----------------|--------------------------|-----------------|-------|
| `enable_table_processing` | `True` | `table` | `True` | Table extraction enabled by default |
| `enable_equation_processing` | `False` | `formula` | `True` | MinerU's `formula` defaults to `True`; runner now enables it |
| `enable_image_processing` | `False` | *(none)* | — | MinerU does not expose an image toggle; omit to avoid `TypeError` |

The runner inspects both `process_document_complete` and
`MineruParser._run_mineru_command` signatures to decide which names to use.

### 7.3 `finalize_storages` removed from config

`RAGAnythingConfig` in 1.2.10 does **not** have a `finalize_storages` method.
Internal code paths may still attempt to call it during storage tear-down.
The runner patches a no-op async function onto the config object when the
attribute is missing, preventing `AttributeError` at shutdown.

### 7.4 CUDA / GPU troubleshooting on FIU cluster

If your NVIDIA driver is too old for the installed PyTorch CUDA build you will
see:

```
CUDA initialization: The NVIDIA driver on your system is too old …
```

Workarounds:

1. **CPU-only** – force PyTorch to skip CUDA entirely:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   python evaluation/eval_runner_doc_raganything.py ...
   ```

2. **Install a matching PyTorch** – pick the wheel built for your driver's
   CUDA version (check with `nvidia-smi`):
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu124
   ```
