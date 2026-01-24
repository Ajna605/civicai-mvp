from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict

from ingestion.loaders.pdf_loader import normalize_pdf


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_json(path: str, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Ingest a document into NormalizedDoc format")
    ap.add_argument("--input", required=True, help="Path to input PDF")
    ap.add_argument("--doc_id", default=None, help="Optional document id override")
    ap.add_argument("--out_normalized_dir", default="data/normalized")
    ap.add_argument("--out_manifest_dir", default="data/manifest")
    args = ap.parse_args()

    ensure_dir(args.out_normalized_dir)
    ensure_dir(args.out_manifest_dir)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    ndoc = normalize_pdf(args.input, doc_id=args.doc_id)

    out_norm = os.path.join(
        args.out_normalized_dir, f"{ndoc.doc_id}.normalized.json"
    )
    write_json(out_norm, ndoc.to_dict())

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "ingested_at": datetime.now().isoformat(),
        "inputs": [
            {
                "path": args.input,
                "doc_id": ndoc.doc_id,
                "file_name": ndoc.source.file_name,
                "file_type": ndoc.source.file_type,
                "sha256": ndoc.source.sha256,
                "output_normalized": out_norm,
                "stats": {
                    "sections": len(ndoc.sections),
                    "tables": sum(len(s.tables) for s in ndoc.sections),
                    "page_range": {
                        "min": min(s.page_start for s in ndoc.sections),
                        "max": max(s.page_end for s in ndoc.sections),
                    },
                },
                "warnings": [],
            }
        ],
        "notes": "Hybrid PDF normalization with heading hierarchy; tables not extracted yet.",
    }

    out_manifest = os.path.join(
        args.out_manifest_dir, f"manifest_{run_id}.json"
    )
    write_json(out_manifest, manifest)

    print("Wrote:")
    print(" ", out_norm)
    print(" ", out_manifest)


if __name__ == "__main__":
    main()
