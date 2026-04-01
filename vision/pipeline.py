"""Full vision pipeline: extract images from documents and interpret them.

Supports PDF and DOCX source files.

Typical usage
-------------
Run as a standalone script::

    python -m vision.pipeline --source data/raw/my_document.pdf \\
        --out data/normalized/pdf/image_descriptions.jsonl

    python -m vision.pipeline --source data/raw/my_report.docx \\
        --out data/normalized/docx/image_descriptions.jsonl

Or call programmatically::

    from vision.pipeline import run_vision_pipeline_to_jsonl

    n = run_vision_pipeline_to_jsonl(
        "data/raw/my_document.pdf",
        "data/normalized/pdf/image_descriptions.jsonl",
    )
    print(f"Wrote {n} image description chunks.")
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

from .image_extractor import extract_images_from_document
from .image_interpreter import ImageInterpreter
from utils.hash_utils import short_hash


def run_vision_pipeline(
    doc_path: str,
    *,
    api_key: Optional[str] = None,
    model: str = ImageInterpreter.DEFAULT_MODEL,
    max_tokens: int = 512,
    min_width: int = 50,
    min_height: int = 50,
) -> List[dict]:
    """Extract and interpret all images in a document (PDF or DOCX).

    Each image is described by a vision LLM and returned as a RAG-ready chunk
    dict whose schema matches ``rag_chunks.jsonl`` records produced by
    ``build_corpus.py``.

    Args:
        doc_path: Path to a ``.pdf`` or ``.docx`` file.
        api_key: OpenAI API key (defaults to ``OPENAI_API_KEY`` env variable).
        model: Vision-capable OpenAI model to use.
        max_tokens: Maximum response tokens per image description.
        min_width: Skip images narrower than this many pixels.
        min_height: Skip images shorter than this many pixels.

    Returns:
        List of chunk dicts ready to be appended to ``rag_chunks.jsonl``.
    """
    records = extract_images_from_document(doc_path, min_width=min_width, min_height=min_height)
    if not records:
        return []

    source_fmt = Path(doc_path).suffix.lstrip(".").lower()
    interpreter = ImageInterpreter(api_key=api_key, model=model, max_tokens=max_tokens)

    chunks: List[dict] = []
    for rec in records:
        page_label = f"page {rec.page}" if rec.page > 0 else "unknown page"
        context = (
            f"{page_label.capitalize()}, image {rec.image_index + 1} "
            f"from document '{rec.doc_id}'"
        )
        description = interpreter.interpret(rec.image_b64, ext=rec.ext, context=context)
        if not description.strip():
            continue

        text = description.strip()
        chunk_id = (
            f"{rec.doc_id}__img_p{rec.page}_i{rec.image_index}__{short_hash(text)}"
        )

        chunks.append(
            {
                "id": chunk_id,
                "doc_id": rec.doc_id,
                "source_path": rec.source_path,
                "section_path": [f"Page {rec.page}"] if rec.page > 0 else [],
                "section_index": rec.page,
                "block_type": "image",
                "block_index": rec.image_index,
                "text": text,
                "extra": {
                    "source": source_fmt,
                    "page": rec.page,
                    "image_width": rec.width,
                    "image_height": rec.height,
                },
            }
        )

    return chunks


def run_vision_pipeline_to_jsonl(
    doc_path: str,
    out_path: str,
    *,
    api_key: Optional[str] = None,
    model: str = ImageInterpreter.DEFAULT_MODEL,
    max_tokens: int = 512,
    min_width: int = 50,
    min_height: int = 50,
    append: bool = False,
) -> int:
    """Run the vision pipeline and write results to a JSONL file.

    Args:
        doc_path: Path to a ``.pdf`` or ``.docx`` file.
        out_path: Destination ``.jsonl`` file path (parent dirs are created).
        api_key: OpenAI API key (defaults to ``OPENAI_API_KEY`` env variable).
        model: Vision-capable OpenAI model to use.
        max_tokens: Maximum response tokens per image description.
        min_width: Skip images narrower than this many pixels.
        min_height: Skip images shorter than this many pixels.
        append: When ``True``, append to an existing file instead of overwriting.

    Returns:
        Number of image description chunks written.
    """
    chunks = run_vision_pipeline(
        doc_path,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        min_width=min_width,
        min_height=min_height,
    )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"
    with open(out, mode, encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return len(chunks)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extract and interpret images from a document (PDF or DOCX) "
            "using the OpenAI Vision API."
        )
    )
    p.add_argument("--source", required=True, help="Path to the input PDF or DOCX file.")
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSONL path for image description chunks. "
            "Defaults to data/normalized/<format>/image_descriptions.jsonl."
        ),
    )
    p.add_argument(
        "--model",
        default=ImageInterpreter.DEFAULT_MODEL,
        help="OpenAI vision model to use (default: %(default)s).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per image description (default: %(default)s).",
    )
    p.add_argument(
        "--min-width",
        type=int,
        default=50,
        help="Minimum image width in pixels (default: %(default)s).",
    )
    p.add_argument(
        "--min-height",
        type=int,
        default=50,
        help="Minimum image height in pixels (default: %(default)s).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source_fmt = Path(args.source).suffix.lstrip(".").lower()
    out_path = args.out or str(
        Path(f"data/normalized/{source_fmt}/image_descriptions.jsonl")
    )

    api_key = os.environ.get("OPENAI_API_KEY")

    print(f"[vision.pipeline] Source : {args.source}")
    print(f"[vision.pipeline] Output : {out_path}")
    print(f"[vision.pipeline] Model  : {args.model}")

    n = run_vision_pipeline_to_jsonl(
        args.source,
        out_path,
        api_key=api_key,
        model=args.model,
        max_tokens=args.max_tokens,
        min_width=args.min_width,
        min_height=args.min_height,
    )

    print(f"[vision.pipeline] Wrote {n} image description chunk(s) → {out_path}")


if __name__ == "__main__":
    main()
