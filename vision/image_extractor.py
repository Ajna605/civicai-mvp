"""Extract images embedded in PDF documents using PyMuPDF."""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class ImageRecord:
    """Metadata and base64-encoded bytes for a single image extracted from a PDF."""

    doc_id: str
    source_path: str
    page: int
    image_index: int
    width: int
    height: int
    image_b64: str
    ext: str = "png"
    colorspace: Optional[str] = None


def extract_images_from_pdf(
    pdf_path: str,
    min_width: int = 50,
    min_height: int = 50,
) -> List[ImageRecord]:
    """Extract all embedded images from a PDF file.

    Each unique image (by xref) is extracted once, even if it appears on
    multiple pages.  Images smaller than *min_width* × *min_height* pixels
    are skipped to avoid tiny decorative elements.

    Args:
        pdf_path: Absolute or relative path to the PDF file.
        min_width: Minimum image width in pixels to include.
        min_height: Minimum image height in pixels to include.

    Returns:
        Ordered list of :class:`ImageRecord` objects, one per extracted image.

    Raises:
        RuntimeError: If PyMuPDF (``fitz``) is not installed.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        ) from exc

    path = Path(pdf_path)
    doc_id = path.stem
    records: List[ImageRecord] = []
    seen_xrefs: set[int] = set()

    doc = fitz.open(str(path))
    try:
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                # Skip duplicates (same image referenced from multiple pages)
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    base_image = doc.extract_image(xref)
                except Exception:
                    continue

                w: int = base_image.get("width", 0)
                h: int = base_image.get("height", 0)
                if w < min_width or h < min_height:
                    continue

                image_b64 = base64.b64encode(base_image["image"]).decode("utf-8")
                ext: str = base_image.get("ext") or "png"
                colorspace = base_image.get("colorspace")

                records.append(
                    ImageRecord(
                        doc_id=doc_id,
                        source_path=str(path),
                        page=page_num + 1,
                        image_index=img_idx,
                        width=w,
                        height=h,
                        image_b64=image_b64,
                        ext=ext,
                        colorspace=str(colorspace) if colorspace is not None else None,
                    )
                )
    finally:
        doc.close()

    return records
