"""Extract images embedded in PDF and DOCX documents."""
from __future__ import annotations

import base64
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from docx import Document as DocxDocument
    _DOCX_AVAILABLE = True
except ImportError:
    DocxDocument = None  # type: ignore[assignment,misc]
    _DOCX_AVAILABLE = False


# MIME types that can be passed directly to the OpenAI Vision API as base64 data URIs
_RASTER_MIME_TO_EXT = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpeg",
    "image/gif": "gif",
    "image/bmp": "bmp",
    "image/tiff": "tiff",
    "image/webp": "webp",
}


@dataclass
class ImageRecord:
    """Metadata and base64-encoded bytes for a single image extracted from a document."""

    doc_id: str
    source_path: str
    page: int          # 1-based page number; 0 when the format has no page concept
    image_index: int
    width: int         # pixels; 0 when undetermined
    height: int        # pixels; 0 when undetermined
    image_b64: str
    ext: str = "png"
    colorspace: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_image_dimensions(data: bytes, content_type: str) -> Tuple[int, int]:
    """Return ``(width, height)`` in pixels, or ``(0, 0)`` if parsing fails."""
    try:
        if content_type == "image/png":
            if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n":
                w = struct.unpack(">I", data[16:20])[0]
                h = struct.unpack(">I", data[20:24])[0]
                return w, h
        elif content_type in ("image/jpeg", "image/jpg"):
            i = 2  # skip SOI marker (FF D8)
            while i + 4 <= len(data):
                if data[i] != 0xFF:
                    break
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2):  # SOF0 / SOF1 / SOF2
                    h = struct.unpack(">H", data[i + 5 : i + 7])[0]
                    w = struct.unpack(">H", data[i + 7 : i + 9])[0]
                    return w, h
                if marker in (0xD8, 0xD9):  # SOI / EOI (no length field)
                    i += 2
                else:
                    length = struct.unpack(">H", data[i + 2 : i + 4])[0]
                    i += 2 + length
    except Exception:
        pass
    return 0, 0


# ---------------------------------------------------------------------------
# PDF extractor
# ---------------------------------------------------------------------------

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
        pdf_path: Path to the PDF file.
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


# ---------------------------------------------------------------------------
# DOCX extractor
# ---------------------------------------------------------------------------

def extract_images_from_docx(
    docx_path: str,
    min_width: int = 50,
    min_height: int = 50,
) -> List[ImageRecord]:
    """Extract all embedded raster images from a DOCX file.

    Images are sourced from the document's OPC relationship parts, so both
    inline and floating images are captured.  Vector formats (WMF, EMF) are
    skipped because they cannot be sent directly to the Vision API.

    DOCX does not expose page numbers natively, so :attr:`ImageRecord.page`
    is set to ``0`` for all records.  The size filter is applied only when
    pixel dimensions can be parsed from the image bytes; images whose
    dimensions cannot be determined are always included.

    Args:
        docx_path: Path to the DOCX file.
        min_width: Minimum image width in pixels to include.
        min_height: Minimum image height in pixels to include.

    Returns:
        Ordered list of :class:`ImageRecord` objects, one per extracted image.

    Raises:
        RuntimeError: If ``python-docx`` is not installed.
    """
    if not _DOCX_AVAILABLE:
        raise RuntimeError(
            "python-docx is not installed. Run: pip install python-docx"
        )

    path = Path(docx_path)
    doc_id = path.stem
    doc = DocxDocument(str(path))

    records: List[ImageRecord] = []
    img_index = 0

    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue

        content_type: str = rel.target_part.content_type
        ext = _RASTER_MIME_TO_EXT.get(content_type)
        if ext is None:
            # Skip vector formats (WMF, EMF) and unknown types
            continue

        blob: bytes = rel.target_part.blob
        w, h = _parse_image_dimensions(blob, content_type)

        # Only apply the size filter when we can actually determine dimensions
        if w > 0 and w < min_width:
            continue
        if h > 0 and h < min_height:
            continue

        image_b64 = base64.b64encode(blob).decode("utf-8")
        records.append(
            ImageRecord(
                doc_id=doc_id,
                source_path=str(path),
                page=0,  # DOCX has no native page numbers
                image_index=img_index,
                width=w,
                height=h,
                image_b64=image_b64,
                ext=ext,
            )
        )
        img_index += 1

    return records


# ---------------------------------------------------------------------------
# Format dispatcher
# ---------------------------------------------------------------------------

def extract_images_from_document(
    doc_path: str,
    min_width: int = 50,
    min_height: int = 50,
) -> List[ImageRecord]:
    """Extract images from a document, dispatching by file extension.

    Supports ``.pdf`` (via PyMuPDF) and ``.docx`` (via python-docx).

    Args:
        doc_path: Path to a ``.pdf`` or ``.docx`` file.
        min_width: Minimum image width in pixels to include.
        min_height: Minimum image height in pixels to include.

    Returns:
        Ordered list of :class:`ImageRecord` objects.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = Path(doc_path).suffix.lower()
    if suffix == ".pdf":
        return extract_images_from_pdf(doc_path, min_width=min_width, min_height=min_height)
    if suffix == ".docx":
        return extract_images_from_docx(doc_path, min_width=min_width, min_height=min_height)
    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        "extract_images_from_document supports .pdf and .docx files."
    )
