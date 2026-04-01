"""Unit tests for the vision module.

Tests cover:
1. Image extraction from a minimal synthetic PDF.
2. Image extraction from DOCX files.
3. extract_images_from_document dispatcher (routing + unsupported format error).
4. ImageInterpreter raises ValueError when no API key is provided.
5. ImageInterpreter.interpret calls the OpenAI client with the correct payload.
6. run_vision_pipeline returns structured chunks with the expected schema.
7. run_vision_pipeline_to_jsonl writes valid JSONL and returns the correct count.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the repo root is importable when running tests from any directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from vision.image_extractor import (
    ImageRecord,
    extract_images_from_pdf,
    extract_images_from_docx,
    extract_images_from_document,
    _parse_image_dimensions,
)
from vision.image_interpreter import ImageInterpreter
from vision.pipeline import run_vision_pipeline, run_vision_pipeline_to_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_image_record(page: int = 1, idx: int = 0) -> ImageRecord:
    """Return a minimal ImageRecord with a tiny 1×1 white PNG."""
    # 1×1 transparent PNG (68 bytes, valid base64)
    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    return ImageRecord(
        doc_id="test_doc",
        source_path="/tmp/test_doc.pdf",
        page=page,
        image_index=idx,
        width=100,
        height=100,
        image_b64=png_b64,
        ext="png",
        colorspace="rgb",
    )


def _mock_openai_response(content: str) -> MagicMock:
    """Build a minimal mock that looks like an OpenAI chat completion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# ImageRecord dataclass
# ---------------------------------------------------------------------------

class TestImageRecord:
    def test_fields_populated(self) -> None:
        rec = _make_image_record()
        assert rec.doc_id == "test_doc"
        assert rec.page == 1
        assert rec.image_index == 0
        assert rec.ext == "png"
        assert isinstance(rec.image_b64, str)

    def test_optional_colorspace_none(self) -> None:
        rec = ImageRecord(
            doc_id="doc",
            source_path="/tmp/doc.pdf",
            page=1,
            image_index=0,
            width=100,
            height=100,
            image_b64="abc",
        )
        assert rec.colorspace is None


# ---------------------------------------------------------------------------
# extract_images_from_pdf
# ---------------------------------------------------------------------------

class TestExtractImagesFromPdf:
    def test_returns_list(self, tmp_path: Path) -> None:
        """extract_images_from_pdf should return a list (possibly empty)."""
        # Create a trivial empty PDF via PyMuPDF
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        pdf_path = tmp_path / "empty.pdf"
        doc = fitz.open()
        doc.new_page()
        doc.save(str(pdf_path))
        doc.close()

        records = extract_images_from_pdf(str(pdf_path))
        assert isinstance(records, list)

    def test_min_size_filter(self, tmp_path: Path) -> None:
        """Images below min_width / min_height should be excluded."""
        try:
            import fitz
        except ImportError:
            pytest.skip("PyMuPDF not installed")

        # Build a PDF with a 10×10 PNG embedded
        tiny_png = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAD"
            "UlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        )

        pdf_path = tmp_path / "tiny_img.pdf"
        doc = fitz.open()
        page = doc.new_page(width=200, height=200)
        rect = fitz.Rect(10, 10, 20, 20)
        page.insert_image(rect, stream=tiny_png)
        doc.save(str(pdf_path))
        doc.close()

        # With default min_width=50 the 10px image should be skipped
        records = extract_images_from_pdf(str(pdf_path), min_width=50, min_height=50)
        assert all(r.width >= 50 and r.height >= 50 for r in records)

    def test_raises_on_missing_pymupdf(self) -> None:
        with patch.dict("sys.modules", {"fitz": None}):
            with pytest.raises(RuntimeError, match="PyMuPDF"):
                extract_images_from_pdf("nonexistent.pdf")


# ---------------------------------------------------------------------------
# _parse_image_dimensions
# ---------------------------------------------------------------------------

class TestParseImageDimensions:
    def _make_png(self, width: int, height: int) -> bytes:
        """Build a minimal valid PNG header with the given dimensions."""
        import struct
        import zlib

        signature = b"\x89PNG\r\n\x1a\n"
        ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
        ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
        # Minimal IDAT (empty compressed data) + IEND
        idat_data = zlib.compress(b"\x00" * (width * 3 + 1) * height)
        idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
        idat_chunk = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
        iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
        iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
        return signature + ihdr_chunk + idat_chunk + iend_chunk

    def test_png_dimensions_parsed(self) -> None:
        png = self._make_png(200, 150)
        w, h = _parse_image_dimensions(png, "image/png")
        assert w == 200
        assert h == 150

    def test_unknown_format_returns_zero(self) -> None:
        w, h = _parse_image_dimensions(b"random data", "image/tiff")
        assert w == 0
        assert h == 0

    def test_truncated_data_returns_zero(self) -> None:
        w, h = _parse_image_dimensions(b"\x89PNG\r\n\x1a\n", "image/png")
        assert w == 0
        assert h == 0


# ---------------------------------------------------------------------------
# extract_images_from_docx
# ---------------------------------------------------------------------------

class TestExtractImagesFromDocx:
    def _make_mock_docx_part(self, blob: bytes, content_type: str) -> MagicMock:
        """Build a minimal mock for an OPC relationship target part."""
        part = MagicMock()
        part.blob = blob
        part.content_type = content_type
        return part

    def _make_mock_rel(self, blob: bytes, content_type: str, reltype_fragment: str = "image") -> MagicMock:
        rel = MagicMock()
        rel.reltype = f"http://schemas.openxmlformats.org/officeDocument/2006/relationships/{reltype_fragment}"
        rel.target_part = self._make_mock_docx_part(blob, content_type)
        return rel

    def _make_png_bytes(self, width: int = 100, height: int = 80) -> bytes:
        import struct
        import zlib
        sig = b"\x89PNG\r\n\x1a\n"
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        ihdr_crc = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
        chunk = struct.pack(">I", 13) + b"IHDR" + ihdr + ihdr_crc
        return sig + chunk + b"\x00" * 20  # truncated but header is present

    def test_raises_on_missing_python_docx(self) -> None:
        with patch("vision.image_extractor._DOCX_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="python-docx"):
                extract_images_from_docx("nonexistent.docx")

    def test_returns_list(self, tmp_path: Path) -> None:
        """Patching python-docx internals: should return a list."""
        png_bytes = self._make_png_bytes()
        mock_rel = self._make_mock_rel(png_bytes, "image/png")

        mock_doc = MagicMock()
        mock_doc.part.rels = {"rId1": mock_rel}

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx")

        assert isinstance(records, list)
        assert len(records) == 1

    def test_skips_non_image_relationships(self, tmp_path: Path) -> None:
        png_bytes = self._make_png_bytes()
        img_rel = self._make_mock_rel(png_bytes, "image/png", reltype_fragment="image")
        other_rel = self._make_mock_rel(b"", "application/xml", reltype_fragment="styles")

        mock_doc = MagicMock()
        mock_doc.part.rels = {"rId1": img_rel, "rId2": other_rel}

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx")

        assert len(records) == 1

    def test_skips_vector_formats(self, tmp_path: Path) -> None:
        wmf_rel = self._make_mock_rel(b"\xd7\xcd\xc6\x9a", "image/x-wmf")
        png_rel = self._make_mock_rel(self._make_png_bytes(), "image/png")

        mock_doc = MagicMock()
        mock_doc.part.rels = {"rId1": wmf_rel, "rId2": png_rel}

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx")

        assert len(records) == 1
        assert records[0].ext == "png"

    def test_page_set_to_zero(self) -> None:
        mock_doc = MagicMock()
        mock_doc.part.rels = {"rId1": self._make_mock_rel(self._make_png_bytes(), "image/png")}

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx")

        assert records[0].page == 0

    def test_min_size_filter_applied_when_dimensions_known(self) -> None:
        small_png = self._make_png_bytes(width=10, height=10)
        large_png = self._make_png_bytes(width=200, height=200)

        mock_doc = MagicMock()
        mock_doc.part.rels = {
            "rId1": self._make_mock_rel(small_png, "image/png"),
            "rId2": self._make_mock_rel(large_png, "image/png"),
        }

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx", min_width=50, min_height=50)

        assert len(records) == 1
        assert records[0].width == 200

    def test_unknown_dimensions_bypass_filter(self) -> None:
        """If we can't parse dimensions (e.g. TIFF), the image is included."""
        # Use TIFF content type — no dimension parser for it, so w=h=0
        mock_doc = MagicMock()
        mock_doc.part.rels = {
            "rId1": self._make_mock_rel(b"\x49\x49\x2a\x00" + b"\x00" * 100, "image/tiff")
        }

        with patch("vision.image_extractor._DOCX_AVAILABLE", True), \
             patch("vision.image_extractor.DocxDocument", return_value=mock_doc):
            records = extract_images_from_docx("/fake/doc.docx", min_width=50, min_height=50)

        assert len(records) == 1
        assert records[0].width == 0


# ---------------------------------------------------------------------------
# extract_images_from_document dispatcher
# ---------------------------------------------------------------------------

class TestExtractImagesFromDocument:
    def test_routes_pdf(self) -> None:
        with patch("vision.image_extractor.extract_images_from_pdf", return_value=[]) as mock_pdf:
            extract_images_from_document("/fake/doc.pdf")
        mock_pdf.assert_called_once()

    def test_routes_docx(self) -> None:
        with patch("vision.image_extractor.extract_images_from_docx", return_value=[]) as mock_docx:
            extract_images_from_document("/fake/doc.docx")
        mock_docx.assert_called_once()

    def test_raises_on_unsupported_extension(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_images_from_document("/fake/doc.csv")

    def test_passes_size_filter_args(self) -> None:
        with patch("vision.image_extractor.extract_images_from_pdf", return_value=[]) as mock_pdf:
            extract_images_from_document("/fake/doc.pdf", min_width=100, min_height=80)
        _, kwargs = mock_pdf.call_args
        assert kwargs["min_width"] == 100
        assert kwargs["min_height"] == 80

class TestImageInterpreter:
    def test_raises_without_api_key(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "OPENAI_API_KEY"}
        mock_client_cls = MagicMock()
        with patch.dict(os.environ, env, clear=True), \
             patch("vision.image_interpreter._OPENAI_AVAILABLE", True), \
             patch("vision.image_interpreter.OpenAI", mock_client_cls):
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                ImageInterpreter(api_key=None)

    def test_raises_on_missing_openai(self) -> None:
        with patch("vision.image_interpreter._OPENAI_AVAILABLE", False):
            with pytest.raises(RuntimeError, match="openai"):
                ImageInterpreter(api_key="test-key")

    def test_interpret_sends_correct_payload(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            "A map of the city."
        )

        with patch("vision.image_interpreter._OPENAI_AVAILABLE", True), \
             patch("vision.image_interpreter.OpenAI", return_value=mock_client):
            interpreter = ImageInterpreter(api_key="test-key", model="gpt-4o")
            result = interpreter.interpret("abc123", ext="png", context="Page 1")

        assert result == "A map of the city."
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert len(messages) == 1
        content = messages[0]["content"]
        # Should have a text part and an image_url part
        types = [c["type"] for c in content]
        assert "text" in types
        assert "image_url" in types

    def test_interpret_includes_context_in_prompt(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("ok")

        with patch("vision.image_interpreter._OPENAI_AVAILABLE", True), \
             patch("vision.image_interpreter.OpenAI", return_value=mock_client):
            interpreter = ImageInterpreter(api_key="test-key")
            interpreter.interpret("b64data", context="Fiscal chart, page 5")

        text_part = next(
            c
            for c in mock_client.chat.completions.create.call_args[1]["messages"][0][
                "content"
            ]
            if c["type"] == "text"
        )
        assert "Fiscal chart, page 5" in text_part["text"]

    def test_interpret_data_uri_format(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response("ok")

        with patch("vision.image_interpreter._OPENAI_AVAILABLE", True), \
             patch("vision.image_interpreter.OpenAI", return_value=mock_client):
            interpreter = ImageInterpreter(api_key="test-key")
            interpreter.interpret("MYBASE64DATA", ext="jpeg")

        img_part = next(
            c
            for c in mock_client.chat.completions.create.call_args[1]["messages"][0][
                "content"
            ]
            if c["type"] == "image_url"
        )
        assert img_part["image_url"]["url"] == "data:image/jpeg;base64,MYBASE64DATA"

    def test_interpret_returns_empty_string_on_none_content(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _mock_openai_response(None)

        with patch("vision.image_interpreter._OPENAI_AVAILABLE", True), \
             patch("vision.image_interpreter.OpenAI", return_value=mock_client):
            interpreter = ImageInterpreter(api_key="test-key")
            result = interpreter.interpret("b64")

        assert result == ""


# ---------------------------------------------------------------------------
# run_vision_pipeline
# ---------------------------------------------------------------------------

class TestRunVisionPipeline:
    def _patch_pipeline(self, descriptions: list[str]):
        """Context manager that patches both extractor and interpreter."""
        records = [_make_image_record(page=i + 1, idx=0) for i in range(len(descriptions))]

        mock_interpreter = MagicMock(spec=ImageInterpreter)
        mock_interpreter.interpret.side_effect = descriptions

        extractor_patch = patch(
            "vision.pipeline.extract_images_from_document", return_value=records
        )
        interp_patch = patch(
            "vision.pipeline.ImageInterpreter", return_value=mock_interpreter
        )
        return extractor_patch, interp_patch

    def test_returns_list_of_chunks(self) -> None:
        ep, ip = self._patch_pipeline(["A civic map.", "A bar chart."])
        with ep, ip:
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")

        assert len(chunks) == 2

    def test_chunk_schema(self) -> None:
        ep, ip = self._patch_pipeline(["Description of image one."])
        with ep, ip:
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")

        chunk = chunks[0]
        for field in ("id", "doc_id", "source_path", "text", "block_type", "extra"):
            assert field in chunk, f"Missing field: {field}"
        assert chunk["block_type"] == "image"
        assert chunk["text"] == "Description of image one."

    def test_empty_description_is_skipped(self) -> None:
        ep, ip = self._patch_pipeline(["", "Valid description."])
        with ep, ip:
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")

        assert len(chunks) == 1
        assert chunks[0]["text"] == "Valid description."

    def test_returns_empty_list_when_no_images(self) -> None:
        with patch("vision.pipeline.extract_images_from_document", return_value=[]):
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")
        assert chunks == []

    def test_chunk_id_is_unique_per_image(self) -> None:
        ep, ip = self._patch_pipeline(["Desc A", "Desc B"])
        with ep, ip:
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")

        ids = [c["id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_extra_contains_page_and_dimensions(self) -> None:
        ep, ip = self._patch_pipeline(["Some image."])
        with ep, ip:
            chunks = run_vision_pipeline("/fake/doc.pdf", api_key="test-key")

        extra = chunks[0]["extra"]
        assert "page" in extra
        assert "image_width" in extra
        assert "image_height" in extra


# ---------------------------------------------------------------------------
# run_vision_pipeline_to_jsonl
# ---------------------------------------------------------------------------

class TestRunVisionPipelineToJsonl:
    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        out_file = tmp_path / "out.jsonl"
        chunks = [
            {
                "id": "doc__img_p1_i0__abc123",
                "doc_id": "doc",
                "source_path": "/tmp/doc.pdf",
                "section_path": ["Page 1"],
                "section_index": 1,
                "block_type": "image",
                "block_index": 0,
                "text": "A civic planning map.",
                "extra": {"source": "pdf", "page": 1, "image_width": 100, "image_height": 100},
            }
        ]

        with patch("vision.pipeline.run_vision_pipeline", return_value=chunks):
            n = run_vision_pipeline_to_jsonl("/fake/doc.pdf", str(out_file))

        assert n == 1
        assert out_file.exists()
        lines = out_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["block_type"] == "image"
        assert parsed["text"] == "A civic planning map."

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out_file = tmp_path / "deep" / "nested" / "out.jsonl"
        with patch("vision.pipeline.run_vision_pipeline", return_value=[]):
            run_vision_pipeline_to_jsonl("/fake/doc.pdf", str(out_file))
        assert out_file.parent.exists()

    def test_returns_zero_when_no_images(self, tmp_path: Path) -> None:
        out_file = tmp_path / "empty.jsonl"
        with patch("vision.pipeline.run_vision_pipeline", return_value=[]):
            n = run_vision_pipeline_to_jsonl("/fake/doc.pdf", str(out_file))
        assert n == 0
        assert out_file.read_text() == ""
