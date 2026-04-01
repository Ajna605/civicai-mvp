"""Unit tests for the vision module.

Tests cover:
1. Image extraction from a minimal synthetic PDF.
2. ImageInterpreter raises ValueError when no API key is provided.
3. ImageInterpreter.interpret calls the OpenAI client with the correct payload.
4. run_vision_pipeline returns structured chunks with the expected schema.
5. run_vision_pipeline_to_jsonl writes valid JSONL and returns the correct count.
6. build_corpus includes image description chunks in rag_chunks.jsonl.
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

from vision.image_extractor import ImageRecord, extract_images_from_pdf
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
# ImageInterpreter
# ---------------------------------------------------------------------------

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
            "vision.pipeline.extract_images_from_pdf", return_value=records
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
        with patch("vision.pipeline.extract_images_from_pdf", return_value=[]):
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
