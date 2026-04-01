"""Interpret images using the OpenAI Vision API (GPT-4o)."""
from __future__ import annotations

import os
from typing import Optional

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_AVAILABLE = False


_DEFAULT_PROMPT = (
    "You are analyzing an image extracted from a civic planning or policy document. "
    "Describe the image thoroughly, focusing on any text, labels, data, maps, charts, "
    "diagrams, or figures present. "
    "If the image contains a map, describe its geographic features and any labels. "
    "If it contains a chart or graph, summarize the data it represents. "
    "If it contains blocks of text or a table, transcribe the key content. "
    "Be concise but comprehensive."
)


class ImageInterpreter:
    """Interprets images using the OpenAI Vision API.

    Example::

        interpreter = ImageInterpreter()  # reads OPENAI_API_KEY from env
        description = interpreter.interpret(image_b64, ext="png")
    """

    DEFAULT_MODEL = "gpt-4o"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
    ) -> None:
        """
        Args:
            api_key: OpenAI API key. Falls back to the ``OPENAI_API_KEY``
                environment variable when not provided.
            model: A vision-capable OpenAI model name (e.g. ``'gpt-4o'``).
            max_tokens: Maximum tokens allowed in the description response.

        Raises:
            RuntimeError: If the ``openai`` package is not installed.
            ValueError: If no API key can be resolved.
        """
        if not _OPENAI_AVAILABLE:
            raise RuntimeError(
                "openai is not installed. Run: pip install openai"
            )

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenAI API key provided. "
                "Set the OPENAI_API_KEY environment variable or pass api_key."
            )

        self._client = OpenAI(api_key=resolved_key)
        self.model = model
        self.max_tokens = max_tokens

    def interpret(
        self,
        image_b64: str,
        ext: str = "png",
        context: str = "",
    ) -> str:
        """Return a natural-language description of a base64-encoded image.

        Args:
            image_b64: Base64-encoded image bytes (no data-URI prefix needed).
            ext: Image format extension, e.g. ``'png'``, ``'jpeg'``.
            context: Optional surrounding context such as page number or
                section name; appended to the prompt when provided.

        Returns:
            Text description of the image content, or an empty string if the
            model returns no content.
        """
        prompt = _DEFAULT_PROMPT
        if context:
            prompt = f"{prompt}\n\nContext: {context}"

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""
