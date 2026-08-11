"""Lazy OCR adapters. Core extraction never imports an OCR engine."""

from __future__ import annotations

import io
from typing import Protocol

from PIL import Image

from .exceptions import OptionalDependencyError


class OCRBackend(Protocol):
    name: str

    def recognize(self, image_bytes: bytes, language: str) -> str:
        """Return recognized text without mutating the source image."""


class NoOCR:
    name = "none"

    def recognize(self, image_bytes: bytes, language: str) -> str:
        return ""


class TesseractOCR:
    name = "tesseract"

    def __init__(self, executable: str | None = None) -> None:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OptionalDependencyError(
                "Tesseract OCR adapter is not installed.",
                hint="Install `pptx-extraction[ocr]` and the Tesseract executable.",
            ) from exc
        self._module = pytesseract
        if executable:
            self._module.pytesseract.tesseract_cmd = executable

    def recognize(self, image_bytes: bytes, language: str) -> str:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return str(self._module.image_to_string(image, lang=language)).strip()
        except self._module.TesseractNotFoundError as exc:
            raise OptionalDependencyError(
                "Tesseract executable was not found.",
                hint="Install Tesseract or pass --tesseract-command.",
            ) from exc


def create_ocr_backend(name: str, executable: str | None = None) -> OCRBackend:
    normalized = name.strip().lower()
    if normalized in {"", "none", "off"}:
        return NoOCR()
    if normalized == "tesseract":
        return TesseractOCR(executable)
    raise OptionalDependencyError(f"Unknown OCR backend: {name}")
