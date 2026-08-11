"""Stable exception hierarchy used by the library, CLI and API."""

from __future__ import annotations


class PptxExtractionError(Exception):
    """Base error with a machine-readable code."""

    code = "pptx_extraction_error"

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint


class InputValidationError(PptxExtractionError):
    code = "invalid_input"


class UnsafePackageError(InputValidationError):
    code = "unsafe_package"


class ExtractionError(PptxExtractionError):
    code = "extraction_failed"


class OptionalDependencyError(PptxExtractionError):
    code = "optional_dependency_missing"


class OutputExistsError(PptxExtractionError):
    code = "output_exists"
