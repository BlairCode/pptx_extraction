"""pptx_extraction public Python API."""

from .models import ExtractionOptions, PresentationRecord
from .pipeline import extract_file, inspect_file

__all__ = ["ExtractionOptions", "PresentationRecord", "extract_file", "inspect_file"]
__version__ = "2.0.0"
