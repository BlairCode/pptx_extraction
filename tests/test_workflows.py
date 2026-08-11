from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pptx_extraction.converter import convert_legacy
from pptx_extraction.exceptions import InputValidationError, OptionalDependencyError
from pptx_extraction.ocr import NoOCR, create_ocr_backend
from pptx_extraction.pipeline import batch_extract, discover_sources
from tests.helpers import build_sample_deck


class WorkflowTests(unittest.TestCase):
    def test_batch_reports_success_and_failure_without_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_sample_deck(root / "valid.pptx")
            (root / "broken.pptx").write_text("not a zip", encoding="utf-8")
            items = batch_extract([root], root / "outputs", workers=2)
            self.assertEqual(len(items), 2)
            self.assertEqual(sum(item.success for item in items), 1)
            self.assertEqual(sum(not item.success for item in items), 1)

    def test_discovery_rejects_missing_input(self) -> None:
        with self.assertRaises(InputValidationError):
            discover_sources(["definitely-missing-directory"])

    def test_legacy_conversion_errors_are_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.ppt"
            source.write_bytes(b"legacy")
            with self.assertRaises(OptionalDependencyError):
                convert_legacy(source, root / "out", soffice_command="missing-soffice-command")
            wrong = root / "modern.pptx"
            wrong.write_bytes(b"modern")
            with self.assertRaises(InputValidationError):
                convert_legacy(wrong, root / "out")

    def test_no_ocr_is_deterministic_and_unknown_backend_fails(self) -> None:
        self.assertEqual(NoOCR().recognize(b"anything", "eng"), "")
        self.assertIsInstance(create_ocr_backend("none"), NoOCR)
        with self.assertRaises(OptionalDependencyError):
            create_ocr_backend("unknown")


if __name__ == "__main__":
    unittest.main()
