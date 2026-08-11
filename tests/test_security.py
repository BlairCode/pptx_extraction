from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from pptx_extraction.exceptions import InputValidationError, UnsafePackageError
from pptx_extraction.security import PackageLimits, validate_package
from tests.helpers import build_sample_deck


class SecurityTests(unittest.TestCase):
    def test_rejects_non_zip_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fake.pptx"
            source.write_text("not a package", encoding="utf-8")
            with self.assertRaises(InputValidationError):
                validate_package(source)

    def test_rejects_traversal_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "traversal.pptx"
            with zipfile.ZipFile(source, "w") as archive:
                archive.writestr("[Content_Types].xml", "<Types/>")
                archive.writestr("ppt/presentation.xml", "<p:presentation/>")
                archive.writestr("../outside.txt", "bad")
            with self.assertRaises(UnsafePackageError):
                validate_package(source)

    def test_applies_configured_source_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = build_sample_deck(Path(directory) / "sample.pptx")
            with self.assertRaises(UnsafePackageError):
                validate_package(source, PackageLimits(max_source_bytes=1))


if __name__ == "__main__":
    unittest.main()
