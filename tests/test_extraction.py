from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pptx_extraction.exceptions import InputValidationError, OutputExistsError
from pptx_extraction.models import ExtractionOptions
from pptx_extraction.pipeline import extract_file, inspect_file
from tests.helpers import build_sample_deck


class ExtractionTests(unittest.TestCase):
    def test_extracts_structured_content_and_deduplicates_assets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sample_deck(root / "sample.pptx")
            result = extract_file(
                source,
                root / "output",
                formats=("json", "markdown", "text"),
            )

            self.assertEqual(result.record.summary["slides"], 2)
            first = result.record.slides[0]
            self.assertEqual(first.title, "Quarterly review")
            self.assertTrue(any(block.text == "Revenue increased" for block in first.text_blocks))
            self.assertEqual(first.tables[0].rows[1], ("ARR", "42"))
            self.assertEqual(first.charts[0].series[0].values, (30.0, 42.0))
            self.assertIn("Confidential speaker note", first.notes or "")
            self.assertTrue(result.record.slides[1].hidden)
            self.assertEqual(first.images[0].sha256, result.record.slides[1].images[0].sha256)
            self.assertEqual(len(list((root / "output" / "assets").iterdir())), 1)

            payload = json.loads((root / "output" / "presentation.json").read_text("utf-8"))
            self.assertEqual(payload["schema_version"], "1.0")
            self.assertEqual(payload["slides"][0]["images"][0]["alt_text"], "Blue test image")
            markdown = (root / "output" / "presentation.md").read_text("utf-8")
            self.assertIn("## Slide 1: Quarterly review", markdown)
            self.assertIn("| ARR | 42 |", markdown)

    def test_metadata_can_be_redacted_and_assets_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sample_deck(root / "sample.pptx")
            record = inspect_file(
                source,
                options=ExtractionOptions(redact_metadata=True),
            )
            self.assertEqual(record.metadata["author"], "[redacted]")
            self.assertIsNone(record.slides[0].images[0].asset_path)

    def test_nonempty_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sample_deck(root / "sample.pptx")
            output = root / "output"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(OutputExistsError):
                extract_file(source, output)
            self.assertEqual((output / "keep.txt").read_text("utf-8"), "keep")

    def test_output_file_is_rejected_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sample_deck(root / "sample.pptx")
            output = root / "output"
            output.write_text("keep", encoding="utf-8")
            with self.assertRaises(InputValidationError):
                extract_file(source, output)
            self.assertEqual(output.read_text("utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
