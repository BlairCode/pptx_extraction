from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pptx_extraction.cli import EXIT_SUCCESS, EXIT_USAGE, main
from tests.helpers import build_sample_deck


class CliTests(unittest.TestCase):
    def test_validate_and_inspect_emit_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = build_sample_deck(Path(directory) / "sample.pptx")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["validate", str(source)])
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(json.loads(output.getvalue())["status"], "valid")

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["inspect", str(source)])
            self.assertEqual(code, EXIT_SUCCESS)
            self.assertEqual(json.loads(output.getvalue())["slides"], 2)

    def test_missing_input_uses_usage_exit(self) -> None:
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors):
            code = main(["validate", "missing.pptx"])
        self.assertEqual(code, EXIT_USAGE)
        self.assertEqual(json.loads(errors.getvalue())["error"]["code"], "invalid_input")


if __name__ == "__main__":
    unittest.main()
