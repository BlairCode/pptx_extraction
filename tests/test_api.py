from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

try:
    from fastapi.testclient import TestClient
except ImportError:  # optional dependency in core-only environments
    TestClient = None  # type: ignore[misc,assignment]

from pptx_extraction.api import ServiceSettings, create_app
from tests.helpers import build_sample_deck


@unittest.skipIf(TestClient is None, "API extras are not installed")
class ApiTests(unittest.TestCase):
    def test_job_upload_poll_and_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = build_sample_deck(root / "sample.pptx")
            app = create_app(ServiceSettings(root / "jobs", workers=1))
            with TestClient(app) as client:  # type: ignore[operator]
                response = client.post(
                    "/v1/jobs",
                    files={
                        "file": (
                            source.name,
                            source.read_bytes(),
                            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                        )
                    },
                )
                self.assertEqual(response.status_code, 202)
                job_id = response.json()["id"]
                for _ in range(100):
                    status = client.get(f"/v1/jobs/{job_id}").json()
                    if status["status"] in {"succeeded", "failed"}:
                        break
                    time.sleep(0.01)
                self.assertEqual(status["status"], "succeeded")
                result = client.get(f"/v1/jobs/{job_id}/result")
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()["schema_version"], "1.0")
                self.assertEqual(result.json()["metadata"]["author"], "[redacted]")

    def test_rejects_unsupported_upload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = create_app(ServiceSettings(Path(directory) / "jobs", workers=1))
            with TestClient(app) as client:  # type: ignore[operator]
                response = client.post(
                    "/v1/jobs", files={"file": ("notes.txt", b"hello", "text/plain")}
                )
                self.assertEqual(response.status_code, 415)


if __name__ == "__main__":
    unittest.main()
