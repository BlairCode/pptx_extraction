"""Optional single-node job API for pptx_extraction."""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from .exceptions import OptionalDependencyError
from .models import ExtractionOptions
from .pipeline import extract_file
from .security import SUPPORTED_OOXML_EXTENSIONS


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    work_dir: Path
    max_upload_bytes: int = 50 * 1024 * 1024
    workers: int = 2

    @classmethod
    def from_env(cls) -> ServiceSettings:
        return cls(
            work_dir=Path(os.getenv("PPTX_EXTRACTION_WORK_DIR", "./work")).expanduser().resolve(),
            max_upload_bytes=int(os.getenv("PPTX_EXTRACTION_MAX_UPLOAD_MB", "50")) * 1024 * 1024,
            workers=max(1, min(int(os.getenv("PPTX_EXTRACTION_WORKERS", "2")), 16)),
        )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str, source_name: str) -> None:
        with self._lock:
            self._jobs[job_id] = {"id": job_id, "status": "queued", "source_name": source_name}

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            value = self._jobs.get(job_id)
            return dict(value) if value else None


def create_app(settings: ServiceSettings | None = None) -> Any:
    try:
        from fastapi import (  # type: ignore[import-not-found]
            FastAPI,
            File,
            HTTPException,
            UploadFile,
        )
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            "HTTP API dependencies are not installed.",
            hint="Install `pptx-extraction[api]`.",
        ) from exc

    # FastAPI resolves postponed endpoint annotations from module globals.
    globals().update({"UploadFile": UploadFile, "File": File})

    active = settings or ServiceSettings.from_env()
    active.work_dir.mkdir(parents=True, exist_ok=True)
    store = JobStore()
    executor = ThreadPoolExecutor(
        max_workers=active.workers, thread_name_prefix="pptx-extraction-api"
    )

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        yield
        executor.shutdown(wait=True, cancel_futures=True)

    app = FastAPI(title="pptx_extraction API", version="2.0.0", docs_url="/docs", lifespan=lifespan)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/jobs", status_code=202)
    async def create_job(file: Annotated[UploadFile, File(...)]) -> dict[str, str]:
        original_name = Path(file.filename or "upload.pptx").name
        suffix = Path(original_name).suffix.lower()
        if suffix not in SUPPORTED_OOXML_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported PowerPoint format.")
        job_id = uuid.uuid4().hex
        job_dir = active.work_dir / job_id
        job_dir.mkdir(parents=False, exist_ok=False)
        source_path = job_dir / f"source{suffix}"
        total = 0
        try:
            with source_path.open("wb") as stream:
                while chunk := await file.read(1024 * 1024):
                    total += len(chunk)
                    if total > active.max_upload_bytes:
                        raise HTTPException(
                            status_code=413, detail="Upload exceeds configured limit."
                        )
                    stream.write(chunk)
        except Exception:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise
        finally:
            await file.close()
        store.create(job_id, original_name)
        executor.submit(_run_job, store, job_id, source_path, job_dir / "result")
        return {"id": job_id, "status": "queued"}

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str) -> Any:
        job = store.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail=f"Job is {job['status']}.")
        result_path = active.work_dir / job_id / "result" / "presentation.json"
        return JSONResponse(json.loads(result_path.read_text(encoding="utf-8")))

    return app


def _run_job(store: JobStore, job_id: str, source_path: Path, output_dir: Path) -> None:
    store.update(job_id, status="running")
    try:
        result = extract_file(
            source_path,
            output_dir,
            options=ExtractionOptions(redact_metadata=True),
            formats=("json", "markdown"),
        )
        store.update(job_id, status="succeeded", summary=result.record.summary)
    except Exception as exc:
        store.update(job_id, status="failed", error=str(exc))
