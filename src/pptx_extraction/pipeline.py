"""High-level extraction workflows with atomic output directories."""

from __future__ import annotations

import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from .exceptions import InputValidationError, OutputExistsError
from .exporters import export_record
from .extractors import PptxExtractor
from .models import ExtractionOptions, PresentationRecord
from .security import SUPPORTED_OOXML_EXTENSIONS, sha256_file


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    output_dir: Path
    files: dict[str, Path]
    record: PresentationRecord


@dataclass(frozen=True, slots=True)
class BatchItem:
    source: Path
    output_dir: Path | None
    success: bool
    error: str | None = None


def extract_file(
    source: str | Path,
    output_dir: str | Path,
    *,
    options: ExtractionOptions | None = None,
    formats: tuple[str, ...] = ("json", "markdown"),
    overwrite: bool = False,
    tesseract_command: str | None = None,
) -> ExtractionResult:
    source_path = Path(source).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_dir():
            raise InputValidationError(f"Output path is not a directory: {destination}")
        if any(destination.iterdir()):
            if not overwrite:
                raise OutputExistsError(
                    f"Output directory is not empty: {destination}",
                    hint="Choose another directory or pass --overwrite.",
                )
            _remove_exact_output(destination)
        else:
            destination.rmdir()

    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=str(destination.parent)))
    try:
        extractor = PptxExtractor(
            options=options or ExtractionOptions(),
            tesseract_command=tesseract_command,
        )
        record = extractor.extract(source_path, temporary / "assets")
        files = export_record(record, temporary, formats)
        os.replace(temporary, destination)
        resolved_files = {
            name: destination / path.relative_to(temporary) for name, path in files.items()
        }
        return ExtractionResult(destination, resolved_files, record)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def inspect_file(
    source: str | Path,
    *,
    options: ExtractionOptions | None = None,
) -> PresentationRecord:
    base = options or ExtractionOptions()
    inspect_options = replace(base, include_assets=False, ocr_backend="none")
    return PptxExtractor(options=inspect_options).extract(source)


def discover_sources(inputs: list[str | Path], recursive: bool = True) -> list[Path]:
    discovered: dict[Path, None] = {}
    for value in inputs:
        candidate = Path(value).expanduser().resolve()
        if candidate.is_file():
            if candidate.suffix.lower() in SUPPORTED_OOXML_EXTENSIONS:
                discovered[candidate] = None
            continue
        if candidate.is_dir():
            iterator = candidate.rglob("*") if recursive else candidate.glob("*")
            for path in iterator:
                if path.is_file() and path.suffix.lower() in SUPPORTED_OOXML_EXTENSIONS:
                    discovered[path.resolve()] = None
            continue
        raise InputValidationError(f"Input does not exist: {candidate}")
    return sorted(discovered)


def batch_extract(
    sources: list[str | Path],
    output_root: str | Path,
    *,
    options: ExtractionOptions | None = None,
    formats: tuple[str, ...] = ("json", "markdown"),
    workers: int = 2,
    overwrite: bool = False,
    recursive: bool = True,
) -> list[BatchItem]:
    if workers < 1 or workers > 32:
        raise InputValidationError("workers must be between 1 and 32.")
    inputs = discover_sources(sources, recursive=recursive)
    if not inputs:
        raise InputValidationError("No supported presentations were found.")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    def run(path: Path) -> BatchItem:
        target = root / f"{path.stem}-{sha256_file(path)[:8]}"
        try:
            result = extract_file(
                path,
                target,
                options=options,
                formats=formats,
                overwrite=overwrite,
            )
            return BatchItem(path, result.output_dir, True)
        except Exception as exc:
            return BatchItem(path, None, False, str(exc))

    by_source: dict[Path, BatchItem] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="pptx-extraction") as pool:
        futures = {pool.submit(run, path): path for path in inputs}
        for future in as_completed(futures):
            by_source[futures[future]] = future.result()
    return [by_source[path] for path in inputs]


def _remove_exact_output(path: Path) -> None:
    resolved = path.resolve()
    if resolved == Path(resolved.anchor) or resolved == resolved.parent:
        raise InputValidationError(f"Refusing to overwrite broad path: {resolved}")
    shutil.rmtree(resolved)
