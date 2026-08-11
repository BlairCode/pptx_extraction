"""Pre-parse validation for untrusted OOXML packages."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .exceptions import InputValidationError, UnsafePackageError

SUPPORTED_OOXML_EXTENSIONS = frozenset({".pptx", ".pptm", ".potx", ".ppsx"})
LEGACY_EXTENSIONS = frozenset({".ppt", ".pot", ".pps"})


@dataclass(frozen=True, slots=True)
class PackageLimits:
    max_source_bytes: int = 250 * 1024 * 1024
    max_entries: int = 10_000
    max_expanded_bytes: int = 1_000 * 1024 * 1024
    max_compression_ratio: float = 250.0


@dataclass(slots=True)
class PackageReport:
    entries: int
    expanded_bytes: int
    has_macros: bool
    warnings: list[str] = field(default_factory=list)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(path: str | Path, limits: PackageLimits | None = None) -> PackageReport:
    source = Path(path).expanduser().resolve()
    active_limits = limits or PackageLimits()
    if not source.is_file():
        raise InputValidationError(f"Presentation does not exist: {source}")
    if source.suffix.lower() not in SUPPORTED_OOXML_EXTENSIONS:
        if source.suffix.lower() in LEGACY_EXTENSIONS:
            raise InputValidationError(
                f"Legacy format requires conversion: {source.suffix}",
                hint="Run `pptx-extraction convert` with LibreOffice installed.",
            )
        raise InputValidationError(
            f"Unsupported presentation format: {source.suffix or '[no extension]'}"
        )
    size = source.stat().st_size
    if size == 0:
        raise InputValidationError("Presentation is empty.")
    if size > active_limits.max_source_bytes:
        raise UnsafePackageError(
            f"Source size {size} exceeds limit {active_limits.max_source_bytes} bytes."
        )
    if not zipfile.is_zipfile(source):
        raise InputValidationError("File is not a valid OOXML ZIP package.")

    expanded = 0
    has_macros = False
    warnings: list[str] = []
    with zipfile.ZipFile(source) as archive:
        entries = archive.infolist()
        if len(entries) > active_limits.max_entries:
            raise UnsafePackageError(
                f"Archive has {len(entries)} entries; limit is {active_limits.max_entries}."
            )
        names: set[str] = set()
        for entry in entries:
            normalized_name = entry.filename.replace("\\", "/")
            normalized = PurePosixPath(normalized_name)
            if normalized.is_absolute() or ".." in normalized.parts:
                raise UnsafePackageError(f"Unsafe archive path: {entry.filename}")
            if entry.flag_bits & 0x1:
                raise UnsafePackageError(
                    f"Encrypted archive entry is unsupported: {entry.filename}"
                )
            expanded += entry.file_size
            if expanded > active_limits.max_expanded_bytes:
                raise UnsafePackageError(
                    f"Expanded archive exceeds {active_limits.max_expanded_bytes} bytes."
                )
            if entry.file_size:
                ratio = entry.file_size / max(entry.compress_size, 1)
                if ratio > active_limits.max_compression_ratio:
                    raise UnsafePackageError(
                        f"Suspicious compression ratio {ratio:.1f} for {entry.filename}."
                    )
            names.add(normalized_name)
            has_macros = has_macros or normalized_name.lower() == "ppt/vbaproject.bin"

    required = {"[Content_Types].xml", "ppt/presentation.xml"}
    missing = sorted(required - names)
    if missing:
        raise InputValidationError(f"OOXML package is missing: {', '.join(missing)}")
    if has_macros:
        warnings.append("The package contains VBA macros; pptx_extraction does not execute them.")
    return PackageReport(len(entries), expanded, has_macros, warnings)
