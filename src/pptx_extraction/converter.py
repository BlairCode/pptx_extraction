"""Explicit legacy PowerPoint conversion through LibreOffice."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .exceptions import ExtractionError, InputValidationError, OptionalDependencyError
from .security import LEGACY_EXTENSIONS


def convert_legacy(
    source: str | Path,
    output_dir: str | Path,
    *,
    soffice_command: str = "soffice",
    timeout_seconds: int = 120,
) -> Path:
    input_path = Path(source).expanduser().resolve()
    target_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise InputValidationError(f"Presentation does not exist: {input_path}")
    if input_path.suffix.lower() not in LEGACY_EXTENSIONS:
        raise InputValidationError("convert accepts only .ppt, .pot or .pps input.")
    executable = shutil.which(soffice_command)
    if not executable:
        raise OptionalDependencyError(
            "LibreOffice executable was not found.",
            hint="Install LibreOffice or pass --soffice with its executable path.",
        )
    target_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            executable,
            "--headless",
            "--convert-to",
            "pptx",
            "--outdir",
            str(target_dir),
            str(input_path),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    destination = target_dir / f"{input_path.stem}.pptx"
    if completed.returncode != 0 or not destination.is_file():
        detail = (completed.stderr or completed.stdout or "unknown conversion error").strip()
        raise ExtractionError(f"LibreOffice conversion failed: {detail}")
    return destination
