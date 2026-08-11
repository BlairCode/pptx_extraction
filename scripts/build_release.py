#!/usr/bin/env python3
"""Create deterministic, privacy-scanned project and Agent Skill release archives."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from privacy_scan import publishable_files, scan

VERSION = "2.0.0"
ZIP_TIMESTAMP = (2026, 8, 11, 0, 0, 0)


def write_zip(destination: Path, entries: list[tuple[Path, str]]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    with zipfile.ZipFile(
        destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for source, archive_name in sorted(entries, key=lambda item: item[1]):
            info = zipfile.ZipInfo(archive_name, ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("release"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    findings = scan(root)
    if findings:
        raise SystemExit("Privacy scan failed:\n" + "\n".join(findings))

    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output
    project_entries = [
        (path, f"pptx_extraction-{VERSION}/{path.relative_to(root).as_posix()}")
        for path in publishable_files(root)
        if "agent-skill" not in path.relative_to(root).parts
    ]
    project_archive = output / f"pptx_extraction-v{VERSION}.zip"
    write_zip(project_archive, project_entries)

    skill_root = root / "agent-skill" / "pptx-extraction"
    skill_entries = [
        (path, f"pptx-extraction/{path.relative_to(skill_root).as_posix()}")
        for path in skill_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    skill_archive = output / f"pptx_extraction-skill-v{VERSION}.zip"
    write_zip(skill_archive, skill_entries)
    print(project_archive)
    print(skill_archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
