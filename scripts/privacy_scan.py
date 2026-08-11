#!/usr/bin/env python3
"""Fail when publishable sources contain common private artifacts or secrets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

PUBLISH_ROOTS = (
    ".github",
    "agent-skill",
    "docs",
    "schemas",
    "scripts",
    "src",
    "tests",
)
ROOT_FILES = (
    ".env.example",
    ".gitignore",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "README.md",
    "SECURITY.md",
    "pyproject.toml",
)
TEXT_SUFFIXES = {".md", ".py", ".toml", ".yaml", ".yml", ".json", ".txt", ".example"}
BANNED_ARTIFACT_SUFFIXES = {".ppt", ".pptx", ".pptm", ".potx", ".ppsx", ".mp3", ".wav", ".srt"}
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
PATTERNS = {
    "Windows absolute path": re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!YOUR_|path[\\/])"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
    "probable secret assignment": re.compile(
        r"(?i)(?:api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"]{8,}"
    ),
    "personal email": re.compile(r"\b[\w.+-]+@(?!example\.invalid\b)[\w.-]+\.[A-Za-z]{2,}\b"),
}


def publishable_files(root: Path) -> list[Path]:
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for relative in PUBLISH_ROOTS:
        base = root / relative
        if base.is_dir():
            files.extend(
                path
                for path in base.rglob("*")
                if path.is_file()
                and not EXCLUDED_PARTS.intersection(path.parts)
                and path.suffix.lower() not in EXCLUDED_SUFFIXES
            )
    return sorted(set(files))


def scan(root: Path) -> list[str]:
    findings: list[str] = []
    scanner = (root / "scripts" / "privacy_scan.py").resolve()
    for path in publishable_files(root):
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() in BANNED_ARTIFACT_SUFFIXES:
            findings.append(f"{relative}: publishable binary/private artifact")
            continue
        if path.resolve() == scanner or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"{relative}: non-UTF-8 text file")
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}")
    return findings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    findings = scan(args.root.resolve())
    if findings:
        print("Privacy scan failed:", file=sys.stderr)
        print("\n".join(f"- {item}" for item in findings), file=sys.stderr)
        return 1
    print("Privacy scan passed: no common private artifacts or secrets in publishable files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
