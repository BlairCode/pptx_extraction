#!/usr/bin/env python3
"""Thin, deterministic Agent wrapper around the pptx_extraction package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a PowerPoint deck with pptx_extraction.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--format", action="append", choices=("json", "markdown", "text"))
    parser.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    parser.add_argument("--ocr-language", default="eng")
    parser.add_argument("--no-assets", action="store_true")
    parser.add_argument("--include-private-metadata", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from pptx_extraction.exceptions import PptxExtractionError
        from pptx_extraction.models import ExtractionOptions
        from pptx_extraction.pipeline import extract_file, inspect_file
    except ImportError:
        print(
            json.dumps(
                {
                    "error": {
                        "code": "pptx_extraction_not_installed",
                        "message": (
                            "Install the pptx_extraction project package before using this skill."
                        ),
                    }
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        return 5

    options = ExtractionOptions(
        include_assets=not args.no_assets,
        redact_metadata=not args.include_private_metadata,
        ocr_backend=args.ocr,
        ocr_language=args.ocr_language,
    )
    try:
        if args.inspect:
            record = inspect_file(args.source, options=options)
            print(json.dumps(record.summary, ensure_ascii=False, indent=2))
            return 0
        if args.output is None:
            raise ValueError("--output is required unless --inspect is used.")
        result = extract_file(
            args.source,
            args.output,
            options=options,
            formats=tuple(args.format or ("json", "markdown")),
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "output_dir": str(result.output_dir),
                    "summary": result.record.summary,
                    "warnings": [warning.code for warning in result.record.warnings],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    except (PptxExtractionError, ValueError) as exc:
        code = getattr(exc, "code", "invalid_arguments")
        print(
            json.dumps({"error": {"code": code, "message": str(exc)}}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
