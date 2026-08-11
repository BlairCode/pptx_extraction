"""Dependency-light pptx_extraction command-line interface."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from . import __version__
from .converter import convert_legacy
from .exceptions import (
    InputValidationError,
    OptionalDependencyError,
    OutputExistsError,
    PptxExtractionError,
)
from .models import ExtractionOptions
from .pipeline import batch_extract, extract_file, inspect_file
from .security import validate_package

EXIT_SUCCESS = 0
EXIT_USAGE = 2
EXIT_EXTRACTION = 3
EXIT_PARTIAL = 4
EXIT_DEPENDENCY = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pptx-extraction",
        description="Extract PowerPoint decks into traceable, AI-ready data.",
    )
    parser.add_argument("--version", action="version", version=f"pptx_extraction {__version__}")
    parser.add_argument("--verbose", action="store_true", help="Enable diagnostic logs.")
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser("extract", help="Extract one OOXML presentation.")
    extract.add_argument("source", type=Path)
    extract.add_argument("--output", "-o", type=Path, required=True)
    extract.add_argument(
        "--format",
        dest="formats",
        action="append",
        choices=("json", "markdown", "text"),
        help="Repeat for multiple formats; defaults to json and markdown.",
    )
    _add_extraction_options(extract)
    extract.add_argument("--overwrite", action="store_true")
    extract.add_argument("--tesseract-command")

    inspect = commands.add_parser("inspect", help="Print a deck summary without writing assets.")
    inspect.add_argument("source", type=Path)
    inspect.add_argument("--full", action="store_true", help="Print the complete JSON record.")
    inspect.add_argument("--redact-metadata", action="store_true")

    validate = commands.add_parser("validate", help="Validate OOXML structure and safety limits.")
    validate.add_argument("source", type=Path)

    batch = commands.add_parser("batch", help="Extract files/directories concurrently.")
    batch.add_argument("sources", type=Path, nargs="+")
    batch.add_argument("--output", "-o", type=Path, required=True)
    batch.add_argument(
        "--format", dest="formats", action="append", choices=("json", "markdown", "text")
    )
    batch.add_argument("--workers", type=int, default=2)
    batch.add_argument("--no-recursive", action="store_true")
    batch.add_argument("--overwrite", action="store_true")
    _add_extraction_options(batch)

    convert = commands.add_parser("convert", help="Convert one legacy deck with LibreOffice.")
    convert.add_argument("source", type=Path)
    convert.add_argument("--output", "-o", type=Path, required=True)
    convert.add_argument("--soffice", default="soffice")
    convert.add_argument("--timeout", type=int, default=120)
    return parser


def _add_extraction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-assets", action="store_true")
    parser.add_argument("--no-notes", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")
    parser.add_argument("--redact-metadata", action="store_true")
    parser.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    parser.add_argument("--ocr-language", default="eng")


def _options(args: argparse.Namespace) -> ExtractionOptions:
    return ExtractionOptions(
        include_assets=not getattr(args, "no_assets", False),
        include_notes=not getattr(args, "no_notes", False),
        include_metadata=not getattr(args, "no_metadata", False),
        redact_metadata=getattr(args, "redact_metadata", False),
        ocr_backend=getattr(args, "ocr", "none"),
        ocr_language=getattr(args, "ocr_language", "eng"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if args.command == "extract":
            result = extract_file(
                args.source,
                args.output,
                options=_options(args),
                formats=tuple(args.formats or ("json", "markdown")),
                overwrite=args.overwrite,
                tesseract_command=args.tesseract_command,
            )
            _print_json(
                {
                    "status": "ok",
                    "output_dir": str(result.output_dir),
                    "files": {key: str(value) for key, value in result.files.items()},
                    "summary": result.record.summary,
                }
            )
            return EXIT_SUCCESS
        if args.command == "inspect":
            record = inspect_file(
                args.source,
                options=ExtractionOptions(redact_metadata=args.redact_metadata),
            )
            _print_json(record.to_dict() if args.full else record.summary)
            return EXIT_SUCCESS
        if args.command == "validate":
            report = validate_package(args.source)
            _print_json(
                {
                    "status": "valid",
                    "entries": report.entries,
                    "expanded_bytes": report.expanded_bytes,
                    "has_macros": report.has_macros,
                    "warnings": report.warnings,
                }
            )
            return EXIT_SUCCESS
        if args.command == "batch":
            items = batch_extract(
                list(args.sources),
                args.output,
                options=_options(args),
                formats=tuple(args.formats or ("json", "markdown")),
                workers=args.workers,
                overwrite=args.overwrite,
                recursive=not args.no_recursive,
            )
            payload = [
                {
                    "source": str(item.source),
                    "success": item.success,
                    "output_dir": str(item.output_dir) if item.output_dir else None,
                    "error": item.error,
                }
                for item in items
            ]
            _print_json({"results": payload})
            return EXIT_SUCCESS if all(item.success for item in items) else EXIT_PARTIAL
        if args.command == "convert":
            destination = convert_legacy(
                args.source,
                args.output,
                soffice_command=args.soffice,
                timeout_seconds=args.timeout,
            )
            _print_json({"status": "ok", "output": str(destination)})
            return EXIT_SUCCESS
        raise InputValidationError(f"Unknown command: {args.command}")
    except OptionalDependencyError as exc:
        _print_error(exc)
        return EXIT_DEPENDENCY
    except (InputValidationError, OutputExistsError) as exc:
        _print_error(exc)
        return EXIT_USAGE
    except PptxExtractionError as exc:
        _print_error(exc)
        return EXIT_EXTRACTION
    except Exception as exc:  # final CLI boundary; library calls retain typed exceptions
        logging.getLogger(__name__).exception("Unexpected failure")
        _print_json({"error": {"code": "unexpected_error", "message": str(exc)}}, sys.stderr)
        return EXIT_EXTRACTION


def _print_error(error: PptxExtractionError) -> None:
    payload = {"error": {"code": error.code, "message": error.message}}
    if error.hint:
        payload["error"]["hint"] = error.hint
    _print_json(payload, sys.stderr)


def _print_json(payload: object, stream: TextIO | None = None) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2), file=stream or sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
