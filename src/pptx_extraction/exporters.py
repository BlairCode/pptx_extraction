"""Deterministic serializers for pptx_extraction records."""

from __future__ import annotations

import json
from pathlib import Path

from .exceptions import InputValidationError
from .models import ChartRecord, ImageRecord, PresentationRecord, TableRecord, TextBlock

SUPPORTED_FORMATS = frozenset({"json", "markdown", "text"})


def export_record(
    record: PresentationRecord,
    output_dir: Path,
    formats: tuple[str, ...] = ("json", "markdown"),
) -> dict[str, Path]:
    normalized = tuple(dict.fromkeys(item.lower().strip() for item in formats if item.strip()))
    invalid = sorted(set(normalized) - SUPPORTED_FORMATS)
    if invalid:
        raise InputValidationError(f"Unsupported output format(s): {', '.join(invalid)}")
    if not normalized:
        raise InputValidationError("At least one output format is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if "json" in normalized:
        destination = output_dir / "presentation.json"
        destination.write_text(
            json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written["json"] = destination
    if "markdown" in normalized:
        destination = output_dir / "presentation.md"
        destination.write_text(to_markdown(record), encoding="utf-8")
        written["markdown"] = destination
    if "text" in normalized:
        destination = output_dir / "presentation.txt"
        destination.write_text(to_text(record), encoding="utf-8")
        written["text"] = destination
    return written


def to_markdown(record: PresentationRecord) -> str:
    lines = [
        f"# {record.metadata.get('title') or Path(record.source_name).stem}",
        "",
        f"> Source: `{record.source_name}` · SHA-256: `{record.source_sha256}` · "
        f"Slides: {len(record.slides)} · Schema: {record.schema_version}",
        "",
    ]
    for slide in record.slides:
        suffix = " _(hidden)_" if slide.hidden else ""
        lines.extend([f"## Slide {slide.number}: {slide.title or 'Untitled'}{suffix}", ""])
        ordered: list[tuple[int, int, object]] = []
        ordered.extend((item.order, 0, item) for item in slide.text_blocks)
        ordered.extend((item.order, 1, item) for item in slide.tables)
        ordered.extend((item.order, 2, item) for item in slide.charts)
        ordered.extend((item.order, 3, item) for item in slide.images)
        for _, _, item in sorted(ordered, key=lambda value: (value[0], value[1])):
            if isinstance(item, TextBlock):
                prefix = "  " * item.level + ("- " if item.level else "")
                lines.append(f"{prefix}{item.text}")
                for link in item.hyperlinks:
                    lines.append(f"  - Link: <{link}>")
                lines.append("")
            elif isinstance(item, TableRecord):
                lines.extend(_markdown_table(item.rows))
            elif isinstance(item, ChartRecord):
                lines.append(f"### Chart: {item.title or item.chart_type}")
                lines.append("")
                if item.categories:
                    lines.append("Categories: " + ", ".join(item.categories))
                for series in item.series:
                    values = ", ".join(
                        "" if value is None else str(value) for value in series.values
                    )
                    lines.append(f"- {series.name or 'Series'}: {values}")
                lines.append("")
            elif isinstance(item, ImageRecord):
                description = item.alt_text or "Embedded image"
                if item.asset_path:
                    lines.append(f"![{_escape_alt(description)}]({item.asset_path})")
                else:
                    lines.append(f"_[{description}; asset export disabled]_ ")
                if item.ocr_text:
                    lines.extend(["", "OCR:", "", item.ocr_text])
                lines.append("")
        if slide.notes:
            lines.extend(["### Speaker notes", "", slide.notes, ""])
    if record.warnings:
        lines.extend(["## Extraction warnings", ""])
        for issue in record.warnings:
            location = f" (slide {issue.slide_number})" if issue.slide_number else ""
            lines.append(f"- `{issue.code}`{location}: {issue.message}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_text(record: PresentationRecord) -> str:
    lines = [
        f"SOURCE: {record.source_name}",
        f"SHA256: {record.source_sha256}",
        f"SLIDES: {len(record.slides)}",
        "",
    ]
    for slide in record.slides:
        lines.append(f"=== SLIDE {slide.number}: {slide.title or 'Untitled'} ===")
        for block in sorted(slide.text_blocks, key=lambda item: (item.order, item.z_order)):
            lines.append(f"{'  ' * block.level}{block.text}")
        for table in sorted(slide.tables, key=lambda item: (item.order, item.z_order)):
            lines.append("[TABLE]")
            lines.extend(" | ".join(row) for row in table.rows)
        for chart in sorted(slide.charts, key=lambda item: (item.order, item.z_order)):
            lines.append(f"[CHART] {chart.title or chart.chart_type}")
            for series in chart.series:
                lines.append(f"{series.name}: {', '.join(map(str, series.values))}")
        for image in sorted(slide.images, key=lambda item: (item.order, item.z_order)):
            lines.append(f"[IMAGE] {image.alt_text or image.sha256[:16]}")
            if image.ocr_text:
                lines.append(image.ocr_text)
        if slide.notes:
            lines.extend(["[NOTES]", slide.notes])
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _markdown_table(rows: tuple[tuple[str, ...], ...]) -> list[str]:
    if not rows:
        return ["_[Empty table]_", ""]
    width = max(len(row) for row in rows)
    normalized = [list(row) + [""] * (width - len(row)) for row in rows]

    def escape(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    lines = ["| " + " | ".join(escape(cell) for cell in normalized[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(escape(cell) for cell in row) + " |" for row in normalized[1:])
    lines.append("")
    return lines


def _escape_alt(value: str) -> str:
    return value.replace("[", "\\[").replace("]", "\\]").replace("\n", " ")
