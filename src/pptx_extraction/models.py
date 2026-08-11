"""Serializable records forming the pptx_extraction schema v1 contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SCHEMA_VERSION = "1.0"


@dataclass(frozen=True, slots=True)
class ExtractionOptions:
    include_assets: bool = True
    include_notes: bool = True
    include_metadata: bool = True
    redact_metadata: bool = False
    ocr_backend: str = "none"
    ocr_language: str = "eng"


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left_pt: float
    top_pt: float
    width_pt: float
    height_pt: float
    left_ratio: float
    top_ratio: float
    width_ratio: float
    height_ratio: float

    @classmethod
    def from_emu(
        cls,
        left: int,
        top: int,
        width: int,
        height: int,
        slide_width: int,
        slide_height: int,
    ) -> BoundingBox:
        emu_per_point = 12_700
        return cls(
            left_pt=round(left / emu_per_point, 3),
            top_pt=round(top / emu_per_point, 3),
            width_pt=round(width / emu_per_point, 3),
            height_pt=round(height / emu_per_point, 3),
            left_ratio=round(left / slide_width, 6) if slide_width else 0.0,
            top_ratio=round(top / slide_height, 6) if slide_height else 0.0,
            width_ratio=round(width / slide_width, 6) if slide_width else 0.0,
            height_ratio=round(height / slide_height, 6) if slide_height else 0.0,
        )


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    message: str
    severity: str = "warning"
    slide_number: int | None = None
    shape_id: int | None = None


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str
    kind: str
    level: int
    order: int
    z_order: int
    shape_id: int
    shape_name: str
    bbox: BoundingBox
    hyperlinks: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TableRecord:
    rows: tuple[tuple[str, ...], ...]
    order: int
    z_order: int
    shape_id: int
    shape_name: str
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class ChartSeries:
    name: str
    values: tuple[float | int | str | None, ...]


@dataclass(frozen=True, slots=True)
class ChartRecord:
    chart_type: str
    title: str | None
    categories: tuple[str, ...]
    series: tuple[ChartSeries, ...]
    order: int
    z_order: int
    shape_id: int
    shape_name: str
    bbox: BoundingBox


@dataclass(frozen=True, slots=True)
class ImageRecord:
    sha256: str
    media_type: str
    asset_path: str | None
    alt_text: str | None
    ocr_text: str | None
    order: int
    z_order: int
    shape_id: int
    shape_name: str
    bbox: BoundingBox


@dataclass(slots=True)
class SlideRecord:
    number: int
    title: str | None
    hidden: bool
    layout_name: str | None
    text_blocks: list[TextBlock] = field(default_factory=list)
    tables: list[TableRecord] = field(default_factory=list)
    charts: list[ChartRecord] = field(default_factory=list)
    images: list[ImageRecord] = field(default_factory=list)
    notes: str | None = None


@dataclass(slots=True)
class PresentationRecord:
    source_name: str
    source_sha256: str
    source_size_bytes: int
    slide_width_pt: float
    slide_height_pt: float
    metadata: dict[str, Any]
    slides: list[SlideRecord]
    warnings: list[Issue]
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def summary(self) -> dict[str, int | str]:
        return {
            "schema_version": self.schema_version,
            "slides": len(self.slides),
            "text_blocks": sum(len(slide.text_blocks) for slide in self.slides),
            "tables": sum(len(slide.tables) for slide in self.slides),
            "charts": sum(len(slide.charts) for slide in self.slides),
            "images": sum(len(slide.images) for slide in self.slides),
            "warnings": len(self.warnings),
        }
