# pptx_extraction changelog

All notable changes follow [Keep a Changelog](https://keepachangelog.com/) and semantic versioning.

## [2.0.0] - 2026-08-11

### Added

- Structured extraction for text, tables, chart data, speaker notes, hyperlinks, images and metadata.
- JSON, Markdown and plain-text exporters with a versioned schema.
- Safe OOXML validation, deterministic image deduplication and optional Tesseract OCR.
- Cross-platform CLI, concurrent batch mode, optional LibreOffice legacy conversion and job-based API.
- Unit/integration tests, CI, security policy, architecture documentation and Agent Skill packaging.

### Changed

- Rebuilt the prototype as a typed `src/` package with explicit domain models and error boundaries.
- Replaced import-time Spacy/Transformers/PaddleOCR loading with deterministic offline defaults.

### Removed

- Hard-coded personal paths, personal contact details and committed user-generated outputs.
- Misleading PDF support and the unrelated audio-caption experiment from the core package.
