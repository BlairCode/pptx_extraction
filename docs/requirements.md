# Product requirements

## Product position

`pptx_extraction` converts PowerPoint material into traceable, machine-readable knowledge for search, RAG, content
migration, accessibility auditing, compliance review and Agent workflows. It is an extraction layer, not a
slide renderer or a generative rewriting system.

## Primary users and jobs

| User | Job | Required outcome |
|---|---|---|
| Knowledge engineer | Ingest decks into RAG/search | Stable JSON, page citations, deterministic IDs and no hidden network calls |
| Data/content team | Migrate large slide libraries | Batch execution, partial-failure reports and collision-safe outputs |
| Compliance/accessibility team | Audit notes, metadata, links and image descriptions | Source coordinates, warnings and optional metadata redaction |
| Developer | Embed extraction in a service | Typed Python API, CLI contract, versioned schema and structured errors |
| Agent | Inspect a deck and cite evidence | Concise Markdown/JSON with slide numbers, element kinds and asset paths |

## Functional requirements

1. Accept `.pptx`, `.pptm`, `.potx` and `.ppsx` OOXML packages. Convert legacy `.ppt/.pot/.pps`
   through an explicitly invoked LibreOffice adapter; reject PDF rather than advertising false support.
2. Extract core metadata, slide dimensions, hidden state, layout name, text hierarchy, hyperlinks, tables,
   chart titles/categories/series, speaker notes and embedded pictures.
3. Preserve provenance through slide number, shape name/id, normalized bounding box and source SHA-256.
4. Order visual elements predictably by rows and horizontal position while retaining original z-order.
5. Save media with content hashes, deduplicate repeated blobs and optionally run OCR once per unique image.
6. Export schema-versioned JSON plus readable Markdown and plain text.
7. Support inspect, validate, extract and concurrent batch workflows from a cross-platform CLI.
8. Offer an optional job-based HTTP API with bounded uploads, non-blocking processing and result polling.
9. Emit actionable warnings for unsupported shapes, missing alt text, encrypted/corrupt packages and
   extraction degradation without silently claiming success.

## Non-functional requirements

- **Safety:** reject path traversal, encrypted ZIP entries, excessive entry counts, expanded-size limits and
  suspicious compression ratios before `python-pptx` parses a file.
- **Privacy:** perform no outbound request by default; never log slide content; expose metadata redaction;
  use random service job IDs and sanitized filenames.
- **Portability:** core works on Windows, Linux and macOS without PowerPoint installed.
- **Determinism:** identical input/options produce identical semantic output; timestamps are excluded from
  content identity and image names are hash-derived.
- **Performance target:** on a modern laptop, parse a normal 100-slide/50 MB deck without OCR in under
  30 seconds and below 1 GB RSS. This is a target to benchmark, not a fabricated guarantee.
- **Maintainability:** typed modules, explicit protocols, no heavyweight import-time side effects, at least
  one integration fixture covering text/table/chart/notes/image behavior.
- **Compatibility:** schema breaking changes require a major schema version; CLI exit codes are documented.

## Deliberate exclusions for v2

- Pixel-perfect slide rendering, animation timelines, video/audio transcription and embedded-object execution.
- Handwriting/image-layout OCR beyond embedded pictures.
- Automatic LLM rewriting. Enrichment belongs downstream so extraction remains auditable.
- Native `.ppt` parsing without conversion and PDF parsing (use a dedicated converter/tool first).

## Acceptance criteria

- A synthetic deck round-trips through the pipeline with expected text, table, chart, notes and one deduped
  image in JSON and Markdown.
- Malformed/non-ZIP input, traversal entries and configured archive-limit violations fail with stable codes.
- Batch mode continues after one failed file and returns a non-zero partial-failure exit.
- The sdist/wheel, Agent Skill validation and repository privacy scan all pass in CI.
