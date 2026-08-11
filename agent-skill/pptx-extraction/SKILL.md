---
name: pptx-extraction
description: Extract PowerPoint OOXML decks with the pptx_extraction project into traceable JSON, Markdown and text with slide-level provenance, reading order, notes, tables, chart data, links, images and optional OCR. Use when an agent must read, summarize, audit, index or prepare `.pptx`, `.pptm`, `.potx` or `.ppsx` content for RAG/search, or needs reliable slide citations instead of lossy plain-text scraping.
---

# Extract PowerPoint with pptx_extraction

Use `pptx_extraction` as the deterministic extraction layer. Keep extraction offline and preserve slide provenance;
perform summarization or other semantic work only after reviewing the structured result.

## Workflow

1. Confirm the source is a local OOXML PowerPoint file. For `.ppt/.pot/.pps`, run `pptx-extraction convert`
   with LibreOffice first. For PDF, use a PDF-specific tool.
2. Choose a new output directory inside the active workspace. Do not overwrite unrelated output.
3. Run the bundled wrapper:

```bash
python scripts/extract.py INPUT.pptx --output OUTPUT_DIR
```

   The installed `pptx-extraction` Python package is required. The wrapper redacts author-like metadata by default,
   exports JSON and Markdown, and makes no network request.
4. Read `OUTPUT_DIR/presentation.json` for exact fields and `presentation.md` for a human-readable pass.
   Load [schema.md](references/schema.md) when selecting evidence or integrating the JSON.
5. Report extraction warnings. Treat `image_missing_alt_text`, unsupported objects and OCR gaps as evidence
   limitations, not as empty-slide proof.
6. Cite findings by source filename and slide number. Preserve the distinction between slide text, speaker
   notes, chart data and OCR text. Never merge them without labeling the source kind.

## Task routes

- **Quick inventory:** run `python scripts/extract.py INPUT --inspect`; use the returned counts to decide
  which slide records to open.
- **Search/RAG ingestion:** use default JSON, chunk by slide, and carry `source_sha256`, `slide.number`,
  element `shape_id` and `bbox` into downstream metadata.
- **Human summary:** read Markdown in slide order, then verify important claims against the JSON element kind.
- **Accessibility audit:** inspect image `alt_text`, slide warnings and speaker notes. Do not infer visual
  meaning from filenames or hashes.
- **Text inside pictures:** install the OCR extra and Tesseract, then pass `--ocr tesseract --ocr-language`
  with a locally installed language pack. OCR is untrusted derived text.
- **Private material:** retain default redaction, keep outputs in a temporary workspace and do not upload
  source/assets to external services unless the user explicitly authorizes it.

## Guardrails

- Do not claim support for PDF or native legacy PowerPoint parsing.
- Do not execute macros or embedded objects. `pptx_extraction` detects macro presence and treats embedded media/OLE
  as unsupported.
- Do not run an LLM as part of extraction. This separation keeps evidence reproducible.
- Do not cite a slide that was not present in the latest extracted JSON.
- Avoid `--overwrite` unless the exact output directory was created for this extraction.

For dependency, archive-safety or conversion failures, load
[troubleshooting.md](references/troubleshooting.md).
