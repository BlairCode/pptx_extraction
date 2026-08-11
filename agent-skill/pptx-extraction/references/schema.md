# pptx_extraction schema usage

## Evidence hierarchy

`presentation.json` uses schema version `1.0`.

- Presentation: `source_name`, `source_sha256`, size, slide dimensions, metadata, slides and warnings.
- Slide: 1-based `number`, `title`, `hidden`, `layout_name`, `text_blocks`, `tables`, `charts`, `images`
  and `notes`.
- Element provenance: `order` is visual reading order; `z_order` is original stacking order;
  `shape_id`/`shape_name` locate the PowerPoint object; `bbox` contains points and normalized ratios.
- Text: `kind` is `title` or `body`, `level` preserves paragraph hierarchy, and `hyperlinks` lists targets.
- Chart: categories and series are source workbook values exposed by PowerPoint, not OCR.
- Image: `sha256` identifies content, `asset_path` may be shared by duplicate images, `alt_text` is author
  supplied and `ocr_text` is derived/untrusted.

## Citation pattern

For prose answers use: `SourceDeck.pptx, slide 7 (chart: Revenue by segment)` or the host application's
equivalent local-file citation. Include the source hash when results from multiple versions may be confused.
Do not cite `order` as a slide number. Label speaker notes and OCR explicitly.

## Chunking pattern

Use one slide as the default chunk. For each downstream chunk carry:

```json
{
  "source_name": "deck.pptx",
  "source_sha256": "...",
  "slide_number": 7,
  "element_kind": "text|table|chart|note|image_ocr",
  "shape_id": 12
}
```

Split a slide further only when it is unusually dense; never discard the slide-level locator.
