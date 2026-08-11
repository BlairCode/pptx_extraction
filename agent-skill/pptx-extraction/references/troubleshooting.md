# Troubleshooting

## Unsupported format

- `.pptx/.pptm/.potx/.ppsx`: process directly.
- `.ppt/.pot/.pps`: install LibreOffice and run `pptx-extraction convert INPUT.ppt -o CONVERTED_DIR`.
- `.pdf`: route to a PDF extraction tool.

## Optional dependency errors

Install the project first (`python -m pip install .` from its source checkout). For OCR install
`pptx-extraction[ocr]` plus the native Tesseract executable and requested language pack. OCR remains optional;
normal PowerPoint text never needs it.

## Unsafe package

`pptx_extraction` rejects encrypted entries, traversal paths, excessive archive entries, expanded-size limits and
suspicious compression ratios. Do not bypass these checks for an untrusted file. If a trusted large deck
exceeds defaults, use the Python API with a narrowly increased `PackageLimits` value and document why.

## Missing content

- SmartArt, equations, diagrams, embedded spreadsheets, OLE objects, audio, video and animations may not
  expose semantic content through `python-pptx`.
- Image-only slides require rendering or slide-level OCR from a presentation/PDF tool; embedded-image OCR
  does not render the whole slide.
- Hidden slides are extracted and marked `hidden: true`.
- Review the `warnings` array before concluding that a slide is empty.
