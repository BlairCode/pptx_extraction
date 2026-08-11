# Upgrade plan and traceability

## Audit summary

The prototype mixed parsing, OCR, text generation, Flask routing and file cleanup in root scripts. It loaded
large ML models at import time, called OCR twice per image, hard-coded personal Windows paths, returned server
filesystem paths, accepted unsupported PDF input and used tests that referenced another computer. The empty
requirements file made the advertised setup unreproducible. Generated text/images/audio and a personal deck
were committed alongside source.

## File-level migration

| Legacy file | Action | Replacement |
|---|---|---|
| `main.py` | remove duplicated synchronous orchestration | `pipeline.py`, `cli.py`, `__main__.py` |
| `app.py`, `static/index.html` | replace blocking Flask demo and wildcard CORS | optional `api.py` job service |
| `modules/config.py` | remove personal absolute paths | CLI args and `ServiceSettings.from_env()` |
| `modules/ppt_text_extraction.py`, `text_extraction.py` | consolidate conflicting parsers | `extractors/pptx.py` |
| `modules/image_extraction_p.py`, `image_extraction_t.py` | replace eager vendor-specific OCR | `ocr.py` lazy backend + hash dedupe |
| `modules/ai_optimizer.py` | remove random, lossy and import-heavy rewriting | deterministic exporters; downstream enrichment boundary |
| `generate_caption.py`, `caption.txt`, audio samples | remove unrelated scope | separate future repository if needed |
| `delete.py` | remove unsafe fixed-directory deletion | atomic pipeline outputs and explicit overwrite flag |
| `requirements.txt` | replace empty file | `pyproject.toml` core/optional/dev dependency groups |
| `tests/test_*.py` | replace machine-specific print scripts | generated fixtures and behavior assertions |
| committed `output2/`, `.srt`, sample deck | remove private/runtime artifacts | ignored local output; tests synthesize fixtures |

## Delivery phases

1. **Foundation:** package metadata, models, errors, logging boundary and safe archive inspection.
2. **Extraction:** recursive grouped-shape traversal, reading order, notes/tables/charts/links/images, warnings.
3. **Interfaces:** atomic multi-format pipeline, Python API, CLI, batch execution, legacy conversion, HTTP jobs.
4. **Assurance:** unit/integration/security tests, schema, CI, lint/type/build configuration and privacy scan.
5. **Adoption:** GitHub README, architecture/security/release docs and a separately distributable Agent Skill.

Each phase is accepted only after executable tests. Optional OCR/API paths must fail with a clear installation
hint rather than breaking import of the core package.
