# Contributing

Use Python 3.10 or newer. Create a virtual environment, install `-e ".[dev]"`, then run:

```bash
ruff check .
ruff format --check .
mypy src/pptx_extraction
pytest
```

Keep extraction deterministic and offline by default. New output fields must remain backward compatible
within the current schema major version. Add a generated fixture or focused unit test for every parser fix;
never commit private decks, extracted media, credentials or machine-specific paths.

Open a small issue before large changes. Use conventional, imperative commit subjects and include the
observable behavior and test evidence in pull requests.
