# HTTP API

Install `pptx-extraction[api]`, then run the reference single-node service:

```bash
uvicorn pptx_extraction.api:create_app --factory --host 127.0.0.1 --port 8000
```

- `GET /healthz` reports process health.
- `POST /v1/jobs` accepts one multipart `file`, returns `202` and a random job ID.
- `GET /v1/jobs/{id}` returns `queued`, `running`, `succeeded` or `failed`.
- `GET /v1/jobs/{id}/result` returns schema v1 JSON after success.

The service caps streamed uploads with `PPTX_EXTRACTION_MAX_UPLOAD_MB`, stores only randomized job paths and
redacts author-like metadata. It intentionally enables no wildcard CORS. Add authentication, TLS, reverse-
proxy limits, rate limiting, durable queue/object storage and lifecycle cleanup before multi-user deployment.
