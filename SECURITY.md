# Security policy

## Supported versions

Security fixes are provided for the latest `2.x` release.

## Reporting

Do not open public issues for suspected vulnerabilities. Use GitHub private vulnerability reporting on the
repository's **Security** tab. Include a minimal reproduction, affected version and impact. Do not attach a
confidential presentation; use a synthetic deck where possible.

## Processing model

`pptx_extraction` validates ZIP paths, entry counts, expanded size and compression ratios before parsing OOXML.
It never executes macros and makes no network request in the default installation. Treat extracted text,
notes, links and images as untrusted data. Run the optional API behind authentication, TLS, request-rate
limits and an isolated worker in production. The bundled service is a reference single-node deployment,
not a multi-tenant security boundary.
