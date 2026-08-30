# This image is intentionally small. Extend it with only the trusted toolchains
# required by a repository, then build and review it locally before use.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# The runtime also enforces a numeric non-root user, dropped capabilities,
# no-new-privileges, a read-only root filesystem, and resource limits.
USER 65532:65532
