# syntax=docker/dockerfile:1.7

FROM python:3.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.8.22 /uv /uvx /bin/

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Cache third-party dependencies independently from source changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra dev \
    && mkdir -p /app/data /app/logs /app/results \
    && chmod 0777 /app/data /app/logs /app/results

CMD ["bash"]

RUN groupadd -g 1000 cos && \
    useradd -u 1000 -g 1000 -m cos
USER cos