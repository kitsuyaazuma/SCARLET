FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . .

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENTRYPOINT ["PYTHON_GIL=0", "uv", "run", "python", "main.py"]

CMD ["--help"]
