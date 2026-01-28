FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

COPY --from=docker.io/astral/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . .

ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync

ENTRYPOINT ["uv", "run", "python", "-m", "scarlet.main"]

CMD ["--help"]
