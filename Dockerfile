# Spec-to-Sim: web app + MuJoCo simulator in one CPU-only container.
# Rendering uses OSMesa (software OpenGL), so no GPU is needed. Shadows are off because they
# make software rendering ~4x slower (see src/spec2sim/sim/render.py). No MP4 is made on the
# server: the browser draws the 3D replay, and only ~3 frames/s are rendered for the Critic.
# Build for Nebius:  docker buildx build --platform linux/amd64 -t <registry>/spec2sim:<tag> .
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa \
    SPEC2SIM_SHADOWS=0 \
    SPEC2SIM_VIDEO=0 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
 && apt-get install -y --no-install-recommends libosmesa6 libgl1 libglib2.0-0 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Robot model first: changes rarely, so this layer stays cached.
COPY scripts/fetch_menagerie.sh scripts/
RUN ./scripts/fetch_menagerie.sh

# Dependencies next (cached unless the lockfile changes), then the code.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY app ./app
COPY scripts/eval.py scripts/generate_data.py ./scripts/
RUN uv sync --frozen --no-dev

# Shared, pre-built web facts (material friction/bounciness, Franka limits): fresh containers
# don't repeat those Tavily searches. Per-object searches are cached at run time.
COPY .cache/physics ./.cache/physics

# Run as a non-root user (uid 1000, as Hugging Face Spaces does) that owns the app's writable
# folders: runs/, .cache/ (search cache, spend log) and app/.viz (browser meshes).
RUN useradd -m -u 1000 user && mkdir -p /app/runs /app/app/.viz && chown -R user:user /app
USER user

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
